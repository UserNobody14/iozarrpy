//! Direct resolution and materialization of lazy selections.
//!
//! Resolves `LazyDimConstraint` values directly via the backend (binary search
//! on coordinate arrays) rather than through a request/cache indirection.
//! Coordinate chunk I/O is already cached by the Moka layer in the backend.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use smallvec::SmallVec;
use zarrs::array::ArraySubset;

use super::grouped_selection::GroupedSelection;
use super::lazy_selection::{
    LazyArraySelection, LazyDimConstraint,
    LazyHyperRectangle,
};
use super::resolver_traits::{
    DimResolutionCtx, ResolutionError,
};
use super::selection::{
    DataArraySelection, DatasetSelection,
    Emptyable, SetOperations,
};
use super::types::{
    DimSignature, ValueRangePresent,
};
use crate::IStr;
use crate::chunk_plan::exprs::expr_plan::{
    ExprPlan, VarSet,
};
use crate::chunk_plan::indexing::selection::ArraySubsetList;
use crate::meta::ZarrMeta;

use super::types::CoordScalar;

// ============================================================================
// Pure binary-search helpers (no I/O)
// ============================================================================

use std::cmp::Ordering as Ord;

#[inline(always)]
fn lower_bound_should_go_left(
    dir: Ord,
    strict: bool,
    cmp: Option<Ord>,
) -> bool {
    matches!(
        (dir, strict, cmp),
        (
            Ord::Less | Ord::Equal,
            false,
            Some(Ord::Greater | Ord::Equal)
        ) | (Ord::Less, true, Some(Ord::Greater))
            | (
                Ord::Greater | Ord::Equal,
                false,
                Some(Ord::Less | Ord::Equal)
            )
            | (
                Ord::Greater,
                true,
                Some(Ord::Less)
            )
    )
}

#[inline(always)]
fn upper_bound_should_go_right(
    dir: Ord,
    strict: bool,
    cmp: Option<Ord>,
) -> bool {
    matches!(
        (dir, strict, cmp),
        (
            Ord::Less | Ord::Equal,
            false,
            Some(Ord::Less | Ord::Equal)
        ) | (Ord::Less, true, Some(Ord::Less))
            | (
                Ord::Greater | Ord::Equal,
                false,
                Some(Ord::Greater | Ord::Equal)
            )
            | (
                Ord::Greater,
                true,
                Some(Ord::Greater)
            )
    )
}

fn monotonic_sample_indices(
    n: u64,
    chunk_size: u64,
) -> [u64; 5] {
    let mut samples = [
        0u64,
        chunk_size.saturating_sub(1).min(n - 1),
        chunk_size.min(n - 1),
        (n / 2).min(n - 1),
        n - 1,
    ];
    samples.sort();
    samples
}

#[inline(always)]
fn monotonic_ord_matches(
    dir: Ord,
    ord: Option<Ord>,
) -> bool {
    matches!(
        (dir, ord),
        (Ord::Less, Some(Ord::Less | Ord::Equal))
            | (
                Ord::Greater,
                Some(Ord::Greater | Ord::Equal)
            )
    )
}

fn check_monotonic_from_samples(
    first: &CoordScalar,
    last: &CoordScalar,
    samples: &[CoordScalar],
) -> Option<Ord> {
    let dir = match first.partial_cmp(last) {
        Some(Ord::Less | Ord::Equal) => Ord::Less,
        Some(Ord::Greater) => Ord::Greater,
        None => return None,
    };
    let mut prev = None;
    for v in samples {
        if let Some(p) = prev {
            if !monotonic_ord_matches(
                dir,
                CoordScalar::partial_cmp(p, v),
            ) {
                return None;
            }
        }
        prev = Some(v);
    }
    Some(dir)
}

#[inline(always)]
fn should_go_right(
    target: &CoordScalar,
    v: &CoordScalar,
    dir: Ord,
    strict: bool,
    is_upper: bool,
) -> bool {
    let cmp = v.partial_cmp(target);
    if is_upper {
        upper_bound_should_go_right(
            dir, strict, cmp,
        )
    } else {
        !lower_bound_should_go_left(
            dir, strict, cmp,
        )
    }
}

// ============================================================================
// Sync resolution
// ============================================================================

use crate::shared::ChunkedDataBackendSync;

fn scalar_at_sync<B: ChunkedDataBackendSync>(
    backend: &B,
    dim: &IStr,
    idx: u64,
    n: u64,
    chunk_size: u64,
    time_enc: Option<&crate::meta::TimeEncoding>,
) -> Option<CoordScalar> {
    if idx >= n {
        return None;
    }
    let chunk_idx = idx / chunk_size;
    let offset = (idx % chunk_size) as usize;
    let chunk = backend
        .read_chunk_sync(dim, &[chunk_idx])
        .ok()?;
    chunk.get_i64(offset).map(|raw| {
        crate::chunk_plan::apply_time_encoding(
            raw, time_enc,
        )
    })
}

fn check_monotonicity_sync<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    ctx: &DimResolutionCtx,
) -> Option<Ord> {
    if ctx.n < 2 {
        return Some(Ord::Less);
    }
    let te = ctx.time_enc.as_ref();
    let ap = &ctx.array_path;
    let first = scalar_at_sync(
        backend,
        ap,
        0,
        ctx.n,
        ctx.chunk_size,
        te,
    )?;
    let last = scalar_at_sync(
        backend,
        ap,
        ctx.n - 1,
        ctx.n,
        ctx.chunk_size,
        te,
    )?;
    let indices = monotonic_sample_indices(
        ctx.n,
        ctx.chunk_size,
    );
    let mut samples =
        Vec::with_capacity(indices.len());
    for &i in &indices {
        samples.push(scalar_at_sync(
            backend,
            ap,
            i,
            ctx.n,
            ctx.chunk_size,
            te,
        )?);
    }
    check_monotonic_from_samples(
        &first, &last, &samples,
    )
}

fn sync_binary_search<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    ctx: &DimResolutionCtx,
    target: &CoordScalar,
    strict: bool,
    dir: Ord,
    is_upper: bool,
) -> Option<u64> {
    let te = ctx.time_enc.as_ref();
    let ap = &ctx.array_path;
    let mut lo = 0u64;
    let mut hi = ctx.n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let v = scalar_at_sync(
            backend,
            ap,
            mid,
            ctx.n,
            ctx.chunk_size,
            te,
        )?;
        if should_go_right(
            target, &v, dir, strict, is_upper,
        ) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

fn resolve_value_range_sync<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    ctx: &DimResolutionCtx,
    vr: &ValueRangePresent,
    dir: Ord,
) -> Option<Range<u64>> {
    use std::ops::Bound;
    let start = match &vr.0 {
        Bound::Included(s)
        | Bound::Excluded(s) => {
            let strict = matches!(
                &vr.0,
                Bound::Excluded(_)
            );
            sync_binary_search(
                backend, ctx, s, strict, dir,
                false,
            )?
        }
        Bound::Unbounded => 0,
    };
    let end = match &vr.1 {
        Bound::Included(s)
        | Bound::Excluded(s) => {
            let strict = matches!(
                &vr.1,
                Bound::Excluded(_)
            );
            sync_binary_search(
                backend, ctx, s, strict, dir,
                true,
            )?
        }
        Bound::Unbounded => ctx.n,
    };
    Some(start..end)
}

/// Resolve a single `LazyDimConstraint` to index ranges synchronously.
fn resolve_constraint_sync<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    dim: &IStr,
    constraint: &LazyDimConstraint,
    dim_range: Range<u64>,
    meta: &ZarrMeta,
) -> Result<Vec<Range<u64>>, ResolutionError> {
    match constraint {
        LazyDimConstraint::All => {
            Ok(vec![dim_range])
        }
        LazyDimConstraint::Empty => Ok(vec![]),
        LazyDimConstraint::Resolved(rl) => {
            Ok(vec![rl.clone()])
        }
        LazyDimConstraint::Unresolved(vr) => {
            if let Some(r) =
                try_resolve_index_only(
                    dim,
                    meta,
                    dim_range.end,
                    vr,
                )
            {
                return Ok(vec![r]);
            }
            let Some(ctx) =
                DimResolutionCtx::from_meta(
                    dim, meta,
                )
            else {
                return Ok(vec![dim_range]);
            };
            let Some(dir) =
                check_monotonicity_sync(
                    backend, &ctx,
                )
            else {
                return Ok(vec![dim_range]);
            };
            Ok(vec![resolve_value_range_sync(
                backend, &ctx, vr, dir,
            )
            .unwrap_or(dim_range)])
        }
        LazyDimConstraint::InterpolationRange(
            vr,
        ) => {
            if let Some(r) =
                try_resolve_index_only(
                    dim,
                    meta,
                    dim_range.end,
                    vr,
                )
            {
                let start =
                    r.start.saturating_sub(1);
                let end =
                    r.end.saturating_add(1);
                return Ok(vec![start..end]);
            }
            let Some(ctx) =
                DimResolutionCtx::from_meta(
                    dim, meta,
                )
            else {
                return Ok(vec![dim_range]);
            };
            let Some(dir) =
                check_monotonicity_sync(
                    backend, &ctx,
                )
            else {
                return Ok(vec![dim_range]);
            };
            let r = resolve_value_range_sync(
                backend, &ctx, vr, dir,
            )
            .unwrap_or(dim_range.clone());
            let start = r.start.saturating_sub(1);
            let end = r.end.saturating_add(1);
            Ok(vec![start..end])
        }
        LazyDimConstraint::InterpolationPoints(
            points,
        ) => {
            let Some(ctx) =
                DimResolutionCtx::from_meta(
                    dim, meta,
                )
            else {
                return Ok(vec![dim_range]);
            };
            let Some(dir) =
                check_monotonicity_sync(
                    backend, &ctx,
                )
            else {
                return Ok(vec![dim_range]);
            };
            let mut ranges = Vec::with_capacity(
                points.len(),
            );
            for point in points.iter() {
                let vr_max = ValueRangePresent::from_max_exclusive(point.clone());
                let vr_min = ValueRangePresent::from_min_inclusive(point.clone());
                let left_r =
                    resolve_value_range_sync(
                        backend, &ctx, &vr_max,
                        dir,
                    )
                    .unwrap_or(dim_range.clone());
                let right_r =
                    resolve_value_range_sync(
                        backend, &ctx, &vr_min,
                        dir,
                    )
                    .unwrap_or(dim_range.clone());
                let left_idx = left_r
                    .end
                    .saturating_sub(1);
                let right_idx = right_r.start;
                let start =
                    left_idx.min(right_idx);
                let end_exclusive = (left_idx
                    .max(right_idx)
                    + 1)
                .min(dim_range.end);
                ranges
                    .push(start..end_exclusive);
            }
            Ok(merge_ranges(ranges))
        }
    }
}

/// Resolve a `LazyArraySelection` to a `DataArraySelection` synchronously.
fn resolve_array_sync<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    selection: &LazyArraySelection,
    dims: &SmallVec<[IStr; 4]>,
    shape: &[u64],
    meta: &ZarrMeta,
) -> Result<DataArraySelection, ResolutionError> {
    match selection {
        LazyArraySelection::Rectangles(rects) => {
            let mut out = ArraySubsetList::new();
            for rect in rects {
                let materialized =
                    resolve_rectangle_sync(
                        backend, rect, dims,
                        shape, meta,
                    )?;
                out = out.union(&materialized);
            }
            Ok(DataArraySelection::from_subsets(
                dims, out,
            ))
        }
        LazyArraySelection::Difference(a, b) => {
            let a_mat = resolve_array_sync(
                backend, a, dims, shape, meta,
            )?;
            let b_mat = resolve_array_sync(
                backend, b, dims, shape, meta,
            )?;
            Ok(a_mat.difference(&b_mat))
        }
        LazyArraySelection::Union(a, b) => {
            let a_mat = resolve_array_sync(
                backend, a, dims, shape, meta,
            )?;
            let b_mat = resolve_array_sync(
                backend, b, dims, shape, meta,
            )?;
            Ok(a_mat.union(&b_mat))
        }
    }
}

fn resolve_rectangle_sync<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    rect: &LazyHyperRectangle,
    dims: &SmallVec<[IStr; 4]>,
    shape: &[u64],
    meta: &ZarrMeta,
) -> Result<ArraySubsetList, ResolutionError> {
    if rect.is_empty() {
        return Ok(ArraySubsetList::empty());
    }
    let mut current_subsets: Vec<
        Vec<Range<u64>>,
    > = vec![
        (0..dims.len())
            .map(|i| 0..shape[i])
            .collect(),
    ];

    for (dim, constraint) in rect.dims() {
        let dim_idx_option =
            dims.iter().position(|d| d == dim);
        if let Some(dim_idx) = dim_idx_option {
            let range_list =
                resolve_constraint_sync(
                    backend,
                    dim,
                    constraint,
                    0..shape[dim_idx],
                    meta,
                )?;
            if range_list.is_empty() {
                return Ok(
                    ArraySubsetList::empty(),
                );
            }
            let mut new_subsets = Vec::new();
            for subset in current_subsets.iter() {
                for range in range_list.iter() {
                    if !range.is_empty() {
                        let mut new_subset =
                            subset.clone();
                        new_subset[dim_idx] =
                            range.clone();
                        new_subsets
                            .push(new_subset);
                    }
                }
            }
            if new_subsets.is_empty() {
                return Ok(
                    ArraySubsetList::empty(),
                );
            }
            current_subsets = new_subsets;
        }
    }
    let mut out_list = ArraySubsetList::new();
    for subset in current_subsets {
        out_list.push(
            ArraySubset::new_with_ranges(&subset),
        );
    }
    Ok(out_list)
}

// ============================================================================
// Async resolution
// ============================================================================

use crate::shared::ChunkedDataBackendAsync;

async fn scalar_at_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    dim: &IStr,
    idx: u64,
    n: u64,
    chunk_size: u64,
    time_enc: Option<&crate::meta::TimeEncoding>,
) -> Option<CoordScalar> {
    if idx >= n {
        return None;
    }
    let chunk_idx = idx / chunk_size;
    let offset = (idx % chunk_size) as usize;
    let chunk = backend
        .read_chunk_async(dim, &[chunk_idx])
        .await
        .ok()?;
    chunk.get_i64(offset).map(|raw| {
        crate::chunk_plan::apply_time_encoding(
            raw, time_enc,
        )
    })
}

async fn check_monotonicity_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    ctx: &DimResolutionCtx,
) -> Option<Ord> {
    if ctx.n < 2 {
        return Some(Ord::Less);
    }
    let te = ctx.time_enc.as_ref();
    let ap = &ctx.array_path;
    let indices = monotonic_sample_indices(
        ctx.n,
        ctx.chunk_size,
    );

    // Fetch all sample points concurrently (first, last, and 5 samples)
    let first_fut = scalar_at_async(
        backend,
        ap,
        0,
        ctx.n,
        ctx.chunk_size,
        te,
    );
    let last_fut = scalar_at_async(
        backend,
        ap,
        ctx.n - 1,
        ctx.n,
        ctx.chunk_size,
        te,
    );
    let sample_futs: Vec<_> = indices
        .iter()
        .map(|&i| {
            scalar_at_async(
                backend,
                ap,
                i,
                ctx.n,
                ctx.chunk_size,
                te,
            )
        })
        .collect();

    let (first, last, sample_results) = tokio::join!(
        first_fut,
        last_fut,
        futures::future::join_all(sample_futs)
    );

    let first = first?;
    let last = last?;
    let samples: Option<Vec<CoordScalar>> =
        sample_results.into_iter().collect();
    let samples = samples?;

    check_monotonic_from_samples(
        &first, &last, &samples,
    )
}

async fn async_binary_search<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    ctx: &DimResolutionCtx,
    target: &CoordScalar,
    strict: bool,
    dir: Ord,
    is_upper: bool,
) -> Option<u64> {
    let te = ctx.time_enc.as_ref();
    let ap = &ctx.array_path;
    let mut lo = 0u64;
    let mut hi = ctx.n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let v = scalar_at_async(
            backend,
            ap,
            mid,
            ctx.n,
            ctx.chunk_size,
            te,
        )
        .await?;
        if should_go_right(
            target, &v, dir, strict, is_upper,
        ) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

async fn resolve_value_range_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    ctx: &DimResolutionCtx,
    vr: &ValueRangePresent,
    dir: Ord,
) -> Option<Range<u64>> {
    use std::ops::Bound;
    let start = match &vr.0 {
        Bound::Included(s)
        | Bound::Excluded(s) => {
            let strict = matches!(
                &vr.0,
                Bound::Excluded(_)
            );
            async_binary_search(
                backend, ctx, s, strict, dir,
                false,
            )
            .await?
        }
        Bound::Unbounded => 0,
    };
    let end = match &vr.1 {
        Bound::Included(s)
        | Bound::Excluded(s) => {
            let strict = matches!(
                &vr.1,
                Bound::Excluded(_)
            );
            async_binary_search(
                backend, ctx, s, strict, dir,
                true,
            )
            .await?
        }
        Bound::Unbounded => ctx.n,
    };
    Some(start..end)
}

async fn resolve_constraint_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    dim: &IStr,
    constraint: &LazyDimConstraint,
    dim_range: Range<u64>,
    meta: &ZarrMeta,
) -> Result<Vec<Range<u64>>, ResolutionError> {
    match constraint {
        LazyDimConstraint::All => {
            Ok(vec![dim_range])
        }
        LazyDimConstraint::Empty => Ok(vec![]),
        LazyDimConstraint::Resolved(rl) => {
            Ok(vec![rl.clone()])
        }
        LazyDimConstraint::Unresolved(vr) => {
            if let Some(r) =
                try_resolve_index_only(
                    dim,
                    meta,
                    dim_range.end,
                    vr,
                )
            {
                return Ok(vec![r]);
            }
            let Some(ctx) =
                DimResolutionCtx::from_meta(
                    dim, meta,
                )
            else {
                return Ok(vec![dim_range]);
            };
            let Some(dir) =
                check_monotonicity_async(
                    backend, &ctx,
                )
                .await
            else {
                return Ok(vec![dim_range]);
            };
            Ok(vec![resolve_value_range_async(
                backend, &ctx, vr, dir,
            )
            .await
            .unwrap_or(dim_range)])
        }
        LazyDimConstraint::InterpolationRange(
            vr,
        ) => {
            if let Some(r) =
                try_resolve_index_only(
                    dim,
                    meta,
                    dim_range.end,
                    vr,
                )
            {
                let start =
                    r.start.saturating_sub(1);
                let end =
                    r.end.saturating_add(1);
                return Ok(vec![start..end]);
            }
            let Some(ctx) =
                DimResolutionCtx::from_meta(
                    dim, meta,
                )
            else {
                return Ok(vec![dim_range]);
            };
            let Some(dir) =
                check_monotonicity_async(
                    backend, &ctx,
                )
                .await
            else {
                return Ok(vec![dim_range]);
            };
            let r = resolve_value_range_async(
                backend, &ctx, vr, dir,
            )
            .await
            .unwrap_or(dim_range.clone());
            let start = r.start.saturating_sub(1);
            let end = r.end.saturating_add(1);
            Ok(vec![start..end])
        }
        LazyDimConstraint::InterpolationPoints(
            points,
        ) => {
            let Some(ctx) =
                DimResolutionCtx::from_meta(
                    dim, meta,
                )
            else {
                return Ok(vec![dim_range]);
            };
            let Some(dir) =
                check_monotonicity_async(
                    backend, &ctx,
                )
                .await
            else {
                return Ok(vec![dim_range]);
            };
            let mut ranges = Vec::with_capacity(
                points.len(),
            );
            for point in points.iter() {
                let vr_max = ValueRangePresent::from_max_exclusive(point.clone());
                let vr_min = ValueRangePresent::from_min_inclusive(point.clone());
                let left_r =
                    resolve_value_range_async(
                        backend, &ctx, &vr_max,
                        dir,
                    )
                    .await
                    .unwrap_or(dim_range.clone());
                let right_r =
                    resolve_value_range_async(
                        backend, &ctx, &vr_min,
                        dir,
                    )
                    .await
                    .unwrap_or(dim_range.clone());
                let left_idx = left_r
                    .end
                    .saturating_sub(1);
                let right_idx = right_r.start;
                let start =
                    left_idx.min(right_idx);
                let end_exclusive = (left_idx
                    .max(right_idx)
                    + 1)
                .min(dim_range.end);
                ranges
                    .push(start..end_exclusive);
            }
            Ok(merge_ranges(ranges))
        }
    }
}

fn resolve_array_async<
    'a,
    B: ChunkedDataBackendAsync,
>(
    backend: &'a B,
    selection: &'a LazyArraySelection,
    dims: &'a SmallVec<[IStr; 4]>,
    shape: &'a [u64],
    meta: &'a ZarrMeta,
) -> std::pin::Pin<
    Box<
        dyn std::future::Future<
                Output = Result<
                    DataArraySelection,
                    ResolutionError,
                >,
            > + Send
            + 'a,
    >,
> {
    Box::pin(async move {
        match selection {
            LazyArraySelection::Rectangles(
                rects,
            ) => {
                let mut out =
                    ArraySubsetList::new();
                for rect in rects {
                    let materialized =
                        resolve_rectangle_async(
                            backend, rect, dims,
                            shape, meta,
                        )
                        .await?;
                    out =
                        out.union(&materialized);
                }
                Ok(
                    DataArraySelection::from_subsets(
                        dims, out,
                    ),
                )
            }
            LazyArraySelection::Difference(
                a,
                b,
            ) => {
                let a_mat = resolve_array_async(
                    backend, a, dims, shape, meta,
                )
                .await?;
                let b_mat = resolve_array_async(
                    backend, b, dims, shape, meta,
                )
                .await?;
                Ok(a_mat.difference(&b_mat))
            }
            LazyArraySelection::Union(a, b) => {
                let a_mat = resolve_array_async(
                    backend, a, dims, shape, meta,
                )
                .await?;
                let b_mat = resolve_array_async(
                    backend, b, dims, shape, meta,
                )
                .await?;
                Ok(a_mat.union(&b_mat))
            }
        }
    })
}

async fn resolve_rectangle_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    rect: &LazyHyperRectangle,
    dims: &SmallVec<[IStr; 4]>,
    shape: &[u64],
    meta: &ZarrMeta,
) -> Result<ArraySubsetList, ResolutionError> {
    if rect.is_empty() {
        return Ok(ArraySubsetList::empty());
    }

    // Collect all constrained dimensions and resolve them concurrently
    let constraint_futs: Vec<_> = rect
        .dims()
        .filter_map(|(dim, constraint)| {
            let dim_idx = dims
                .iter()
                .position(|d| d == dim)?;
            Some(async move {
                let range_list =
                    resolve_constraint_async(
                        backend,
                        dim,
                        constraint,
                        0..shape[dim_idx],
                        meta,
                    )
                    .await?;
                Ok::<_, ResolutionError>((
                    dim_idx, range_list,
                ))
            })
        })
        .collect();

    let resolved_dims =
        futures::future::join_all(
            constraint_futs,
        )
        .await;

    // Assemble the cross-product from resolved dimensions
    let mut current_subsets: Vec<
        Vec<Range<u64>>,
    > = vec![
        (0..dims.len())
            .map(|i| 0..shape[i])
            .collect(),
    ];

    for result in resolved_dims {
        let (dim_idx, range_list) = result?;
        if range_list.is_empty() {
            return Ok(ArraySubsetList::empty());
        }
        let mut new_subsets = Vec::new();
        for subset in current_subsets.iter() {
            for range in range_list.iter() {
                if !range.is_empty() {
                    let mut new_subset =
                        subset.clone();
                    new_subset[dim_idx] =
                        range.clone();
                    new_subsets.push(new_subset);
                }
            }
        }
        if new_subsets.is_empty() {
            return Ok(ArraySubsetList::empty());
        }
        current_subsets = new_subsets;
    }
    let mut out_list = ArraySubsetList::new();
    for subset in current_subsets {
        out_list.push(
            ArraySubset::new_with_ranges(&subset),
        );
    }
    Ok(out_list)
}

// ============================================================================
// ExprPlan -> DatasetSelection (top-level entry points)
// ============================================================================

/// Get shape for a dimension signature by looking up dimension lengths from metadata.
fn dims_to_shape(
    dims: &[IStr],
    meta: &ZarrMeta,
) -> Result<Arc<[u64]>, ResolutionError> {
    let dim_shape_reduced: Option<Vec<u64>> =
        dims.iter()
            .map(|dim| {
                meta.dim_analysis
                    .dim_lengths
                    .get(dim)
                    .copied()
            })
            .collect();

    if let Some(reduced) = dim_shape_reduced {
        return Ok(reduced.into());
    }
    let mut shape =
        Vec::with_capacity(dims.len());
    for dim in dims {
        if let Some(coord_array) =
            meta.array_by_path(dim)
        {
            if let Some(&len) =
                coord_array.shape.first()
            {
                shape.push(len);
            } else {
                return Err(
                    ResolutionError::Unresolvable(
                        format!(
                            "dimension '{}' has no shape",
                            dim
                        ),
                    ),
                );
            }
        } else {
            return Err(
                ResolutionError::Unresolvable(
                    format!(
                        "cannot determine shape for dimension '{}'",
                        dim
                    ),
                ),
            );
        }
    }
    Ok(shape.into())
}

/// Build the var list and grouped signature map from an ExprPlan.
fn build_var_grouping(
    vars: &VarSet,
    meta: &ZarrMeta,
) -> Option<(
    Vec<IStr>,
    BTreeMap<Arc<DimSignature>, Vec<IStr>>,
    BTreeMap<IStr, Arc<DimSignature>>,
)> {
    let var_list: Vec<IStr> = match vars {
        VarSet::Specific(v) if v.is_empty() => {
            return None;
        }
        VarSet::All => {
            meta.all_array_paths().to_vec()
        }
        VarSet::Specific(v) => v.to_vec(),
    };

    let mut sig_cache: BTreeMap<
        DimSignature,
        Arc<DimSignature>,
    > = BTreeMap::new();
    let mut by_sig: BTreeMap<
        Arc<DimSignature>,
        Vec<IStr>,
    > = BTreeMap::new();
    let mut var_to_sig: BTreeMap<
        IStr,
        Arc<DimSignature>,
    > = BTreeMap::new();

    for var in &var_list {
        let sig = if let Some(array_meta) =
            meta.array_by_path(var.clone())
        {
            DimSignature::from_dims_only(
                array_meta.dims.clone(),
            )
        } else {
            DimSignature::from_dims_only(
                SmallVec::new(),
            )
        };
        let sig_arc = sig_cache
            .entry(sig.clone())
            .or_insert_with(|| Arc::new(sig))
            .clone();
        var_to_sig
            .insert(var.clone(), sig_arc.clone());
        by_sig
            .entry(sig_arc)
            .or_default()
            .push(var.clone());
    }

    Some((var_list, by_sig, var_to_sig))
}

/// Resolve an `ExprPlan` to a `DatasetSelection` synchronously.
pub(crate) fn resolve_expr_plan_sync<
    B: ChunkedDataBackendSync,
>(
    plan: &ExprPlan,
    meta: &ZarrMeta,
    backend: &B,
) -> Result<DatasetSelection, ResolutionError> {
    match plan {
        ExprPlan::NoConstraint => {
            Ok(DatasetSelection::NoSelectionMade)
        }
        ExprPlan::Empty => {
            Ok(DatasetSelection::Empty)
        }
        ExprPlan::Active {
            vars,
            constraints,
        } => {
            let Some((
                _var_list,
                by_sig,
                var_to_sig,
            )) = build_var_grouping(vars, meta)
            else {
                return Ok(
                    DatasetSelection::Empty,
                );
            };

            let mut by_dims: BTreeMap<
                Arc<DimSignature>,
                DataArraySelection,
            > = BTreeMap::new();

            for (sig_arc, _vars_for_sig) in
                &by_sig
            {
                let dims = sig_arc.dims();
                let shape =
                    dims_to_shape(dims, meta)?;
                let dims_sv: SmallVec<[IStr; 4]> =
                    dims.iter().cloned().collect();
                let resolved =
                    resolve_array_sync(
                        backend,
                        constraints,
                        &dims_sv,
                        &shape,
                        meta,
                    )?;
                if !resolved.is_empty() {
                    by_dims.insert(
                        sig_arc.clone(),
                        resolved,
                    );
                }
            }

            if by_dims.is_empty() {
                Ok(DatasetSelection::Empty)
            } else {
                let grouped =
                    GroupedSelection::from_parts(
                        by_dims, var_to_sig,
                    );
                Ok(DatasetSelection::Selection(
                    grouped,
                ))
            }
        }
    }
}

/// Resolve an `ExprPlan` to a `DatasetSelection` asynchronously.
pub(crate) async fn resolve_expr_plan_async<
    B: ChunkedDataBackendAsync,
>(
    plan: &ExprPlan,
    meta: &ZarrMeta,
    backend: &B,
) -> Result<DatasetSelection, ResolutionError> {
    match plan {
        ExprPlan::NoConstraint => {
            Ok(DatasetSelection::NoSelectionMade)
        }
        ExprPlan::Empty => {
            Ok(DatasetSelection::Empty)
        }
        ExprPlan::Active {
            vars,
            constraints,
        } => {
            let Some((
                _var_list,
                by_sig,
                var_to_sig,
            )) = build_var_grouping(vars, meta)
            else {
                return Ok(
                    DatasetSelection::Empty,
                );
            };

            let mut by_dims: BTreeMap<
                Arc<DimSignature>,
                DataArraySelection,
            > = BTreeMap::new();

            for (sig_arc, _vars_for_sig) in
                &by_sig
            {
                let dims = sig_arc.dims();
                let shape =
                    dims_to_shape(dims, meta)?;
                let dims_sv: SmallVec<[IStr; 4]> =
                    dims.iter().cloned().collect();
                let resolved =
                    resolve_array_async(
                        backend,
                        constraints,
                        &dims_sv,
                        &shape,
                        meta,
                    )
                    .await?;
                if !resolved.is_empty() {
                    by_dims.insert(
                        sig_arc.clone(),
                        resolved,
                    );
                }
            }

            if by_dims.is_empty() {
                Ok(DatasetSelection::Empty)
            } else {
                let grouped =
                    GroupedSelection::from_parts(
                        by_dims, var_to_sig,
                    );
                Ok(DatasetSelection::Selection(
                    grouped,
                ))
            }
        }
    }
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Try to resolve a value range immediately for an index-only dimension (no coordinate array).
fn try_resolve_index_only(
    dim: &IStr,
    meta: &ZarrMeta,
    dim_len: u64,
    vr: &ValueRangePresent,
) -> Option<Range<u64>> {
    if meta.array_by_path_contains(dim) {
        return None;
    }
    vr.index_range_for_index_dim(dim_len)
}

/// Merge overlapping or adjacent ranges into a minimal set.
fn merge_ranges(
    mut ranges: Vec<Range<u64>>,
) -> Vec<Range<u64>> {
    if ranges.len() <= 1 {
        return ranges;
    }
    ranges.sort_unstable_by_key(|r| r.start);
    let mut result =
        Vec::with_capacity(ranges.len());
    let mut current = ranges[0].clone();
    for range in ranges.into_iter().skip(1) {
        if range.start <= current.end {
            current.end =
                current.end.max(range.end);
        } else {
            result.push(current);
            current = range;
        }
    }
    result.push(current);
    result
}
