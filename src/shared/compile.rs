use std::collections::BTreeMap;

use std::cmp::Ordering as Ord;

use super::traits::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync, HasAsyncStore,
    HasMetadataBackendAsync,
    HasMetadataBackendSync, HasStore,
};
use crate::errors::BackendError;
use polars::prelude::Expr;

use crate::PlannerStats;
use crate::chunk_plan::GroupedChunkPlan;
use crate::chunk_plan::SyncCoordResolver;
use crate::chunk_plan::compile_expr;
use crate::chunk_plan::selection_to_grouped_chunk_plan_unified_from_meta;
use crate::chunk_plan::{
    MergedCache, collect_requests_with_meta,
    materialize,
};
use crate::meta::ZarrMeta;
use crate::{IStr, chunk_plan::*};

// -----------------------------------------------------------------------------
// Pure helper functions (no I/O, no async - safe to extract without perf impact)
// -----------------------------------------------------------------------------

/// Group resolution requests by dimension for batch processing.
fn group_requests_by_dimension(
    requests: Vec<ResolutionRequest>,
) -> BTreeMap<
    IStr,
    Vec<(ResolutionRequest, ValueRangePresent)>,
> {
    let mut by_dim: BTreeMap<
        IStr,
        Vec<(
            ResolutionRequest,
            ValueRangePresent,
        )>,
    > = BTreeMap::new();
    for req in requests {
        by_dim
            .entry(req.dim.clone())
            .or_default()
            .push((
                req.clone(),
                req.value_range.clone(),
            ));
    }
    by_dim
}

/// Binary search: should we move the high bound left? (lower_bound semantics)
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

/// Binary search: should we move the low bound right? (upper_bound semantics)
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

/// Sample indices for monotonicity verification (must be sorted ascending).
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

/// Check if the observed ordering between consecutive samples matches expected direction.
#[inline(always)]
fn monotonic_ord_matches(
    dir: Ord,
    ord: Option<Ord>,
) -> bool {
    match (dir, ord) {
        (
            Ord::Less,
            Some(Ord::Less | Ord::Equal),
        ) => true,
        (
            Ord::Greater,
            Some(Ord::Greater | Ord::Equal),
        ) => true,
        _ => false,
    }
}

// -----------------------------------------------------------------------------
// Shared pure functions for coordinate resolution (used by both sync and async)
// -----------------------------------------------------------------------------

/// Pre-computed metadata for resolving a single dimension's coordinate array.
struct DimResolutionCtx {
    n: u64,
    chunk_size: u64,
    time_enc: Option<crate::meta::TimeEncoding>,
    array_path: IStr,
}

impl DimResolutionCtx {
    fn from_meta(
        dim: &IStr,
        meta: &ZarrMeta,
    ) -> Option<Self> {
        let coord_meta =
            meta.array_by_path(dim.clone())?;
        if coord_meta.shape.len() != 1 {
            return None;
        }
        let n = coord_meta.shape[0];
        Some(Self {
            n,
            chunk_size: coord_meta
                .chunk_shape
                .first()
                .copied()
                .unwrap_or(n),
            time_enc: coord_meta
                .time_encoding
                .clone(),
            array_path: coord_meta.path.clone(),
        })
    }
}

/// Determine monotonicity direction from pre-fetched sample values.
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

/// Generic binary search: `go_right_fn` returns true when `lo` should advance.
/// Used for both lower-bound and upper-bound searches.
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

/// Prepare compilation inputs: compile expr to ExprPlan, convert to lazy selection,
/// and collect resolution requests.
fn prepare_compile_inputs(
    meta: &ZarrMeta,
    expr: &Expr,
) -> Result<
    (
        LazyDatasetSelection,
        Vec<ResolutionRequest>,
        HashMapCache,
    ),
    BackendError,
> {
    let (dims, dim_lengths) =
        compute_dims_and_lengths_unified(meta);
    let mut ctx =
        LazyCompileCtx::new(&meta, &dims);
    let expr_plan = compile_expr(expr, &mut ctx)?;
    let lazy_selection = expr_plan
        .into_lazy_dataset_selection(meta);
    let (requests, immediate_cache) =
        collect_requests_with_meta(
            &lazy_selection,
            &meta,
            &dim_lengths,
            &dims,
        );
    Ok((
        lazy_selection,
        requests,
        immediate_cache,
    ))
}

/// Materialize lazy selection with resolved cache and convert to grouped chunk plan.
fn finish_compile_with_resolved_cache(
    lazy_selection: &LazyDatasetSelection,
    meta: &ZarrMeta,
    resolved_cache: &dyn ResolutionCache,
    immediate_cache: &HashMapCache,
) -> Result<
    (GroupedChunkPlan, PlannerStats),
    BackendError,
> {
    let merged = MergedCache::new(
        resolved_cache,
        immediate_cache,
    );
    let selection = materialize(
        lazy_selection,
        &meta,
        &merged,
    )?;
    let stats = PlannerStats { coord_reads: 0 };
    let grouped_plan =
        selection_to_grouped_chunk_plan_unified_from_meta(&selection, meta)?;
    Ok((grouped_plan, stats))
}

/// Compile a Polars expression to a chunk plan synchronously.
pub trait ChunkedExpressionCompilerSync:
    HasMetadataBackendSync<ZarrMeta>
    + ChunkedDataBackendSync
{
    fn compile_expression_sync(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    >;
}

#[async_trait::async_trait]
pub trait ChunkedExpressionCompilerAsync:
    HasMetadataBackendAsync<ZarrMeta>
    + ChunkedDataBackendAsync
{
    async fn compile_expression_async(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    >;
}

impl<
    B: HasMetadataBackendSync<ZarrMeta>
        + ChunkedDataBackendSync
        + HasStore
        + SyncCoordResolver,
> ChunkedExpressionCompilerSync for B
{
    fn compile_expression_sync(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    > {
        let meta = self.metadata()?;
        let (
            lazy_selection,
            requests,
            immediate_cache,
        ) = prepare_compile_inputs(&meta, expr)?;
        let resolved_cache =
            self.resolve_batch(requests);
        finish_compile_with_resolved_cache(
            &lazy_selection,
            &meta,
            &*resolved_cache,
            &immediate_cache,
        )
    }
}

#[async_trait::async_trait]
impl<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync
        + HasAsyncStore,
> ChunkedExpressionCompilerAsync for B
{
    async fn compile_expression_async(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    > {
        let meta = self.metadata().await?;
        let (
            lazy_selection,
            requests,
            immediate_cache,
        ) = prepare_compile_inputs(&meta, expr)?;
        let resolved_cache =
            self.resolve_batch(requests).await;
        finish_compile_with_resolved_cache(
            &lazy_selection,
            &meta,
            &*resolved_cache,
            &immediate_cache,
        )
    }
}

#[async_trait::async_trait]
impl<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync
        + HasAsyncStore,
> AsyncCoordResolver for B
{
    async fn resolve_batch(
        &self,
        requests: Vec<ResolutionRequest>,
    ) -> Box<dyn ResolutionCache + Send + Sync>
    {
        let mut cache = HashMapCache::new();
        let by_dim =
            group_requests_by_dimension(requests);

        for (dim, reqs) in by_dim {
            let results = self
                .resolve_dimension(&dim, reqs)
                .await;
            for (req, result) in results {
                cache.insert(req, result);
            }
        }

        Box::new(cache)
    }
}

impl<
    B: HasMetadataBackendSync<ZarrMeta>
        + ChunkedDataBackendSync
        + HasStore
        + Sync,
> SyncCoordResolver for B
{
    fn resolve_batch(
        &self,
        requests: Vec<ResolutionRequest>,
    ) -> Box<dyn ResolutionCache + Send + Sync>
    {
        use rayon::prelude::*;

        let mut cache = HashMapCache::new();
        let by_dim =
            group_requests_by_dimension(requests);

        let results: Vec<_> = by_dim
            .into_par_iter()
            .flat_map_iter(|(dim, reqs)| {
                self.resolve_dimension_sync(
                    &dim, reqs,
                )
            })
            .collect();

        for (req, result) in results {
            cache.insert(req, result);
        }

        Box::new(cache)
    }
}

trait ResolveDimension {
    async fn resolve_dimension(
        &self,
        dim: &IStr,
        reqs: Vec<(
            ResolutionRequest,
            ValueRangePresent,
        )>,
    ) -> Vec<(
        ResolutionRequest,
        Option<std::ops::Range<u64>>,
    )>;

    async fn scalar_at(
        &self,
        dim: &IStr,
        idx: u64,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<CoordScalar>;
}

trait ResolveDimensionSync {
    fn resolve_dimension_sync(
        &self,
        dim: &IStr,
        reqs: Vec<(
            ResolutionRequest,
            ValueRangePresent,
        )>,
    ) -> Vec<(
        ResolutionRequest,
        Option<std::ops::Range<u64>>,
    )>;

    fn scalar_at_sync(
        &self,
        dim: &IStr,
        idx: u64,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<CoordScalar>;
}

impl<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync
        + HasAsyncStore,
> ResolveDimension for B
{
    async fn resolve_dimension(
        &self,
        dim: &IStr,
        reqs: Vec<(
            ResolutionRequest,
            ValueRangePresent,
        )>,
    ) -> Vec<(
        ResolutionRequest,
        Option<std::ops::Range<u64>>,
    )> {
        let meta =
            match self.metadata().await.ok() {
                Some(m) => m,
                None => {
                    return reqs
                        .into_iter()
                        .map(|(r, _)| (r, None))
                        .collect();
                }
            };
        let Some(ctx) =
            DimResolutionCtx::from_meta(
                dim, &meta,
            )
        else {
            return reqs
                .into_iter()
                .map(|(r, _)| (r, None))
                .collect();
        };
        if ctx.n == 0 {
            return reqs
                .into_iter()
                .map(|(r, _)| (r, Some(0..0)))
                .collect();
        }

        let te = ctx.time_enc.as_ref();
        let ap = &ctx.array_path;

        // Check monotonicity via sampled scalars
        let dir = if ctx.n < 2 {
            Ord::Less
        } else {
            let first = self
                .scalar_at(
                    ap,
                    0,
                    ctx.n,
                    ctx.chunk_size,
                    te,
                )
                .await;
            let last = self
                .scalar_at(
                    ap,
                    ctx.n - 1,
                    ctx.n,
                    ctx.chunk_size,
                    te,
                )
                .await;
            let (Some(first), Some(last)) =
                (first, last)
            else {
                return reqs
                    .into_iter()
                    .map(|(r, _)| (r, None))
                    .collect();
            };
            let indices =
                monotonic_sample_indices(
                    ctx.n,
                    ctx.chunk_size,
                );
            let mut samples =
                Vec::with_capacity(indices.len());
            for &i in &indices {
                let Some(v) = self
                    .scalar_at(
                        ap,
                        i,
                        ctx.n,
                        ctx.chunk_size,
                        te,
                    )
                    .await
                else {
                    return reqs
                        .into_iter()
                        .map(|(r, _)| (r, None))
                        .collect();
                };
                samples.push(v);
            }
            let Some(d) =
                check_monotonic_from_samples(
                    &first, &last, &samples,
                )
            else {
                return reqs
                    .into_iter()
                    .map(|(r, _)| (r, None))
                    .collect();
            };
            d
        };

        use futures::future::join_all;
        let futures =
            reqs.into_iter().map(|(req, vr)| {
                let ap2 = ctx.array_path.clone();
                let te2 = ctx.time_enc.clone();
                async move {
                    let result =
                        async_resolve_single(
                            self,
                            &ap2,
                            &vr,
                            dir,
                            ctx.n,
                            ctx.chunk_size,
                            te2.as_ref(),
                        )
                        .await;
                    (req, result)
                }
            });
        join_all(futures).await
    }

    async fn scalar_at(
        &self,
        dim: &IStr,
        idx: u64,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<CoordScalar> {
        if idx >= n {
            return None;
        }
        let chunk_idx = idx / chunk_size;
        let offset = (idx % chunk_size) as usize;
        let chunk = self
            .read_chunk_async(dim, &[chunk_idx])
            .await
            .ok()?;
        chunk.get_i64(offset).map(|raw| {
            crate::chunk_plan::apply_time_encoding(raw, time_enc)
        })
    }
}

/// Resolve a single value range to an index range (async).
async fn async_resolve_single<B>(
    backend: &B,
    dim: &IStr,
    vr: &ValueRangePresent,
    dir: Ord,
    n: u64,
    chunk_size: u64,
    time_enc: Option<&crate::meta::TimeEncoding>,
) -> Option<std::ops::Range<u64>>
where
    B: ResolveDimension,
{
    use std::ops::Bound;
    let start = match &vr.0 {
        Bound::Included(s)
        | Bound::Excluded(s) => {
            let t = s.clone();
            let strict = matches!(
                &vr.0,
                Bound::Excluded(_)
            );
            async_binary_search(
                backend, dim, &t, strict, dir, n,
                chunk_size, time_enc, false,
            )
            .await?
        }
        Bound::Unbounded => 0,
    };
    let end = match &vr.1 {
        Bound::Included(s)
        | Bound::Excluded(s) => {
            let t = s.clone();
            let strict = matches!(
                &vr.1,
                Bound::Excluded(_)
            );
            async_binary_search(
                backend, dim, &t, strict, dir, n,
                chunk_size, time_enc, true,
            )
            .await?
        }
        Bound::Unbounded => n,
    };
    Some(start..end)
}

/// Binary search on a coordinate array (async). `is_upper` selects upper-bound vs lower-bound semantics.
async fn async_binary_search<
    B: ResolveDimension,
>(
    backend: &B,
    dim: &IStr,
    target: &CoordScalar,
    strict: bool,
    dir: Ord,
    n: u64,
    chunk_size: u64,
    time_enc: Option<&crate::meta::TimeEncoding>,
    is_upper: bool,
) -> Option<u64> {
    let mut lo = 0u64;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let v = backend
            .scalar_at(
                dim, mid, n, chunk_size, time_enc,
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

impl<
    B: HasMetadataBackendSync<ZarrMeta>
        + ChunkedDataBackendSync
        + HasStore
        + Sync,
> ResolveDimensionSync for B
{
    fn resolve_dimension_sync(
        &self,
        dim: &IStr,
        reqs: Vec<(
            ResolutionRequest,
            ValueRangePresent,
        )>,
    ) -> Vec<(
        ResolutionRequest,
        Option<std::ops::Range<u64>>,
    )> {
        use rayon::prelude::*;

        let meta = match self.metadata().ok() {
            Some(m) => m,
            None => {
                return reqs
                    .into_iter()
                    .map(|(r, _)| (r, None))
                    .collect();
            }
        };
        let Some(ctx) =
            DimResolutionCtx::from_meta(
                dim, &meta,
            )
        else {
            return reqs
                .into_iter()
                .map(|(r, _)| (r, None))
                .collect();
        };
        if ctx.n == 0 {
            return reqs
                .into_iter()
                .map(|(r, _)| (r, Some(0..0)))
                .collect();
        }

        let te = ctx.time_enc.as_ref();
        let ap = &ctx.array_path;

        let dir = if ctx.n < 2 {
            Ord::Less
        } else {
            let first = self.scalar_at_sync(
                ap,
                0,
                ctx.n,
                ctx.chunk_size,
                te,
            );
            let last = self.scalar_at_sync(
                ap,
                ctx.n - 1,
                ctx.n,
                ctx.chunk_size,
                te,
            );
            let (Some(first), Some(last)) =
                (first, last)
            else {
                return reqs
                    .into_iter()
                    .map(|(r, _)| (r, None))
                    .collect();
            };
            let indices =
                monotonic_sample_indices(
                    ctx.n,
                    ctx.chunk_size,
                );
            let mut samples =
                Vec::with_capacity(indices.len());
            for &i in &indices {
                let Some(v) = self
                    .scalar_at_sync(
                        ap,
                        i,
                        ctx.n,
                        ctx.chunk_size,
                        te,
                    )
                else {
                    return reqs
                        .into_iter()
                        .map(|(r, _)| (r, None))
                        .collect();
                };
                samples.push(v);
            }
            let Some(d) =
                check_monotonic_from_samples(
                    &first, &last, &samples,
                )
            else {
                return reqs
                    .into_iter()
                    .map(|(r, _)| (r, None))
                    .collect();
            };
            d
        };

        reqs.into_par_iter()
            .map(|(req, vr)| {
                let resolved =
                    sync_resolve_single(
                        self,
                        ap,
                        &vr,
                        dir,
                        ctx.n,
                        ctx.chunk_size,
                        te,
                    );
                (req, resolved)
            })
            .collect()
    }

    fn scalar_at_sync(
        &self,
        dim: &IStr,
        idx: u64,
        n: u64,
        chunk_size: u64,
        time_enc: Option<
            &crate::meta::TimeEncoding,
        >,
    ) -> Option<CoordScalar> {
        if idx >= n {
            return None;
        }
        let chunk_idx = idx / chunk_size;
        let offset = (idx % chunk_size) as usize;
        let chunk = self
            .read_chunk_sync(dim, &[chunk_idx])
            .ok()?;
        chunk.get_i64(offset).map(|raw| {
            crate::chunk_plan::apply_time_encoding(raw, time_enc)
        })
    }
}

/// Resolve a single value range to an index range (sync).
fn sync_resolve_single<
    B: ResolveDimensionSync,
>(
    backend: &B,
    dim: &IStr,
    vr: &ValueRangePresent,
    dir: Ord,
    n: u64,
    chunk_size: u64,
    time_enc: Option<&crate::meta::TimeEncoding>,
) -> Option<std::ops::Range<u64>> {
    use std::ops::Bound;
    let start = match &vr.0 {
        Bound::Included(s)
        | Bound::Excluded(s) => {
            let strict = matches!(
                &vr.0,
                Bound::Excluded(_)
            );
            sync_binary_search(
                backend, dim, s, strict, dir, n,
                chunk_size, time_enc, false,
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
                backend, dim, s, strict, dir, n,
                chunk_size, time_enc, true,
            )?
        }
        Bound::Unbounded => n,
    };
    Some(start..end)
}

/// Binary search on a coordinate array (sync). `is_upper` selects upper-bound vs lower-bound semantics.
fn sync_binary_search<B: ResolveDimensionSync>(
    backend: &B,
    dim: &IStr,
    target: &CoordScalar,
    strict: bool,
    dir: Ord,
    n: u64,
    chunk_size: u64,
    time_enc: Option<&crate::meta::TimeEncoding>,
    is_upper: bool,
) -> Option<u64> {
    let mut lo = 0u64;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let v = backend.scalar_at_sync(
            dim, mid, n, chunk_size, time_enc,
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
