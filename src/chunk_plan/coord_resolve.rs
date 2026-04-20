//! Coordinate-to-index resolution helpers.
//!
//! Resolves [`ValueRangePresent`] constraints against backend-cached coordinate
//! arrays via binary search. Both sync and async variants are exposed via free
//! functions; the [`Expansion`] enum picks between exact selection,
//! interpolation neighbor expansion (±1 cell), and wrapping ghost expansion
//! (±[`GHOST_EXPANSION`] cells with periodic wrap).

use std::cmp::Ordering as Ord;
use std::ops::Range;

use crate::meta::{TimeEncoding, ZarrMeta};
use crate::reader::ColumnData;
use crate::shared::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync, IStr,
};

use super::indexing::types::{
    CoordScalar, ValueRangePresent,
};

/// Error produced when a value range cannot be resolved to a concrete index
/// range (e.g. the coordinate array is non-monotonic or the dimension has no
/// associated coordinate metadata).
#[derive(Debug, Clone)]
pub enum ResolutionError {
    Unresolvable(String),
}

impl std::fmt::Display for ResolutionError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            ResolutionError::Unresolvable(
                msg,
            ) => {
                write!(f, "unresolvable: {msg}")
            }
        }
    }
}

impl std::error::Error for ResolutionError {}

/// Pre-computed metadata for resolving a single dimension's coordinate array.
#[derive(Debug, Clone)]
pub(crate) struct DimResolutionCtx {
    pub(crate) n: u64,
    pub(crate) chunk_size: u64,
    pub(crate) time_enc: Option<TimeEncoding>,
    pub(crate) array_path: IStr,
}

impl DimResolutionCtx {
    pub(crate) fn from_meta(
        dim: &IStr,
        meta: &ZarrMeta,
    ) -> Option<Self> {
        let coord_meta =
            meta.array_by_path(*dim)?;
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
                .encoding
                .as_ref()
                .and_then(|e| {
                    e.as_time_encoding().cloned()
                }),
            array_path: coord_meta.path,
        })
    }
}

/// Selects how a [`ValueRangePresent`] is converted into one or more index
/// ranges after the coordinate-space binary search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expansion {
    /// No expansion — return the searched cells exactly.
    Exact,
    /// Add ±1 cell on each side when the target lies strictly between grid
    /// points (used for non-wrapping interpolation).
    InterpolationNeighbor,
    /// Add ±[`GHOST_EXPANSION`] cells and wrap to the opposite boundary when
    /// near an edge (used for periodic-grid interpolation, e.g. longitude).
    WrappingGhost,
}

/// Ghost-point expansion size for wrapping interpolation; matches interpolars'
/// `k = min(n - 1, 3)`.
pub const GHOST_EXPANSION: u64 = 3;

// ============================================================================
// Pure helpers (no I/O)
// ============================================================================

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

fn monotonic_sample_indices(
    n: u64,
    chunk_size: u64,
) -> [u64; 5] {
    if n == 0 {
        panic!("n cannot be 0");
    }
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
        if let Some(p) = prev
            && !monotonic_ord_matches(
                dir,
                CoordScalar::partial_cmp(p, v),
            )
        {
            return None;
        }
        prev = Some(v);
    }
    Some(dir)
}

#[inline]
fn pin_interpolation_without_neighbor_cells(
    vr: &ValueRangePresent,
    r: &Range<u64>,
) -> bool {
    vr.is_point_included_equal()
        && r.end.saturating_sub(r.start) == 1
}

/// Compute wrapping ghost ranges given a primary index range and dimension size.
///
/// Returns a `Vec<Range<u64>>` containing the primary range (expanded by
/// [`GHOST_EXPANSION`]) plus up to two ghost ranges at the opposite boundary
/// when the primary range is within `GHOST_EXPANSION` of either edge.
pub(crate) fn wrapping_ghost_ranges(
    primary: Range<u64>,
    n: u64,
) -> Vec<Range<u64>> {
    let start = primary
        .start
        .saturating_sub(GHOST_EXPANSION);
    let end =
        (primary.end + GHOST_EXPANSION).min(n);
    let mut ranges = Vec::with_capacity(3);
    if start < end {
        ranges.push(start..end);
    }
    if start < GHOST_EXPANSION
        && n > GHOST_EXPANSION
    {
        let ghost_start =
            n.saturating_sub(GHOST_EXPANSION);
        if ghost_start < n {
            ranges.push(ghost_start..n);
        }
    }
    if end > n.saturating_sub(GHOST_EXPANSION)
        && n > GHOST_EXPANSION
    {
        let ghost_end = GHOST_EXPANSION.min(n);
        if ghost_end > 0 {
            ranges.push(0..ghost_end);
        }
    }
    ranges
}

/// Apply an [`Expansion`] policy to a single resolved index range.
fn apply_expansion(
    r: Range<u64>,
    n: u64,
    vr: &ValueRangePresent,
    exp: Expansion,
) -> Vec<Range<u64>> {
    match exp {
        Expansion::Exact => {
            if r.start < r.end {
                vec![r]
            } else {
                vec![]
            }
        }
        Expansion::InterpolationNeighbor => {
            if pin_interpolation_without_neighbor_cells(
                vr, &r,
            ) {
                return vec![r];
            }
            let start = r.start.saturating_sub(1);
            let end =
                r.end.saturating_add(1).min(n);
            if start < end {
                vec![start..end]
            } else {
                vec![]
            }
        }
        Expansion::WrappingGhost => {
            wrapping_ghost_ranges(r, n)
        }
    }
}

/// Try to resolve a value range immediately for an index-only dimension
/// (no coordinate array). Returns `None` when the dimension has a coordinate
/// array and the search must use the backend.
pub(crate) fn try_resolve_index_only(
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

// ============================================================================
// Sync resolution
// ============================================================================

fn coord_scalar_from_chunk(
    chunk: &ColumnData,
    offset: usize,
    time_enc: Option<&TimeEncoding>,
) -> Option<CoordScalar> {
    match chunk {
        ColumnData::F64(v) => {
            let val = v[offset];
            if let Some(enc) = time_enc {
                enc.decode_f64(val).map(|ns| {
                    if enc.is_duration {
                        CoordScalar::DurationNs(
                            ns,
                        )
                    } else {
                        CoordScalar::DatetimeNs(
                            ns,
                        )
                    }
                })
            } else {
                Some(CoordScalar::F64(val))
            }
        }
        ColumnData::F32(v) => {
            let val = v[offset] as f64;
            if let Some(enc) = time_enc {
                enc.decode_f64(val).map(|ns| {
                    if enc.is_duration {
                        CoordScalar::DurationNs(
                            ns,
                        )
                    } else {
                        CoordScalar::DatetimeNs(
                            ns,
                        )
                    }
                })
            } else {
                Some(CoordScalar::F64(val))
            }
        }
        _ => chunk.get_i64(offset).map(|raw| {
            super::exprs::apply_time_encoding(
                raw, time_enc,
            )
        }),
    }
}

fn scalar_at_sync<B: ChunkedDataBackendSync>(
    backend: &B,
    dim: &IStr,
    idx: u64,
    n: u64,
    chunk_size: u64,
    time_enc: Option<&TimeEncoding>,
) -> Option<CoordScalar> {
    if idx >= n {
        return None;
    }
    let chunk_idx = idx / chunk_size;
    let offset = (idx % chunk_size) as usize;
    let chunk = backend
        .read_chunk_sync(dim, &[chunk_idx])
        .ok()?;
    coord_scalar_from_chunk(
        &chunk, offset, time_enc,
    )
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

fn resolve_value_range_inner_sync<
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

/// Resolve a value range against the dimension `dim` synchronously.
///
/// Returns the resolved index ranges (post-expansion). Falls back to the full
/// `0..dim_len` range when monotonicity / binary search cannot determine a
/// tighter result.
pub(crate) fn resolve_value_range_sync<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    dim: &IStr,
    meta: &ZarrMeta,
    dim_len: u64,
    vr: &ValueRangePresent,
    expansion: Expansion,
) -> Result<Vec<Range<u64>>, ResolutionError> {
    if let Some(r) = try_resolve_index_only(
        dim, meta, dim_len, vr,
    ) {
        return Ok(apply_expansion(
            r, dim_len, vr, expansion,
        ));
    }
    let Some(ctx) =
        DimResolutionCtx::from_meta(dim, meta)
    else {
        // Without a coord array we cannot binary-search: fall back to the
        // whole range for non-interpolation queries; interpolation requires
        // a coord array so we surface the failure.
        return match expansion {
            Expansion::Exact => {
                Ok(vec![0..dim_len])
            }
            _ => Err(
                ResolutionError::Unresolvable(
                    format!(
                        "dimension '{}' has no coordinate array",
                        AsRef::<str>::as_ref(dim)
                    ),
                ),
            ),
        };
    };
    let Some(dir) =
        check_monotonicity_sync(backend, &ctx)
    else {
        return match expansion {
            Expansion::Exact => {
                Ok(vec![0..dim_len])
            }
            _ => Err(
                ResolutionError::Unresolvable(
                    format!(
                        "dimension '{}' coordinate array is not monotonic",
                        AsRef::<str>::as_ref(dim)
                    ),
                ),
            ),
        };
    };
    let r = resolve_value_range_inner_sync(
        backend, &ctx, vr, dir,
    )
    .unwrap_or(0..ctx.n);
    Ok(apply_expansion(r, ctx.n, vr, expansion))
}

// ============================================================================
// Async resolution
// ============================================================================

async fn scalar_at_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    dim: &IStr,
    idx: u64,
    n: u64,
    chunk_size: u64,
    time_enc: Option<&TimeEncoding>,
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
    coord_scalar_from_chunk(
        &chunk, offset, time_enc,
    )
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
    check_monotonic_from_samples(
        &first, &last, &samples?,
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

async fn resolve_value_range_inner_async<
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

/// Resolve a value range against the dimension `dim` asynchronously.
pub(crate) async fn resolve_value_range_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    dim: &IStr,
    meta: &ZarrMeta,
    dim_len: u64,
    vr: &ValueRangePresent,
    expansion: Expansion,
) -> Result<Vec<Range<u64>>, ResolutionError> {
    if let Some(r) = try_resolve_index_only(
        dim, meta, dim_len, vr,
    ) {
        return Ok(apply_expansion(
            r, dim_len, vr, expansion,
        ));
    }
    let Some(ctx) =
        DimResolutionCtx::from_meta(dim, meta)
    else {
        return match expansion {
            Expansion::Exact => {
                Ok(vec![0..dim_len])
            }
            _ => Err(
                ResolutionError::Unresolvable(
                    format!(
                        "dimension '{}' has no coordinate array",
                        AsRef::<str>::as_ref(dim)
                    ),
                ),
            ),
        };
    };
    let Some(dir) =
        check_monotonicity_async(backend, &ctx)
            .await
    else {
        return match expansion {
            Expansion::Exact => {
                Ok(vec![0..dim_len])
            }
            _ => Err(
                ResolutionError::Unresolvable(
                    format!(
                        "dimension '{}' coordinate array is not monotonic",
                        AsRef::<str>::as_ref(dim)
                    ),
                ),
            ),
        };
    };
    let r = resolve_value_range_inner_async(
        backend, &ctx, vr, dir,
    )
    .await
    .unwrap_or(0..ctx.n);
    Ok(apply_expansion(r, ctx.n, vr, expansion))
}

// ============================================================================
// Resolver traits
// ============================================================================

/// Synchronous coordinate resolver, abstracting over the concrete backend.
///
/// Lives behind a trait so the [`super::indexing::builder::GridJoinTreeBuilder`]
/// can be instantiated for both sync and async backends without dragging
/// `async_trait` into hot paths that don't need it.
pub trait CoordResolverSync {
    fn resolve(
        &self,
        dim: &IStr,
        meta: &ZarrMeta,
        dim_len: u64,
        vr: &ValueRangePresent,
        expansion: Expansion,
    ) -> Result<Vec<Range<u64>>, ResolutionError>;
}

impl<B: ChunkedDataBackendSync> CoordResolverSync
    for B
{
    fn resolve(
        &self,
        dim: &IStr,
        meta: &ZarrMeta,
        dim_len: u64,
        vr: &ValueRangePresent,
        expansion: Expansion,
    ) -> Result<Vec<Range<u64>>, ResolutionError>
    {
        resolve_value_range_sync(
            self, dim, meta, dim_len, vr,
            expansion,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_ranges_interior_single_range() {
        let ranges =
            wrapping_ghost_ranges(50..55, 360);
        assert_eq!(ranges, vec![47..58]);
    }

    #[test]
    fn ghost_ranges_near_start_adds_end_ghost() {
        let ranges =
            wrapping_ghost_ranges(1..5, 360);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], 0..8);
        assert_eq!(ranges[1], 357..360);
    }

    #[test]
    fn ghost_ranges_at_start_adds_end_ghost() {
        let ranges =
            wrapping_ghost_ranges(0..3, 360);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], 0..6);
        assert_eq!(ranges[1], 357..360);
    }

    #[test]
    fn ghost_ranges_near_end_adds_start_ghost() {
        let ranges =
            wrapping_ghost_ranges(356..360, 360);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], 353..360);
        assert_eq!(ranges[1], 0..3);
    }

    #[test]
    fn ghost_ranges_at_end_adds_start_ghost() {
        let ranges =
            wrapping_ghost_ranges(358..360, 360);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], 355..360);
        assert_eq!(ranges[1], 0..3);
    }

    #[test]
    fn ghost_ranges_small_dimension_both_ghosts()
    {
        let ranges =
            wrapping_ghost_ranges(2..4, 5);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 0..5);
        assert_eq!(ranges[1], 2..5);
        assert_eq!(ranges[2], 0..3);
    }

    #[test]
    fn ghost_ranges_very_small_dimension() {
        let ranges =
            wrapping_ghost_ranges(0..2, 3);
        assert_eq!(ranges, vec![0..3]);
    }

    #[test]
    fn ghost_ranges_exact_ghost_expansion_size() {
        let ranges =
            wrapping_ghost_ranges(1..2, 3);
        assert_eq!(ranges, vec![0..3]);
    }

    #[test]
    fn ghost_expansion_constant_matches_interpolars()
     {
        assert_eq!(GHOST_EXPANSION, 3);
    }
}
