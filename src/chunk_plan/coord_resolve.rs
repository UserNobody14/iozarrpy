//! Coordinate-to-index resolution.
//!
//! Maps a [`ValueRangePresent`] over a dim's coordinate values to a
//! `Range<u64>` of array indices via [`slice::partition_point`] on a
//! type-native coordinate buffer (`Vec<f64>` or `Vec<i64>`). Backends may
//! declare a dim already-sorted (via
//! [`ChunkedDataBackendSync::assume_sorted_dim`]) to skip the
//! [`slice::is_sorted_by`] monotonicity probe.
//!
//! `CoordScalar` is only used as the IR for `ValueRangePresent` bounds —
//! the search itself runs on raw types. Time-encoded coords decode once
//! into `Vec<i64>` nanoseconds; plain floats stay as `Vec<f64>`. NaN values
//! produced by float-time decoding are dropped silently.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::{Bound, Range};

use crate::chunk_plan::exprs::compile_ctx::CoordResolver;
use crate::errors::BackendError;
use crate::meta::{TimeEncoding, ZarrMeta};
use crate::reader::ColumnData;
use crate::shared::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync, IStr, MaybeParIter,
};

use super::indexing::types::{
    CoordScalar, ValueRangePresent,
};

/// Selects how a [`ValueRangePresent`] is converted into one or more index
/// ranges after the binary search.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash,
)]
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
// Coord buffer + dim context
// ============================================================================

/// Type-native coord array. Two variants cover all coord dtypes:
/// `F64` for plain floats; `I64` for plain ints (any signed/unsigned width
/// gets cast) AND for time-encoded coords (decoded to nanoseconds).
enum CoordBuffer {
    F64(Vec<f64>),
    I64(Vec<i64>),
}

impl CoordBuffer {
    fn len(&self) -> usize {
        match self {
            CoordBuffer::F64(v) => v.len(),
            CoordBuffer::I64(v) => v.len(),
        }
    }
}

#[derive(Debug)]
struct DimCtx {
    n: u64,
    chunk_size: u64,
    time_enc: Option<TimeEncoding>,
    array_path: IStr,
}

impl DimCtx {
    fn from_meta(
        dim: &IStr,
        meta: &ZarrMeta,
    ) -> Option<Self> {
        let arr = meta.array_by_path(*dim)?;
        if arr.shape.len() != 1 {
            return None;
        }
        let n = arr.shape[0];
        Some(Self {
            n,
            chunk_size: arr
                .chunk_shape
                .first()
                .copied()
                .unwrap_or(n),
            time_enc: arr
                .encoding
                .as_ref()
                .and_then(|e| {
                    e.as_time_encoding().cloned()
                }),
            array_path: arr.path,
        })
    }

    fn n_chunks(&self) -> u64 {
        if self.chunk_size == 0 {
            0
        } else {
            self.n.div_ceil(self.chunk_size)
        }
    }
}

// ============================================================================
// Decode chunks into a typed buffer
// ============================================================================

/// Decide the buffer type once: plain F64/F32 chunks → `F64`; everything
/// else (signed/unsigned ints, time-encoded floats, time-encoded ints) →
/// `I64`. Time-encoded coords decode to `i64` nanoseconds during the load
/// pass, never producing a `CoordScalar`.
fn buffer_kind(
    sample: &ColumnData,
    has_time_enc: bool,
) -> BufferKind {
    if has_time_enc {
        return BufferKind::I64;
    }
    match sample {
        ColumnData::F32(_) | ColumnData::F64(_) => {
            BufferKind::F64
        }
        _ => BufferKind::I64,
    }
}

#[derive(Clone, Copy)]
enum BufferKind {
    F64,
    I64,
}

fn build_buffer(
    chunks: &[std::sync::Arc<ColumnData>],
    n: usize,
    te: Option<&TimeEncoding>,
) -> CoordBuffer {
    let kind = chunks
        .first()
        .map(|c| {
            buffer_kind(c.as_ref(), te.is_some())
        })
        .unwrap_or(BufferKind::I64);
    match kind {
        BufferKind::F64 => {
            let mut out: Vec<f64> =
                Vec::with_capacity(n);
            for chunk in chunks {
                if let Some(v) =
                    chunk.as_f64_values()
                {
                    out.extend_from_slice(v);
                } else if let Some(v) =
                    chunk.as_f32_values()
                {
                    out.extend(
                        v.iter().map(|&x| x as f64),
                    );
                } else {
                    // Mixed-dtype chunks
                    // shouldn't happen; fall
                    // through via get_i64 cast
                    // for safety.
                    for i in 0..chunk.len() {
                        out.push(
                            chunk
                                .get_i64(i)
                                .unwrap_or(0)
                                as f64,
                        );
                    }
                }
            }
            out.truncate(n);
            CoordBuffer::F64(out)
        }
        BufferKind::I64 => {
            let mut out: Vec<i64> =
                Vec::with_capacity(n);
            for chunk in chunks {
                decode_chunk_to_i64(
                    chunk.as_ref(),
                    te,
                    &mut out,
                );
            }
            out.truncate(n);
            CoordBuffer::I64(out)
        }
    }
}

fn decode_chunk_to_i64(
    chunk: &ColumnData,
    te: Option<&TimeEncoding>,
    out: &mut Vec<i64>,
) {
    match te {
        Some(enc) => {
            if let Some(v) = chunk.as_f64_values() {
                // F64 raw + time encoding: decode each
                // value to i64 ns; drop NaN/non-finite.
                for &x in v {
                    if let Some(ns) =
                        enc.decode_f64(x)
                    {
                        out.push(ns);
                    }
                }
                return;
            }
            if let Some(v) = chunk.as_f32_values() {
                for &x in v {
                    if let Some(ns) =
                        enc.decode_f64(x as f64)
                    {
                        out.push(ns);
                    }
                }
                return;
            }
            if let Some(v) = chunk.as_i64_values() {
                out.extend(
                    v.iter().map(|&x| enc.decode(x)),
                );
                return;
            }
        }
        None => {
            if let Some(v) = chunk.as_i64_values() {
                out.extend_from_slice(v);
                return;
            }
        }
    }
    // Other integer widths or float-without-encoding (rare for
    // I64 buffer): cast each element via get_i64 then optionally
    // apply time encoding.
    for i in 0..chunk.len() {
        let raw =
            chunk.get_i64(i).unwrap_or(0);
        let val = te
            .map_or(raw, |enc| enc.decode(raw));
        out.push(val);
    }
}

// ============================================================================
// Async/sync chunk loading
// ============================================================================

fn load_chunks_sync<B: ChunkedDataBackendSync>(
    backend: &B,
    ctx: &DimCtx,
) -> Result<
    Vec<std::sync::Arc<ColumnData>>,
    BackendError,
> {
    let n_chunks = ctx.n_chunks() as usize;
    const PARALLEL_CHUNK_THRESHOLD: usize = 2;
    (0..n_chunks)
        .collect::<Vec<_>>()
        .maybe_par_iter(PARALLEL_CHUNK_THRESHOLD)
        .map_collect(|ci| {
            backend.read_chunk_sync(
                &ctx.array_path,
                &[*ci as u64],
            )
        })
}

async fn load_chunks_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    ctx: &DimCtx,
) -> Result<
    Vec<std::sync::Arc<ColumnData>>,
    BackendError,
> {
    futures::future::try_join_all(
        (0..ctx.n_chunks()).map(
            |ci| async move {
                backend
                    .read_chunk_async(
                        &ctx.array_path,
                        &[ci],
                    )
                    .await
            },
        ),
    )
    .await
}

fn load_coord_buffer_sync<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    ctx: &DimCtx,
) -> Result<CoordBuffer, BackendError> {
    let chunks = load_chunks_sync(backend, ctx)?;
    Ok(build_buffer(
        &chunks,
        ctx.n as usize,
        ctx.time_enc.as_ref(),
    ))
}

async fn load_coord_buffer_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    ctx: &DimCtx,
) -> Result<CoordBuffer, BackendError> {
    let chunks =
        load_chunks_async(backend, ctx).await?;
    Ok(build_buffer(
        &chunks,
        ctx.n as usize,
        ctx.time_enc.as_ref(),
    ))
}

// ============================================================================
// Direction + monotonicity
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    Ascending,
    Descending,
}

fn detect_direction<T: PartialOrd>(
    coords: &[T],
    assume_sorted: bool,
) -> Option<Dir> {
    let first = coords.first()?;
    let last = coords.last()?;
    let dir = match first.partial_cmp(last)? {
        Ordering::Greater => Dir::Descending,
        _ => Dir::Ascending,
    };
    if assume_sorted {
        return Some(dir);
    }
    let monotonic = match dir {
        Dir::Ascending => coords
            .is_sorted_by(|a, b| a <= b),
        Dir::Descending => coords
            .is_sorted_by(|a, b| a >= b),
    };
    monotonic.then_some(dir)
}

// ============================================================================
// Native binary search via partition_point
// ============================================================================

/// Ascending-sorted slice search. Comparators are monomorphic — no
/// `match dir` inside the binary-search loop.
#[inline]
fn resolve_ascending<T: PartialOrd>(
    coords: &[T],
    start: &Bound<T>,
    end: &Bound<T>,
) -> Range<u64> {
    let s_idx = match start {
        Bound::Included(t) => {
            coords.partition_point(|v| v < t)
        }
        Bound::Excluded(t) => {
            coords.partition_point(|v| v <= t)
        }
        Bound::Unbounded => 0,
    };
    let e_idx = match end {
        Bound::Included(t) => {
            coords.partition_point(|v| v <= t)
        }
        Bound::Excluded(t) => {
            coords.partition_point(|v| v < t)
        }
        Bound::Unbounded => coords.len(),
    };
    (s_idx as u64)..(e_idx.max(s_idx) as u64)
}

/// Descending-sorted slice search. The result still spans contiguous
/// indices in the original coord order: in a descending array the
/// value-range *upper* governs the result *start* and vice versa, and the
/// strict/non-strict comparator flips.
#[inline]
fn resolve_descending<T: PartialOrd>(
    coords: &[T],
    start: &Bound<T>,
    end: &Bound<T>,
) -> Range<u64> {
    // start_idx uses the value-range UPPER bound (`end`), descending compare.
    let s_idx = match end {
        Bound::Included(t) => {
            coords.partition_point(|v| v > t)
        }
        Bound::Excluded(t) => {
            coords.partition_point(|v| v >= t)
        }
        Bound::Unbounded => 0,
    };
    // end_idx uses the value-range LOWER bound (`start`), descending compare.
    let e_idx = match start {
        Bound::Included(t) => {
            coords.partition_point(|v| v >= t)
        }
        Bound::Excluded(t) => {
            coords.partition_point(|v| v > t)
        }
        Bound::Unbounded => coords.len(),
    };
    (s_idx as u64)..(e_idx.max(s_idx) as u64)
}

#[inline]
fn resolve_against_sorted<T: PartialOrd>(
    coords: &[T],
    start: &Bound<T>,
    end: &Bound<T>,
    dir: Dir,
) -> Range<u64> {
    match dir {
        Dir::Ascending => {
            resolve_ascending(coords, start, end)
        }
        Dir::Descending => {
            resolve_descending(coords, start, end)
        }
    }
}

// ============================================================================
// Target normalization: CoordScalar bounds → native f64/i64 bounds
// ============================================================================

#[inline]
fn coord_to_f64(s: &CoordScalar) -> f64 {
    match s {
        CoordScalar::F64(v) => *v,
        CoordScalar::I64(v) => *v as f64,
        CoordScalar::U64(v) => *v as f64,
        CoordScalar::DatetimeNs(v) => *v as f64,
        CoordScalar::DurationNs(v) => *v as f64,
    }
}

#[inline]
fn coord_to_i64(s: &CoordScalar) -> i64 {
    match s {
        CoordScalar::I64(v) => *v,
        CoordScalar::U64(v) => *v as i64,
        CoordScalar::DatetimeNs(v) => *v,
        CoordScalar::DurationNs(v) => *v,
        CoordScalar::F64(v) => *v as i64,
    }
}

#[inline]
fn bound_map<T, F: Fn(&CoordScalar) -> T>(
    b: &Bound<CoordScalar>,
    f: F,
) -> Bound<T> {
    match b {
        Bound::Included(s) => Bound::Included(f(s)),
        Bound::Excluded(s) => Bound::Excluded(f(s)),
        Bound::Unbounded => Bound::Unbounded,
    }
}

// ============================================================================
// Expansion (interpolation neighbor / wrapping ghost)
// ============================================================================

#[inline]
fn pin_interpolation_without_neighbor_cells(
    vr: &ValueRangePresent,
    r: &Range<u64>,
) -> bool {
    vr.is_point_included_equal()
        && r.end.saturating_sub(r.start) == 1
}

/// Compute wrapping ghost ranges given a primary index range and dimension size.
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

#[inline]
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
                Vec::new()
            }
        }
        Expansion::InterpolationNeighbor => {
            if r.start < r.end
                && pin_interpolation_without_neighbor_cells(
                    vr, &r,
                )
            {
                return vec![r];
            }
            let start = r.start.saturating_sub(1);
            let end =
                r.end.saturating_add(1).min(n);
            if start < end {
                vec![start..end]
            } else {
                Vec::new()
            }
        }
        Expansion::WrappingGhost => {
            wrapping_ghost_ranges(r, n)
        }
    }
}

// ============================================================================
// Top-level entry points
// ============================================================================

fn resolve_via_buffer(
    buf: &CoordBuffer,
    dim: &IStr,
    dim_len: u64,
    vr: &ValueRangePresent,
    expansion: Expansion,
    assume_sorted: bool,
) -> Result<Vec<Range<u64>>, BackendError> {
    if buf.len() == 0 {
        return Ok(Vec::new());
    }
    let primary = match buf {
        CoordBuffer::F64(coords) => {
            let Some(dir) = detect_direction(
                coords,
                assume_sorted,
            ) else {
                return non_monotonic_fallback(
                    dim, dim_len, expansion,
                );
            };
            let start = bound_map(
                &vr.0,
                coord_to_f64,
            );
            let end = bound_map(
                &vr.1,
                coord_to_f64,
            );
            resolve_against_sorted(
                coords, &start, &end, dir,
            )
        }
        CoordBuffer::I64(coords) => {
            let Some(dir) = detect_direction(
                coords,
                assume_sorted,
            ) else {
                return non_monotonic_fallback(
                    dim, dim_len, expansion,
                );
            };
            let start = bound_map(
                &vr.0,
                coord_to_i64,
            );
            let end = bound_map(
                &vr.1,
                coord_to_i64,
            );
            resolve_against_sorted(
                coords, &start, &end, dir,
            )
        }
    };
    Ok(apply_expansion(
        primary, dim_len, vr, expansion,
    ))
}

fn non_monotonic_fallback(
    dim: &IStr,
    dim_len: u64,
    expansion: Expansion,
) -> Result<Vec<Range<u64>>, BackendError> {
    if expansion == Expansion::Exact {
        Ok(vec![0..dim_len])
    } else {
        Err(BackendError::other(format!(
            "dimension '{}' coordinate array is not monotonic",
            AsRef::<str>::as_ref(dim),
        )))
    }
}

fn resolve_index_only_or_missing(
    dim: &IStr,
    meta: &ZarrMeta,
    dim_len: u64,
    vr: &ValueRangePresent,
    expansion: Expansion,
) -> Option<Result<Vec<Range<u64>>, BackendError>>
{
    if meta.array_by_path_contains(dim) {
        return None;
    }
    Some(
        vr.index_range_for_index_dim(dim_len)
            .map(|r| {
                apply_expansion(
                    r, dim_len, vr, expansion,
                )
            })
            .ok_or_else(|| {
                BackendError::other(format!(
                    "dimension '{}' has no coordinate array",
                    AsRef::<str>::as_ref(dim),
                ))
            }),
    )
}

pub(crate) fn resolve_value_range_sync<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    dim: &IStr,
    meta: &ZarrMeta,
    dim_len: u64,
    vr: &ValueRangePresent,
    expansion: Expansion,
) -> Result<Vec<Range<u64>>, BackendError> {
    if let Some(r) = resolve_index_only_or_missing(
        dim, meta, dim_len, vr, expansion,
    ) {
        return r;
    }
    let ctx = DimCtx::from_meta(dim, meta)
        .expect("checked by resolve_index_only_or_missing");
    let buf =
        load_coord_buffer_sync(backend, &ctx)?;
    resolve_via_buffer(
        &buf,
        dim,
        ctx.n,
        vr,
        expansion,
        backend.assume_sorted_dim(dim),
    )
}

pub(crate) async fn resolve_value_range_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    dim: &IStr,
    meta: &ZarrMeta,
    dim_len: u64,
    vr: &ValueRangePresent,
    expansion: Expansion,
) -> Result<Vec<Range<u64>>, BackendError> {
    if let Some(r) = resolve_index_only_or_missing(
        dim, meta, dim_len, vr, expansion,
    ) {
        return r;
    }
    let ctx = DimCtx::from_meta(dim, meta)
        .expect("checked by resolve_index_only_or_missing");
    let buf =
        load_coord_buffer_async(backend, &ctx)
            .await?;
    resolve_via_buffer(
        &buf,
        dim,
        ctx.n,
        vr,
        expansion,
        backend.assume_sorted_dim(dim),
    )
}

// ============================================================================
// CoordResolver impls
// ============================================================================

/// Lookup-only resolver backed by a pre-populated `HashMap`. `HashMap`
/// scales O(1) for the large-N case (multi-row interpolation queries can
/// have hundreds of distinct keys) and the per-lookup `vr.clone()` is
/// dominated by the hash itself.
pub(crate) struct CachedResolver {
    table:
        HashMap<ResolveKey, Vec<Range<u64>>>,
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash,
)]
pub(crate) struct ResolveKey {
    pub(crate) dim: IStr,
    pub(crate) vr: ValueRangePresent,
    pub(crate) expansion: Expansion,
}

impl CachedResolver {
    pub(crate) fn from_table(
        table: HashMap<ResolveKey, Vec<Range<u64>>>,
    ) -> Self {
        Self { table }
    }
}

impl CoordResolver for CachedResolver {
    #[inline]
    fn resolve(
        &self,
        dim: IStr,
        _meta: &ZarrMeta,
        _dim_len: u64,
        vr: &ValueRangePresent,
        expansion: Expansion,
    ) -> Result<Vec<Range<u64>>, BackendError>
    {
        let key = ResolveKey {
            dim,
            vr: vr.clone(),
            expansion,
        };
        self.table
            .get(&key)
            .cloned()
            .ok_or_else(|| {
                BackendError::other(format!(
                    "missing pre-resolved entry for dim '{}'",
                    AsRef::<str>::as_ref(&dim),
                ))
            })
    }
}

/// Sync resolver that calls the backend on cache miss and memoizes per
/// `(dim, vr, expansion)`. Single-pass compile uses this directly so the
/// dry-run + parallel pre-resolve dance is avoided.
pub(crate) struct MemoizingSyncResolver<
    'b,
    B: ChunkedDataBackendSync,
> {
    backend: &'b B,
    cache: std::cell::RefCell<
        HashMap<ResolveKey, Vec<Range<u64>>>,
    >,
}

impl<'b, B: ChunkedDataBackendSync>
    MemoizingSyncResolver<'b, B>
{
    pub(crate) fn new(backend: &'b B) -> Self {
        Self {
            backend,
            cache: std::cell::RefCell::new(
                HashMap::new(),
            ),
        }
    }
}

impl<B: ChunkedDataBackendSync> CoordResolver
    for MemoizingSyncResolver<'_, B>
{
    fn resolve(
        &self,
        dim: IStr,
        meta: &ZarrMeta,
        dim_len: u64,
        vr: &ValueRangePresent,
        expansion: Expansion,
    ) -> Result<Vec<Range<u64>>, BackendError>
    {
        let key = ResolveKey {
            dim,
            vr: vr.clone(),
            expansion,
        };
        if let Some(ranges) =
            self.cache.borrow().get(&key)
        {
            return Ok(ranges.clone());
        }
        let ranges = resolve_value_range_sync(
            self.backend,
            &dim,
            meta,
            dim_len,
            vr,
            expansion,
        )?;
        self.cache
            .borrow_mut()
            .insert(key, ranges.clone());
        Ok(ranges)
    }
}

/// Collecting resolver used during the dry-run pre-walk.
pub(crate) struct CollectingResolver {
    pub(crate) keys:
        std::cell::RefCell<Vec<ResolveKey>>,
}

impl CollectingResolver {
    pub(crate) fn new() -> Self {
        Self {
            keys: std::cell::RefCell::new(
                Vec::new(),
            ),
        }
    }

    pub(crate) fn into_keys(
        self,
    ) -> Vec<ResolveKey> {
        self.keys.into_inner()
    }
}

impl CoordResolver for CollectingResolver {
    fn resolve(
        &self,
        dim: IStr,
        _meta: &ZarrMeta,
        dim_len: u64,
        vr: &ValueRangePresent,
        expansion: Expansion,
    ) -> Result<Vec<Range<u64>>, BackendError>
    {
        self.keys.borrow_mut().push(ResolveKey {
            dim,
            vr: vr.clone(),
            expansion,
        });
        Ok(vec![0..dim_len.max(1)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vr(
        start: Bound<i64>,
        end: Bound<i64>,
    ) -> ValueRangePresent {
        let map = |b: Bound<i64>| match b {
            Bound::Included(v) => Bound::Included(
                CoordScalar::I64(v),
            ),
            Bound::Excluded(v) => Bound::Excluded(
                CoordScalar::I64(v),
            ),
            Bound::Unbounded => Bound::Unbounded,
        };
        ValueRangePresent(map(start), map(end))
    }

    fn search_i64(
        coords: &[i64],
        v: &ValueRangePresent,
        dir: Dir,
    ) -> Range<u64> {
        let start =
            bound_map(&v.0, coord_to_i64);
        let end = bound_map(&v.1, coord_to_i64);
        resolve_against_sorted(
            coords, &start, &end, dir,
        )
    }

    fn asc_i64(n: i64) -> Vec<i64> {
        (0..n).collect()
    }

    fn desc_i64(n: i64) -> Vec<i64> {
        (0..n).rev().collect()
    }

    #[test]
    fn ascending_inclusive_inclusive() {
        let r = search_i64(
            &asc_i64(11),
            &vr(
                Bound::Included(3),
                Bound::Included(7),
            ),
            Dir::Ascending,
        );
        assert_eq!(r, 3..8);
    }

    #[test]
    fn ascending_exclusive_exclusive() {
        let r = search_i64(
            &asc_i64(11),
            &vr(
                Bound::Excluded(3),
                Bound::Excluded(7),
            ),
            Dir::Ascending,
        );
        assert_eq!(r, 4..7);
    }

    #[test]
    fn ascending_unbounded_start() {
        let r = search_i64(
            &asc_i64(11),
            &vr(
                Bound::Unbounded,
                Bound::Excluded(5),
            ),
            Dir::Ascending,
        );
        assert_eq!(r, 0..5);
    }

    #[test]
    fn ascending_unbounded_end() {
        let r = search_i64(
            &asc_i64(11),
            &vr(
                Bound::Included(5),
                Bound::Unbounded,
            ),
            Dir::Ascending,
        );
        assert_eq!(r, 5..11);
    }

    #[test]
    fn ascending_target_off_grid_f64() {
        let coords: Vec<f64> = (0..11)
            .map(|i| i as f64)
            .collect();
        let v = ValueRangePresent(
            Bound::Included(CoordScalar::F64(3.5)),
            Bound::Included(CoordScalar::F64(7.5)),
        );
        let start = bound_map(&v.0, coord_to_f64);
        let end = bound_map(&v.1, coord_to_f64);
        let r = resolve_against_sorted(
            &coords,
            &start,
            &end,
            Dir::Ascending,
        );
        assert_eq!(r, 4..8);
    }

    #[test]
    fn descending_inclusive_inclusive() {
        let r = search_i64(
            &desc_i64(11),
            &vr(
                Bound::Included(3),
                Bound::Included(7),
            ),
            Dir::Descending,
        );
        assert_eq!(r, 3..8);
    }

    #[test]
    fn descending_exclusive_exclusive() {
        let r = search_i64(
            &desc_i64(11),
            &vr(
                Bound::Excluded(3),
                Bound::Excluded(7),
            ),
            Dir::Descending,
        );
        assert_eq!(r, 4..7);
    }

    #[test]
    fn descending_unbounded_start() {
        let r = search_i64(
            &desc_i64(11),
            &vr(
                Bound::Unbounded,
                Bound::Excluded(5),
            ),
            Dir::Descending,
        );
        assert_eq!(r, 6..11);
    }

    #[test]
    fn detect_direction_assumed_skips_check() {
        let coords =
            vec![0i64, 5, 2, 9, 10];
        let dir =
            detect_direction(&coords, true).unwrap();
        assert_eq!(dir, Dir::Ascending);
    }

    #[test]
    fn detect_direction_rejects_non_monotonic() {
        let coords =
            vec![0i64, 5, 2, 9, 10];
        assert!(
            detect_direction(&coords, false)
                .is_none()
        );
    }

    #[test]
    fn ghost_ranges_interior_single_range() {
        assert_eq!(
            wrapping_ghost_ranges(50..55, 360),
            vec![47..58]
        );
    }

    #[test]
    fn ghost_ranges_near_start_adds_end_ghost() {
        let ranges =
            wrapping_ghost_ranges(1..5, 360);
        assert_eq!(ranges, vec![0..8, 357..360]);
    }

    #[test]
    fn ghost_ranges_at_start_adds_end_ghost() {
        let ranges =
            wrapping_ghost_ranges(0..3, 360);
        assert_eq!(ranges, vec![0..6, 357..360]);
    }

    #[test]
    fn ghost_ranges_near_end_adds_start_ghost() {
        let ranges =
            wrapping_ghost_ranges(356..360, 360);
        assert_eq!(ranges, vec![353..360, 0..3]);
    }

    #[test]
    fn ghost_ranges_at_end_adds_start_ghost() {
        let ranges =
            wrapping_ghost_ranges(358..360, 360);
        assert_eq!(ranges, vec![355..360, 0..3]);
    }

    #[test]
    fn ghost_ranges_small_dimension_both_ghosts() {
        let ranges =
            wrapping_ghost_ranges(2..4, 5);
        assert_eq!(
            ranges,
            vec![0..5, 2..5, 0..3]
        );
    }

    #[test]
    fn ghost_ranges_very_small_dimension() {
        assert_eq!(
            wrapping_ghost_ranges(0..2, 3),
            vec![0..3]
        );
    }

    #[test]
    fn ghost_ranges_exact_ghost_expansion_size() {
        assert_eq!(
            wrapping_ghost_ranges(1..2, 3),
            vec![0..3]
        );
    }

    #[test]
    fn ghost_expansion_constant() {
        assert_eq!(GHOST_EXPANSION, 3);
    }
}
