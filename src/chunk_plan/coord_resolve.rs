//! Coordinate-to-index resolution.
//!
//! Maps a [`ValueRangePresent`] over a dim's coordinate values to a
//! `Range<u64>` of array indices via [`slice::partition_point`] on the
//! materialized coordinate array. Backends may declare a dim already-sorted
//! (via [`ChunkedDataBackendSync::assume_sorted_dim`]) to skip the
//! [`slice::is_sorted_by`] monotonicity probe.

use std::cmp::Ordering;
use std::ops::{Bound, Range};

use crate::meta::{TimeEncoding, ZarrMeta};
use crate::reader::ColumnData;
use crate::shared::{
    ChunkedDataBackendAsync, ChunkedDataBackendSync,
    IStr,
};

use super::indexing::types::{
    CoordScalar, ValueRangePresent,
};

#[derive(Debug, Clone)]
pub enum ResolutionError {
    Unresolvable(String),
}

impl std::fmt::Display for ResolutionError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let ResolutionError::Unresolvable(msg) =
            self;
        write!(f, "unresolvable: {msg}")
    }
}

impl std::error::Error for ResolutionError {}

/// Selects how a [`ValueRangePresent`] is converted into one or more index
/// ranges after the binary search.
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
// Coord array fetch + decode
// ============================================================================

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

fn decode_chunk_into(
    chunk: &ColumnData,
    out: &mut Vec<CoordScalar>,
    te: Option<&TimeEncoding>,
) {
    let decode_float = |x: f64| match te {
        Some(enc) => enc
            .decode_f64(x)
            .map(|ns| time_scalar(ns, enc))
            .unwrap_or(CoordScalar::F64(x)),
        None => CoordScalar::F64(x),
    };
    match chunk {
        ColumnData::F64(v) => out.extend(
            v.iter().copied().map(decode_float),
        ),
        ColumnData::F32(v) => out.extend(
            v.iter()
                .copied()
                .map(|x| decode_float(x as f64)),
        ),
        _ => out.extend((0..chunk.len()).map(|i| {
            super::exprs::apply_time_encoding(
                chunk.get_i64(i).unwrap_or(0),
                te,
            )
        })),
    }
}

#[inline]
fn time_scalar(
    ns: i64,
    enc: &TimeEncoding,
) -> CoordScalar {
    if enc.is_duration {
        CoordScalar::DurationNs(ns)
    } else {
        CoordScalar::DatetimeNs(ns)
    }
}

fn load_coord_array_sync<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    ctx: &DimCtx,
) -> Result<Vec<CoordScalar>, ResolutionError> {
    let mut out =
        Vec::with_capacity(ctx.n as usize);
    let te = ctx.time_enc.as_ref();
    for ci in 0..ctx.n_chunks() {
        let chunk = backend
            .read_chunk_sync(
                &ctx.array_path,
                &[ci],
            )
            .map_err(|e| {
                ResolutionError::Unresolvable(
                    e.to_string(),
                )
            })?;
        decode_chunk_into(&chunk, &mut out, te);
    }
    out.truncate(ctx.n as usize);
    Ok(out)
}

async fn load_coord_array_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    ctx: &DimCtx,
) -> Result<Vec<CoordScalar>, ResolutionError> {
    let chunks = futures::future::try_join_all(
        (0..ctx.n_chunks()).map(|ci| async move {
            backend
                .read_chunk_async(
                    &ctx.array_path,
                    &[ci],
                )
                .await
        }),
    )
    .await
    .map_err(|e| {
        ResolutionError::Unresolvable(
            e.to_string(),
        )
    })?;
    let mut out =
        Vec::with_capacity(ctx.n as usize);
    let te = ctx.time_enc.as_ref();
    for chunk in &chunks {
        decode_chunk_into(chunk, &mut out, te);
    }
    out.truncate(ctx.n as usize);
    Ok(out)
}

// ============================================================================
// Direction + monotonicity
// ============================================================================

/// Sort direction of a coordinate array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    Ascending,
    Descending,
}

impl Dir {
    fn from_endpoints(
        first: &CoordScalar,
        last: &CoordScalar,
    ) -> Option<Self> {
        match first.partial_cmp(last)? {
            Ordering::Greater => {
                Some(Dir::Descending)
            }
            _ => Some(Dir::Ascending),
        }
    }
}

fn detect_direction(
    coords: &[CoordScalar],
    assume_sorted: bool,
) -> Result<Dir, ResolutionError> {
    let (first, last) = match (
        coords.first(),
        coords.last(),
    ) {
        (Some(f), Some(l)) => (f, l),
        _ => {
            return Err(
                ResolutionError::Unresolvable(
                    "empty coordinate array".into(),
                ),
            );
        }
    };
    let dir = Dir::from_endpoints(first, last)
        .ok_or_else(|| {
            ResolutionError::Unresolvable(
                "non-comparable coordinate values"
                    .into(),
            )
        })?;
    if assume_sorted {
        return Ok(dir);
    }
    let monotonic = match dir {
        Dir::Ascending => coords
            .is_sorted_by(|a, b| a <= b),
        Dir::Descending => coords
            .is_sorted_by(|a, b| a >= b),
    };
    if monotonic {
        Ok(dir)
    } else {
        Err(ResolutionError::Unresolvable(
            "coordinate array is not monotonic"
                .into(),
        ))
    }
}

// ============================================================================
// Binary search via partition_point
// ============================================================================

/// Convert a [`ValueRangePresent`] to an index range against a sorted
/// coordinate slice.
///
/// For descending coords the value-range bounds are swapped: the in-range
/// slice is still contiguous, but the value-range *upper* bound governs the
/// result *start* (the largest value lives at the lowest index), and the
/// *lower* bound governs the result *end*. The `lt` / `le` closures
/// likewise flip to the descending comparison so the rest of the function
/// reads identically to the ascending case.
fn resolve_against_sorted(
    coords: &[CoordScalar],
    vr: &ValueRangePresent,
    dir: Dir,
) -> Range<u64> {
    let (start_b, end_b) = match dir {
        Dir::Ascending => (
            vr.0.as_ref(),
            vr.1.as_ref(),
        ),
        Dir::Descending => (
            vr.1.as_ref(),
            vr.0.as_ref(),
        ),
    };
    // partition_point walks while the predicate holds; these are direction-
    // aware "v is before target" predicates over `PartialOrd`. NaN/None
    // results return `false`, which keeps the search safely conservative.
    let lt = |v: &CoordScalar, t: &CoordScalar| {
        match dir {
            Dir::Ascending => v < t,
            Dir::Descending => v > t,
        }
    };
    let le = |v: &CoordScalar, t: &CoordScalar| {
        match dir {
            Dir::Ascending => v <= t,
            Dir::Descending => v >= t,
        }
    };

    let start = match start_b {
        Bound::Included(t) => coords
            .partition_point(|v| lt(v, t)),
        Bound::Excluded(t) => coords
            .partition_point(|v| le(v, t)),
        Bound::Unbounded => 0,
    };
    let end = match end_b {
        Bound::Included(t) => coords
            .partition_point(|v| le(v, t)),
        Bound::Excluded(t) => coords
            .partition_point(|v| lt(v, t)),
        Bound::Unbounded => coords.len(),
    };
    (start as u64)..(end.max(start) as u64)
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
///
/// Returns a `Vec<Range<u64>>` containing the primary range (expanded by
/// [`GHOST_EXPANSION`]) plus up to two ghost ranges at the opposite boundary
/// when the primary range is within `GHOST_EXPANSION` of either edge.
pub(crate) fn wrapping_ghost_ranges(
    primary: Range<u64>,
    n: u64,
) -> Vec<Range<u64>> {
    let start =
        primary.start.saturating_sub(GHOST_EXPANSION);
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
        // Interpolation needs ±1 cells even when the in-range slice is
        // empty (target lies strictly between two grid points), so we don't
        // short-circuit on `r.start == r.end` here.
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
// Top-level: index-only fast path + sync/async drivers
// ============================================================================

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

fn resolve_with_coords(
    coords: &[CoordScalar],
    dim_len: u64,
    vr: &ValueRangePresent,
    expansion: Expansion,
    assume_sorted: bool,
) -> Result<Vec<Range<u64>>, ResolutionError> {
    if coords.is_empty() {
        return Ok(Vec::new());
    }
    match detect_direction(coords, assume_sorted) {
        Ok(dir) => Ok(apply_expansion(
            resolve_against_sorted(coords, vr, dir),
            dim_len,
            vr,
            expansion,
        )),
        // Non-monotonic coords: scan-everything is still safe for plain
        // filtering, but interpolation needs an actual ordering.
        Err(_) if expansion == Expansion::Exact => {
            Ok(vec![0..dim_len])
        }
        Err(e) => Err(e),
    }
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
) -> Result<Vec<Range<u64>>, ResolutionError> {
    if let Some(r) = try_resolve_index_only(
        dim, meta, dim_len, vr,
    ) {
        return Ok(apply_expansion(
            r, dim_len, vr, expansion,
        ));
    }
    let Some(ctx) = DimCtx::from_meta(dim, meta)
    else {
        return missing_coord_array(
            dim, dim_len, expansion,
        );
    };
    let coords =
        load_coord_array_sync(backend, &ctx)?;
    resolve_with_coords(
        &coords,
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
) -> Result<Vec<Range<u64>>, ResolutionError> {
    if let Some(r) = try_resolve_index_only(
        dim, meta, dim_len, vr,
    ) {
        return Ok(apply_expansion(
            r, dim_len, vr, expansion,
        ));
    }
    let Some(ctx) = DimCtx::from_meta(dim, meta)
    else {
        return missing_coord_array(
            dim, dim_len, expansion,
        );
    };
    let coords =
        load_coord_array_async(backend, &ctx)
            .await?;
    resolve_with_coords(
        &coords,
        ctx.n,
        vr,
        expansion,
        backend.assume_sorted_dim(dim),
    )
}

fn missing_coord_array(
    dim: &IStr,
    dim_len: u64,
    expansion: Expansion,
) -> Result<Vec<Range<u64>>, ResolutionError> {
    match expansion {
        Expansion::Exact => Ok(vec![0..dim_len]),
        _ => Err(ResolutionError::Unresolvable(
            format!(
                "dimension '{}' has no coordinate array",
                AsRef::<str>::as_ref(dim),
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asc(n: i64) -> Vec<CoordScalar> {
        (0..n).map(CoordScalar::I64).collect()
    }

    fn desc(n: i64) -> Vec<CoordScalar> {
        (0..n).rev().map(CoordScalar::I64).collect()
    }

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

    #[test]
    fn ascending_inclusive_inclusive() {
        let r = resolve_against_sorted(
            &asc(11),
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
        let r = resolve_against_sorted(
            &asc(11),
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
        let r = resolve_against_sorted(
            &asc(11),
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
        let r = resolve_against_sorted(
            &asc(11),
            &vr(
                Bound::Included(5),
                Bound::Unbounded,
            ),
            Dir::Ascending,
        );
        assert_eq!(r, 5..11);
    }

    #[test]
    fn ascending_target_off_grid() {
        // Range [3.5, 7.5] over int coords: result should bracket the
        // values strictly between, here {4,5,6,7}.
        let coords: Vec<_> = (0..11)
            .map(|i| CoordScalar::F64(i as f64))
            .collect();
        let v = ValueRangePresent(
            Bound::Included(CoordScalar::F64(3.5)),
            Bound::Included(CoordScalar::F64(7.5)),
        );
        let r = resolve_against_sorted(
            &coords,
            &v,
            Dir::Ascending,
        );
        assert_eq!(r, 4..8);
    }

    #[test]
    fn descending_inclusive_inclusive() {
        // Descending [10..0] with [3,7] -> indices 3..8 (vals 7,6,5,4,3).
        let r = resolve_against_sorted(
            &desc(11),
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
        // Descending [10..0] with (3,7) -> indices 4..7 (vals 6,5,4).
        let r = resolve_against_sorted(
            &desc(11),
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
        // Descending [10..0] with (-inf, 5) -> indices 6..11 (vals 4..0).
        let r = resolve_against_sorted(
            &desc(11),
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
        // Non-monotonic coords; assume_sorted=true uses only endpoints.
        let coords = vec![
            CoordScalar::I64(0),
            CoordScalar::I64(5),
            CoordScalar::I64(2),
            CoordScalar::I64(9),
            CoordScalar::I64(10),
        ];
        let dir =
            detect_direction(&coords, true).unwrap();
        assert_eq!(dir, Dir::Ascending);
    }

    #[test]
    fn detect_direction_rejects_non_monotonic() {
        let coords = vec![
            CoordScalar::I64(0),
            CoordScalar::I64(5),
            CoordScalar::I64(2),
            CoordScalar::I64(9),
            CoordScalar::I64(10),
        ];
        assert!(
            detect_direction(&coords, false).is_err()
        );
    }

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
    fn ghost_ranges_small_dimension_both_ghosts() {
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
    fn ghost_expansion_constant() {
        assert_eq!(GHOST_EXPANSION, 3);
    }
}
