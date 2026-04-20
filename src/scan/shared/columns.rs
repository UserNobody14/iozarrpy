pub use crate::reader::{
    ColumnData, compute_strides,
};
pub(crate) use polars::prelude::*;
pub(crate) use std::collections::BTreeSet;
use std::ops::Range;

use smallvec::SmallVec;

use crate::chunk_plan::ChunkSubset;
use crate::meta::VarEncoding;
use crate::shared::IStr;

/// Represents which flat indices in a chunk are in-bounds.
///
/// For interior chunks (the common case), all elements are kept
/// and we avoid allocating a full index list — the O(chunk_len × ndim)
/// mask computation is replaced by an O(ndim) check.
pub enum KeepMask {
    /// All elements `0..len` are in bounds (interior chunk).
    All(usize),
    /// Only specific elements are in bounds (edge chunk).
    Sparse(Vec<usize>),
}

impl KeepMask {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            KeepMask::All(n) => *n,
            KeepMask::Sparse(v) => v.len(),
        }
    }
}

/// Compute which chunk indices to read for a variable based on
/// the primary chunk being processed.
pub fn compute_var_chunk_indices(
    primary_idx: &[u64],
    primary_chunk_shape: &[u64],
    primary_dims: &[IStr],
    var_dims: &[IStr],
    var_chunk_shape: &[u64],
    var_shape: &[u64],
) -> (Vec<u64>, Vec<u64>) {
    // If same dimensions, use same indices (possibly clamped)
    if var_dims.len() == primary_dims.len()
        && var_dims == primary_dims
    {
        // Clamp indices to valid range, inlining the
        // grid-shape computation to avoid an
        // intermediate Vec allocation.
        let clamped: Vec<u64> = primary_idx
            .iter()
            .zip(
                var_shape
                    .iter()
                    .zip(var_chunk_shape.iter()),
            )
            .map(|(&i, (&s, &c))| {
                let grid_size = s.div_ceil(c);
                i.min(grid_size.saturating_sub(1))
            })
            .collect();

        return (
            clamped,
            vec![0; var_dims.len()],
        );
    }

    // Different dimensions - map through dimension names
    let mut var_chunk_indices =
        Vec::with_capacity(var_dims.len());
    let mut var_offsets =
        Vec::with_capacity(var_dims.len());

    for (vd, var_dim) in
        var_dims.iter().enumerate()
    {
        if let Some(pd) = primary_dims
            .iter()
            .position(|d| d == var_dim)
        {
            // This dimension exists in primary
            let primary_origin = primary_idx[pd]
                * primary_chunk_shape[pd];
            let var_chunk_idx = primary_origin
                / var_chunk_shape[vd];
            let offset = primary_origin
                % var_chunk_shape[vd];
            var_chunk_indices.push(var_chunk_idx);
            var_offsets.push(offset);
        } else {
            // Dimension doesn't exist in primary, use chunk 0
            var_chunk_indices.push(0);
            var_offsets.push(0);
        }
    }

    (var_chunk_indices, var_offsets)
}

pub fn should_include_column(
    name: &IStr,
    with_columns: Option<&BTreeSet<IStr>>,
) -> bool {
    with_columns
        .map(|s| s.contains(name))
        .unwrap_or(true)
}

/// Build a coordinate column for the DataFrame.
///
/// For `KeepMask::All` (interior chunks), uses
/// `repeat_tile` to fill the output via memcpy,
/// completely avoiding the per-element integer
/// division that `gather_by` requires.
///
/// For `KeepMask::Sparse` (edge chunks), falls back
/// to `gather_by` with inline index computation.
pub fn build_coord_column(
    dim_name: &str,
    dim_idx: usize,
    keep: &KeepMask,
    strides: &[u64],
    chunk_shape: &[u64],
    origin: &[u64],
    coord_data: Option<&ColumnData>,
    encoding: Option<&VarEncoding>,
) -> Column {
    let stride = strides[dim_idx] as usize;
    let cs = chunk_shape[dim_idx] as usize;
    let origin_val = origin[dim_idx] as i64;

    if let Some(coord) = coord_data {
        if let Some(enc) = encoding {
            match enc {
                VarEncoding::Time(te) => {
                    let decoded =
                        coord.map_i64(|v| {
                            te.decode(v)
                        });
                    let gathered = match keep {
                        KeepMask::All(n) => {
                            let tile_count = *n
                                / (cs * stride);
                            decoded.repeat_tile(
                                stride,
                                tile_count,
                            )
                        }
                        KeepMask::Sparse(idx) => {
                            let local_idx =
                                |row: usize| {
                                    (row / stride)
                                        % cs
                                };
                            decoded.gather_by(
                                idx.len(),
                                |i| {
                                    local_idx(
                                        idx[i],
                                    )
                                },
                            )
                        }
                    };
                    let series_uncast = gathered
                        .into_series(dim_name);
                    return series_uncast
                        .cast(
                            &te.to_polars_dtype(),
                        )
                        .unwrap_or(series_uncast)
                        .into();
                }
                VarEncoding::ScaleOffset {
                    scale_factor,
                    add_offset,
                    fill_value,
                } => {
                    let decoded = coord
                        .to_f64_scaled(
                            *scale_factor,
                            *add_offset,
                            *fill_value,
                        );
                    let gathered = match keep {
                        KeepMask::All(n) => {
                            let tile_count = *n
                                / (cs * stride);
                            decoded.repeat_tile(
                                stride,
                                tile_count,
                            )
                        }
                        KeepMask::Sparse(idx) => {
                            let local_idx =
                                |row: usize| {
                                    (row / stride)
                                        % cs
                                };
                            decoded.gather_by(
                                idx.len(),
                                |i| {
                                    local_idx(
                                        idx[i],
                                    )
                                },
                            )
                        }
                    };
                    return gathered
                        .into_series(dim_name)
                        .into();
                }
            }
        }

        // Coord data, no encoding
        let gathered = match keep {
            KeepMask::All(n) => {
                let tile_count =
                    *n / (cs * stride);
                coord.repeat_tile(
                    stride, tile_count,
                )
            }
            KeepMask::Sparse(idx) => {
                let local_idx = |row: usize| {
                    (row / stride) % cs
                };
                coord.gather_by(idx.len(), |i| {
                    local_idx(idx[i])
                })
            }
        };
        return gathered
            .into_series(dim_name)
            .into();
    }

    // No coord data: integer index + origin
    match keep {
        KeepMask::All(n) => {
            let small: Vec<i64> = (0..cs as i64)
                .map(|j| j + origin_val)
                .collect();
            let tile_count = *n / (cs * stride);
            ColumnData::I64(small)
                .repeat_tile(stride, tile_count)
                .into_series(dim_name)
                .into()
        }
        KeepMask::Sparse(idx) => {
            let local_idx =
                |row: usize| (row / stride) % cs;
            let values: Vec<i64> = idx
                .iter()
                .map(|&row| {
                    local_idx(row) as i64
                        + origin_val
                })
                .collect();
            Series::new(dim_name.into(), values)
                .into()
        }
    }
}

/// Compute the in-bounds mask for edge chunk handling,
/// optionally constrained by a chunk-local subset.
///
/// Both edge clipping (per-dim `array_shape[d] - origin[d]`) and the
/// chunk-local subset reduce to a contiguous per-dim range of valid
/// `local` indices. We collapse them into one per-dim range and enumerate
/// the cartesian product in C-order (rightmost dim varies fastest, so
/// emitted flat indices are monotonically increasing).
///
/// Cost is `O(num_kept × ndim)` (only additions / comparisons), versus
/// the previous `O(chunk_len × ndim)` with a division per element. For
/// dense interior chunks we still short-circuit to `KeepMask::All`.
pub fn compute_in_bounds_mask(
    chunk_len: usize,
    chunk_shape: &[u64],
    origin: &[u64],
    array_shape: &[u64],
    strides: &[u64],
    chunk_subset: Option<&ChunkSubset>,
) -> KeepMask {
    let ndim = chunk_shape.len();

    // Compute per-dim effective ranges (intersection of edge clip and subset).
    // Tracks whether any clipping/subsetting was applied; if not, we can return
    // `KeepMask::All(chunk_len)` without enumerating.
    let mut ranges: SmallVec<[Range<u64>; 4]> =
        SmallVec::with_capacity(ndim);
    let mut total: u64 = 1;
    let mut full = true;
    for d in 0..ndim {
        let edge_end = chunk_shape[d].min(
            array_shape[d]
                .saturating_sub(origin[d]),
        );
        let (start, end) = if let Some(sub) =
            chunk_subset
        {
            let s = sub.ranges[d].start;
            let e =
                sub.ranges[d].end.min(edge_end);
            (s, e)
        } else {
            (0u64, edge_end)
        };
        if start >= end {
            return KeepMask::Sparse(Vec::new());
        }
        if start != 0 || end != chunk_shape[d] {
            full = false;
        }
        total = total.saturating_mul(end - start);
        ranges.push(start..end);
    }

    if full {
        return KeepMask::All(chunk_len);
    }

    let total_usize = total as usize;
    let mut keep: Vec<usize> =
        Vec::with_capacity(total_usize);

    // Cartesian-product enumeration in C-order. We maintain the
    // running flat index (sum of `local[d] * strides[d]`) so the
    // inner emit loop on the fastest-varying dim is just an
    // addition by `strides[ndim-1]`.
    let mut local: SmallVec<[u64; 4]> =
        ranges.iter().map(|r| r.start).collect();
    let mut row: u64 = local
        .iter()
        .zip(strides.iter())
        .map(|(l, s)| l * s)
        .sum();

    let last_d = ndim - 1;
    let last_stride = strides[last_d];
    let last_start = ranges[last_d].start;
    let last_end = ranges[last_d].end;

    'outer: loop {
        // Emit every row along the fastest-varying axis without
        // touching the per-dim cursor: stride is constant.
        let mut r = row;
        for _ in last_start..last_end {
            keep.push(r as usize);
            r += last_stride;
        }

        if last_d == 0 {
            break;
        }

        // Carry-style increment of the slower-varying dims.
        let mut d = last_d;
        loop {
            if d == 0 {
                break 'outer;
            }
            d -= 1;
            row += strides[d];
            local[d] += 1;
            if local[d] < ranges[d].end {
                break;
            }
            // Roll this dim back to its start; subtract the
            // contribution we just added so `row` stays consistent.
            let span = local[d] - ranges[d].start;
            row -= span * strides[d];
            local[d] = ranges[d].start;
        }
    }

    KeepMask::Sparse(keep)
}

/// Apply encoding to a gathered `ColumnData`, returning
/// a Polars `Column`. For `Time`, maps i64 then casts.
/// For `ScaleOffset`, decodes to f64. No-op if `None`.
pub(crate) fn apply_encoding(
    data: ColumnData,
    name: &str,
    encoding: Option<&VarEncoding>,
) -> Column {
    match encoding {
        Some(VarEncoding::Time(te)) => {
            let decoded =
                data.map_i64(|v| te.decode(v));
            let series_uncast =
                decoded.into_series(name);
            series_uncast
                .cast(&te.to_polars_dtype())
                .unwrap_or(series_uncast)
                .into()
        }
        Some(VarEncoding::ScaleOffset {
            scale_factor,
            add_offset,
            fill_value,
        }) => data
            .to_f64_scaled(
                *scale_factor,
                *add_offset,
                *fill_value,
            )
            .into_series(name)
            .into(),
        None => data.into_series(name).into(),
    }
}

/// Build a variable column for the DataFrame.
///
/// Takes `data` by value so the common fast path
/// (`KeepMask::All` + same shape) can hand the
/// buffer directly to Polars with zero copying.
pub fn build_var_column(
    name: &IStr,
    data: Arc<ColumnData>,
    var_dims: &[IStr],
    var_chunk_shape: &[u64],
    var_offsets: &[u64],
    primary_dims: &[IStr],
    primary_chunk_shape: &[u64],
    primary_strides: &[u64],
    keep: &KeepMask,
    encoding: Option<&VarEncoding>,
) -> Column {
    // Check if we can use direct indexing
    let same_dims = var_dims.len()
        == primary_dims.len()
        && var_dims == primary_dims;
    let same_chunk_shape =
        var_chunk_shape == primary_chunk_shape;
    let zero_offsets =
        var_offsets.iter().all(|&o| o == 0);

    if same_dims
        && same_chunk_shape
        && zero_offsets
    {
        if encoding.is_none() {
            return match keep {
                KeepMask::All(_) => data
                    .borrow_into_series(
                        name.as_ref(),
                    )
                    .into(),
                KeepMask::Sparse(idx) => data
                    .take_indices(idx)
                    .into_series(name.as_ref())
                    .into(),
            };
        }

        let gathered = match keep {
            KeepMask::All(_) => {
                // Cannot zero-copy when encoding
                // needs to transform data.
                (*data).clone()
            }
            KeepMask::Sparse(idx) => {
                data.take_indices(idx)
            }
        };
        return apply_encoding(
            gathered,
            name.as_ref(),
            encoding,
        );
    }

    // Slow path: map indices through dimension
    // differences. Uses gather_by to fuse index
    // computation and data gathering into a single
    // pass (avoids intermediate Vec<usize>).
    let dim_mapping: Vec<Option<usize>> =
        primary_dims
            .iter()
            .map(|pd| {
                var_dims
                    .iter()
                    .position(|vd| vd == pd)
            })
            .collect();
    let var_strides =
        compute_strides(var_chunk_shape);
    let var_data_len = data.len();

    let compute_var_idx = |row: usize| -> usize {
        let mut var_idx: u64 = 0;
        for (primary_d, maybe_var_d) in
            dim_mapping.iter().enumerate()
        {
            if let Some(var_d) = *maybe_var_d {
                let local = (row as u64
                    / primary_strides[primary_d])
                    % primary_chunk_shape
                        [primary_d];
                let var_local = if same_dims
                    && var_chunk_shape.len()
                        > var_d
                {
                    local.min(
                        var_chunk_shape[var_d]
                            .saturating_sub(1),
                    )
                } else {
                    local
                };
                let local_with_offset = var_local
                    + var_offsets[var_d];
                var_idx += local_with_offset
                    * var_strides[var_d];
            }
        }
        (var_idx as usize)
            .min(var_data_len.saturating_sub(1))
    };

    let gathered = match keep {
        KeepMask::All(n) => data
            .gather_by(*n, |i| {
                compute_var_idx(i)
            }),
        KeepMask::Sparse(idx) => data
            .gather_by(idx.len(), |i| {
                compute_var_idx(idx[i])
            }),
    };
    apply_encoding(
        gathered,
        name.as_ref(),
        encoding,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    fn legacy_mask(
        chunk_len: usize,
        chunk_shape: &[u64],
        origin: &[u64],
        array_shape: &[u64],
        strides: &[u64],
        chunk_subset: Option<&ChunkSubset>,
    ) -> Vec<usize> {
        let is_interior = chunk_shape
            .iter()
            .zip(origin.iter())
            .zip(array_shape.iter())
            .all(|((cs, o), a)| o + cs <= *a);

        if is_interior && chunk_subset.is_none() {
            return (0..chunk_len).collect();
        }

        let mut keep: Vec<usize> = Vec::new();
        for row in 0..chunk_len {
            let mut ok = true;
            for d in 0..chunk_shape.len() {
                let local = (row as u64
                    / strides[d])
                    % chunk_shape[d];
                if !is_interior {
                    let global =
                        origin[d] + local;
                    if global >= array_shape[d] {
                        ok = false;
                        break;
                    }
                }
                if let Some(sub) = chunk_subset
                    && (local
                        < sub.ranges[d].start
                        || local
                            >= sub.ranges[d].end)
                {
                    ok = false;
                    break;
                }
            }
            if ok {
                keep.push(row);
            }
        }
        keep
    }

    fn assert_matches_legacy(
        chunk_shape: &[u64],
        origin: &[u64],
        array_shape: &[u64],
        chunk_subset: Option<&ChunkSubset>,
    ) {
        let chunk_len: usize =
            chunk_shape.iter().product::<u64>()
                as usize;
        let strides =
            compute_strides(chunk_shape);
        let expected = legacy_mask(
            chunk_len,
            chunk_shape,
            origin,
            array_shape,
            &strides,
            chunk_subset,
        );
        let mask = compute_in_bounds_mask(
            chunk_len,
            chunk_shape,
            origin,
            array_shape,
            &strides,
            chunk_subset,
        );
        let actual: Vec<usize> = match mask {
            KeepMask::All(n) => (0..n).collect(),
            KeepMask::Sparse(v) => v,
        };
        assert_eq!(
            actual, expected,
            "mismatch for shape={chunk_shape:?} origin={origin:?} array={array_shape:?} subset={chunk_subset:?}"
        );
    }

    #[test]
    fn interior_no_subset_is_all() {
        assert_matches_legacy(
            &[10, 10, 10],
            &[20, 30, 10],
            &[100, 100, 50],
            None,
        );
    }

    #[test]
    fn edge_no_subset() {
        assert_matches_legacy(
            &[10, 10, 10],
            &[90, 90, 40],
            &[100, 100, 50],
            None,
        );
    }

    #[test]
    fn interior_with_subset_partial() {
        let subset = ChunkSubset {
            ranges: smallvec![
                0u64..3,
                2u64..5,
                4u64..7,
            ],
        };
        assert_matches_legacy(
            &[10, 10, 10],
            &[20, 30, 10],
            &[100, 100, 50],
            Some(&subset),
        );
    }

    #[test]
    fn edge_with_subset() {
        let subset = ChunkSubset {
            ranges: smallvec![
                0u64..15,
                3u64..8,
                0u64..7,
            ],
        };
        assert_matches_legacy(
            &[10, 10, 10],
            &[90, 90, 40],
            &[100, 100, 50],
            Some(&subset),
        );
    }

    #[test]
    fn subset_covers_full_chunk_returns_all() {
        let subset = ChunkSubset {
            ranges: smallvec![
                0u64..10,
                0u64..10,
                0u64..10,
            ],
        };
        let mask = compute_in_bounds_mask(
            1000,
            &[10, 10, 10],
            &[20, 30, 10],
            &[100, 100, 50],
            &compute_strides(&[10, 10, 10]),
            Some(&subset),
        );
        assert!(matches!(
            mask,
            KeepMask::All(1000)
        ));
    }

    #[test]
    fn empty_intersection_returns_empty() {
        // Subset entirely beyond the edge-clipped extent.
        let subset = ChunkSubset {
            ranges: smallvec![
                0u64..3,
                15u64..20,
                0u64..3,
            ],
        };
        let mask = compute_in_bounds_mask(
            1000,
            &[10, 10, 10],
            &[90, 90, 40],
            &[100, 100, 50],
            &compute_strides(&[10, 10, 10]),
            Some(&subset),
        );
        match mask {
            KeepMask::Sparse(v) => {
                assert!(v.is_empty())
            }
            KeepMask::All(_) => {
                panic!("expected empty Sparse")
            }
        }
    }

    #[test]
    fn one_dim_subset() {
        let subset = ChunkSubset {
            ranges: smallvec![3u64..7],
        };
        assert_matches_legacy(
            &[10],
            &[5],
            &[20],
            Some(&subset),
        );
    }
}
