pub(crate) use crate::reader::{
    ColumnData, compute_strides,
};
pub(crate) use polars::prelude::*;
pub(crate) use std::collections::BTreeSet;

use crate::IStr;

/// Represents which flat indices in a chunk are in-bounds.
///
/// For interior chunks (the common case), all elements are kept
/// and we avoid allocating a full index list — the O(chunk_len × ndim)
/// mask computation is replaced by an O(ndim) check.
pub(crate) enum KeepMask {
    /// All elements `0..len` are in bounds (interior chunk).
    All(usize),
    /// Only specific elements are in bounds (edge chunk).
    Sparse(Vec<usize>),
}

impl KeepMask {
    pub(crate) fn len(&self) -> usize {
        match self {
            KeepMask::All(n) => *n,
            KeepMask::Sparse(v) => v.len(),
        }
    }
}

/// Compute which chunk indices to read for a variable based on
/// the primary chunk being processed.
pub(crate) fn compute_var_chunk_indices(
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
                let grid_size = (s + c - 1) / c;
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

/// Compute actual chunk shape (handling edge chunks).
pub(crate) fn compute_actual_chunk_shape(
    chunk_indices: &[u64],
    regular_chunk_shape: &[u64],
    array_shape: &[u64],
) -> Vec<u64> {
    chunk_indices
        .iter()
        .zip(regular_chunk_shape.iter())
        .zip(array_shape.iter())
        .map(|((idx, chunk_size), array_size)| {
            let start = idx * chunk_size;
            let remaining =
                array_size.saturating_sub(start);
            (*chunk_size).min(remaining)
        })
        .collect()
}

pub(crate) fn should_include_column(
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
pub(crate) fn build_coord_column(
    dim_name: &str,
    dim_idx: usize,
    keep: &KeepMask,
    strides: &[u64],
    chunk_shape: &[u64],
    origin: &[u64],
    coord_data: Option<&ColumnData>,
    time_encoding: Option<
        &crate::meta::TimeEncoding,
    >,
) -> Column {
    let stride = strides[dim_idx] as usize;
    let cs = chunk_shape[dim_idx] as usize;
    let origin_val = origin[dim_idx] as i64;

    if let Some(coord) = coord_data {
        if let Some(te) = time_encoding {
            // Decode coord values first (only cs
            // elements), then expand to output size.
            let decoded =
                coord.map_i64(|v| te.decode(v));
            let gathered = match keep {
                KeepMask::All(n) => {
                    let tile_count =
                        *n / (cs * stride);
                    decoded.repeat_tile(
                        stride, tile_count,
                    )
                }
                KeepMask::Sparse(idx) => {
                    let local_idx =
                        |row: usize| {
                            (row / stride) % cs
                        };
                    decoded.gather_by(
                        idx.len(),
                        |i| local_idx(idx[i]),
                    )
                }
            };
            let series_uncast = gathered
                .into_series(dim_name.into());
            return series_uncast
                .cast(&te.to_polars_dtype())
                .unwrap_or(series_uncast)
                .into();
        }

        // Coord data, no time encoding
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
            .into_series(dim_name.into())
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
                .into_series(dim_name.into())
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

/// Compute the in-bounds mask for edge chunk handling.
///
/// For interior chunks (all elements in-bounds), returns
/// `KeepMask::All` in O(ndim) without iterating over elements.
/// Only edge chunks pay the O(chunk_len × ndim) cost.
pub(crate) fn compute_in_bounds_mask(
    chunk_len: usize,
    chunk_shape: &[u64],
    origin: &[u64],
    array_shape: &[u64],
    strides: &[u64],
) -> KeepMask {
    // O(ndim) check: is every dimension fully in-bounds?
    let is_interior = chunk_shape
        .iter()
        .zip(origin.iter())
        .zip(array_shape.iter())
        .all(|((cs, o), a)| o + cs <= *a);

    if is_interior {
        return KeepMask::All(chunk_len);
    }

    // Edge chunk: compute sparse mask
    let mut keep: Vec<usize> =
        Vec::with_capacity(chunk_len);
    for row in 0..chunk_len {
        let mut ok = true;
        for d in 0..chunk_shape.len() {
            let local = (row as u64 / strides[d])
                % chunk_shape[d];
            let global = origin[d] + local;
            if global >= array_shape[d] {
                ok = false;
                break;
            }
        }
        if ok {
            keep.push(row);
        }
    }
    KeepMask::Sparse(keep)
}

/// Build a variable column for the DataFrame.
///
/// Takes `data` by value so the common fast path
/// (`KeepMask::All` + same shape) can hand the
/// buffer directly to Polars with zero copying.
pub(crate) fn build_var_column(
    name: &IStr,
    data: ColumnData,
    var_dims: &[IStr],
    var_chunk_shape: &[u64],
    var_offsets: &[u64],
    primary_dims: &[IStr],
    primary_chunk_shape: &[u64],
    primary_strides: &[u64],
    keep: &KeepMask,
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
        return match keep {
            KeepMask::All(_) => {
                // Zero-copy: hand buffer directly
                // to Polars Series (no clone).
                data.into_series(name.as_ref())
                    .into()
            }
            KeepMask::Sparse(idx) => data
                .take_indices(idx)
                .into_series(name.as_ref())
                .into(),
        };
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

    match keep {
        KeepMask::All(n) => data
            .gather_by(*n, |i| {
                compute_var_idx(i)
            }),
        KeepMask::Sparse(idx) => data
            .gather_by(idx.len(), |i| {
                compute_var_idx(idx[i])
            }),
    }
    .into_series(name.as_ref())
    .into()
}
