pub(crate) use crate::reader::{
    ColumnData, compute_strides,
};
pub(crate) use polars::prelude::*;
pub(crate) use std::collections::BTreeSet;

use crate::IStr;

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
        // Compute grid shape for this variable
        let var_grid_shape: Vec<u64> = var_shape
            .iter()
            .zip(var_chunk_shape.iter())
            .map(|(s, c)| (s + c - 1) / c)
            .collect();

        // Clamp indices to valid range
        let clamped: Vec<u64> = primary_idx
            .iter()
            .zip(var_grid_shape.iter())
            .map(|(i, g)| {
                (*i).min(g.saturating_sub(1))
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

pub(crate) fn transform_columns<T: From<i64>>(
    keep: &[usize],
    strides_at_dim: usize,
    chunk_shape_at_dim: usize,
    origin_at_dim: usize,
    coord_data: Option<&ColumnData>,
) -> Vec<T> {
    let mut out: Vec<T> =
        Vec::with_capacity(keep.len());
    for &row in keep {
        let local = (row / strides_at_dim)
            % chunk_shape_at_dim;
        let raw_value = T::from(
            coord_data
                .and_then(|c| c.get_i64(local))
                .unwrap_or(
                    (origin_at_dim + local)
                        as i64,
                ),
        );
        out.push(raw_value);
    }
    out
}

pub(crate) fn build_keep_indices(
    keep: &[usize],
    strides_at_dim: usize,
    chunk_shape_at_dim: usize,
) -> Vec<usize> {
    let mut out: Vec<usize> =
        Vec::with_capacity(keep.len());
    for &row in keep {
        let local = (row / strides_at_dim)
            % chunk_shape_at_dim;
        out.push(local);
    }
    out
}

/// Build a coordinate column for the DataFrame.
pub(crate) fn build_coord_column(
    dim_name: &str,
    dim_idx: usize,
    keep: &[usize],
    strides: &[u64],
    chunk_shape: &[u64],
    origin: &[u64],
    coord_data: Option<&ColumnData>,
    time_encoding: Option<
        &crate::meta::TimeEncoding,
    >,
) -> Column {
    let keep_indices = build_keep_indices(
        keep,
        strides[dim_idx] as usize,
        chunk_shape[dim_idx] as usize,
    );
    if let Some(coord) = coord_data {
        if let Some(time_encoding) = time_encoding
        {
            let series_uncast = coord
                .take_indices(&keep_indices)
                .map_i64(|v| {
                    time_encoding.decode(v)
                })
                .into_series(dim_name.into());
            return series_uncast
                .cast(
                    &time_encoding
                        .to_polars_dtype(),
                )
                .unwrap_or(series_uncast)
                .into();
        }

        return coord
            .take_indices(&keep_indices)
            .into_series(dim_name.into())
            .into();
    } else {
        let series = Series::new(
            dim_name.into(),
            keep_indices
                .iter()
                .map(|&i| {
                    (i as i64)
                        + (origin[dim_idx] as i64)
                })
                .collect::<Vec<i64>>(),
        );
        return series.into();
    }
}

/// Compute the in-bounds mask for edge chunk handling.
pub(crate) fn compute_in_bounds_mask(
    chunk_len: usize,
    chunk_shape: &[u64],
    origin: &[u64],
    array_shape: &[u64],
    strides: &[u64],
) -> Vec<usize> {
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
    keep
}

/// Build a variable column for the DataFrame.
pub(crate) fn build_var_column(
    name: &IStr,
    data: &ColumnData,
    var_dims: &[IStr],
    var_chunk_shape: &[u64],
    var_offsets: &[u64],
    primary_dims: &[IStr],
    primary_chunk_shape: &[u64],
    primary_strides: &[u64],
    keep: &[usize],
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
        // Fast path: direct index mapping
        data.take_indices(keep)
            .into_series(name.as_ref())
            .into()
    } else {
        // Slow path: map indices through dimension differences
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

        let indices: Vec<usize> = keep
            .iter()
            .map(|&row| {
                let mut var_idx: u64 = 0;
                for (primary_d, maybe_var_d) in
                    dim_mapping.iter().enumerate()
                {
                    if let Some(var_d) = *maybe_var_d {
                        let local = (row as u64
                            / primary_strides[primary_d])
                            % primary_chunk_shape[primary_d];
                        let var_local = if same_dims
                            && var_chunk_shape.len() > var_d
                        {
                            local.min(
                                var_chunk_shape[var_d]
                                    .saturating_sub(1),
                            )
                        } else {
                            local
                        };
                        let local_with_offset =
                            var_local + var_offsets[var_d];
                        var_idx += local_with_offset
                            * var_strides[var_d];
                    }
                }
                (var_idx as usize)
                    .min(var_data_len.saturating_sub(1))
            })
            .collect();

        data.take_indices(&indices)
            .into_series(name.as_ref())
            .into()
    }
}
