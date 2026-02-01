//! Chunk-to-DataFrame conversion using generic backend traits.
//!
//! This module provides functions to convert zarr chunks into Polars DataFrames,
//! using a generic backend interface that can work with any implementation
//! (zarr, icechunk, gribberish, etc.).

use super::prelude::*;
use crate::backend::{
    BackendError, ChunkDataSourceAsync,
    ChunkedDataBackendAsync,
};
use crate::chunk_plan::ChunkGridSignature;

// =============================================================================
// Chunked Coordinate Reading (replaces retrieve_1d_subset)
// =============================================================================

/// Read coordinate values for a range using chunked access.
///
/// This is more efficient than `retrieve_1d_subset` because it uses the same
/// chunk-based access pattern as variable data, enabling caching.
async fn read_coord_range_chunked<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    coord_path: &IStr,
    coord_chunk_shape: u64,
    start: u64,
    len: u64,
) -> Result<ColumnData, BackendError> {
    if len == 0 {
        return Ok(ColumnData::I64(vec![]));
    }

    // For 1D coordinate arrays, compute which chunks we need
    let first_chunk = start / coord_chunk_shape;
    let last_pos = start + len - 1;
    let last_chunk = last_pos / coord_chunk_shape;

    // Read chunks and extract the needed range
    let mut result_data: Option<ColumnData> =
        None;

    for chunk_idx in first_chunk..=last_chunk {
        let chunk_data = backend
            .read_chunk_async(
                coord_path,
                &[chunk_idx],
            )
            .await?;

        // Calculate what portion of this chunk we need
        let chunk_start_pos =
            chunk_idx * coord_chunk_shape;
        let chunk_end_pos =
            chunk_start_pos + coord_chunk_shape;

        // Our range within global coords
        let range_start = start;
        let range_end = start + len;

        // Intersection with this chunk
        let local_start =
            if range_start > chunk_start_pos {
                (range_start - chunk_start_pos)
                    as usize
            } else {
                0
            };
        let local_end = if range_end
            < chunk_end_pos
        {
            (range_end - chunk_start_pos) as usize
        } else {
            chunk_data.len()
        };
        let local_len =
            local_end.saturating_sub(local_start);

        let slice = chunk_data
            .slice(local_start, local_len);

        result_data = Some(match result_data {
            None => slice,
            Some(existing) => {
                existing.concat(&slice)
            }
        });
    }

    Ok(result_data.unwrap_or_else(|| {
        ColumnData::I64(vec![])
    }))
}

// =============================================================================
// Main Chunk-to-DataFrame Function (Generic Backend)
// =============================================================================

/// Convert a zarr chunk to a DataFrame using a generic backend.
///
/// This is the main entry point for chunk processing. It reads chunk data
/// from the backend and constructs a tidy DataFrame with coordinate columns
/// and variable columns.
pub async fn chunk_to_df_with_backend<
    B: ChunkDataSourceAsync,
>(
    backend: &B,
    idx: Vec<u64>,
    chunk_shape: &[u64],
    origin: &[u64],
    array_shape: &[u64],
    dims: &[IStr],
    vars: &[IStr],
    with_columns: Option<&BTreeSet<IStr>>,
) -> Result<DataFrame, PyErr> {
    let meta = backend
        .metadata()
        .await
        .map_err(to_py_err)?;
    let planning_meta = meta.planning_meta();

    let chunk_len =
        checked_chunk_len(chunk_shape)?;
    let strides = compute_strides(chunk_shape);

    // In-bounds mask (handles edge chunks)
    let keep: Vec<usize> = compute_in_bounds_mask(
        chunk_len,
        chunk_shape,
        origin,
        array_shape,
        &strides,
    );

    // Read coordinate chunks in parallel
    let coord_slices = read_coord_chunks(
        backend,
        &meta,
        dims,
        origin,
        chunk_shape,
        with_columns,
    )
    .await?;

    // Read variable chunks in parallel
    let var_chunks = read_var_chunks(
        backend,
        &planning_meta,
        &idx,
        chunk_shape,
        dims,
        vars,
        with_columns,
    )
    .await?;

    // Build DataFrame columns
    let mut cols: Vec<Column> = Vec::new();
    let height = keep.len();

    // Coordinate columns
    for (d, dim_name) in dims.iter().enumerate() {
        if !should_include_column(
            dim_name,
            with_columns,
        ) {
            continue;
        }

        let time_encoding = planning_meta
            .arrays
            .get(dim_name)
            .and_then(|m| {
                m.time_encoding.as_ref()
            });

        let col = build_coord_column(
            dim_name.as_ref(),
            d,
            &keep,
            &strides,
            chunk_shape,
            origin,
            coord_slices.get(dim_name),
            time_encoding,
        );
        cols.push(col);
    }

    // Variable columns
    for (
        name,
        data,
        var_dims,
        var_chunk_shape,
        var_offsets,
    ) in var_chunks
    {
        let col = build_var_column(
            &name,
            &data,
            &var_dims,
            &var_chunk_shape,
            &var_offsets,
            dims,
            chunk_shape,
            &strides,
            &keep,
        );
        cols.push(col);
    }

    Ok(DataFrame::new(height, cols)
        .map_err(PyPolarsErr::from)?)
}

/// Compute the in-bounds mask for edge chunk handling.
fn compute_in_bounds_mask(
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

/// Read coordinate chunks for all dimensions.
async fn read_coord_chunks<
    B: ChunkDataSourceAsync,
>(
    backend: &B,
    meta: &ZarrMeta,
    dims: &[IStr],
    origin: &[u64],
    chunk_shape: &[u64],
    with_columns: Option<&BTreeSet<IStr>>,
) -> Result<
    std::collections::BTreeMap<IStr, ColumnData>,
    PyErr,
> {
    let mut coord_reads = FuturesUnordered::new();

    for (d, dim_name) in dims.iter().enumerate() {
        if !should_include_column(
            dim_name,
            with_columns,
        ) {
            continue;
        }

        // Check if this dimension has a coordinate array
        let Some(coord_meta) =
            meta.path_to_array.get(dim_name)
        else {
            continue;
        };

        // Only process 1D coordinate arrays
        if coord_meta.shape.len() != 1 {
            continue;
        }

        let dim_start = origin[d];
        let dim_len = chunk_shape[d];
        let coord_chunk_shape =
            coord_meta.chunk_shape[0];
        let dim_name = dim_name.clone();
        // Use full path from metadata for array access
        let coord_path = coord_meta.path.clone();

        coord_reads.push(async move {
            let data = read_coord_range_chunked(
                backend,
                &coord_path,
                coord_chunk_shape,
                dim_start,
                dim_len,
            )
            .await;
            (dim_name, dim_len as usize, data)
        });
    }

    let mut coord_slices: std::collections::BTreeMap<
        IStr,
        ColumnData,
    > = Default::default();

    while let Some((name, expected_len, res)) =
        coord_reads.next().await
    {
        let coord = res.map_err(to_py_err)?;
        if coord.len() != expected_len {
            return Err(PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(format!(
                "coord '{}' length mismatch: expected {}, got {}",
                name,
                expected_len,
                coord.len()
            )));
        }
        coord_slices.insert(name, coord);
    }

    Ok(coord_slices)
}

/// Read variable chunks for all requested variables.
async fn read_var_chunks<
    B: ChunkDataSourceAsync,
>(
    backend: &B,
    meta: &ZarrDatasetMeta,
    idx: &[u64],
    chunk_shape: &[u64],
    dims: &[IStr],
    vars: &[IStr],
    with_columns: Option<&BTreeSet<IStr>>,
) -> Result<
    Vec<(
        IStr,
        ColumnData,
        Vec<IStr>,
        Vec<u64>,
        Vec<u64>,
    )>,
    PyErr,
> {
    let mut var_reads = FuturesUnordered::new();

    for name in vars.iter() {
        if !should_include_column(
            name,
            with_columns,
        ) {
            continue;
        }

        let var_meta =
            meta.arrays.get(name).ok_or_else(|| {
                PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >(format!(
                    "unknown variable: {}",
                    name
                ))
            })?;

        let name = name.clone();
        let var_meta = var_meta.clone();
        let dims = dims.to_vec();
        let idx = idx.to_vec();
        let chunk_shape = chunk_shape.to_vec();

        var_reads.push(async move {
            let var_dims: Vec<IStr> = var_meta
                .dims
                .iter()
                .cloned()
                .collect();

            // Compute which chunk to read for this variable
            let (var_chunk_indices, var_offsets) =
                compute_var_chunk_indices(
                    &idx,
                    &chunk_shape,
                    &dims,
                    &var_dims,
                    &var_meta.chunk_shape,
                    &var_meta.shape,
                );

            // Get actual chunk shape (may differ for edge chunks)
            let var_chunk_shape =
                compute_actual_chunk_shape(
                    &var_chunk_indices,
                    &var_meta.chunk_shape,
                    &var_meta.shape,
                );

            // Read the chunk using full path from metadata
            let data = backend
                .read_chunk_async(
                    &var_meta.path,
                    &var_chunk_indices,
                )
                .await
                .map_err(to_py_err)?;

            Ok::<_, PyErr>((
                name,
                data,
                var_dims,
                var_chunk_shape,
                var_offsets,
            ))
        });
    }

    let mut var_chunks = Vec::new();
    while let Some(r) = var_reads.next().await {
        var_chunks.push(r?);
    }

    Ok(var_chunks)
}

/// Compute which chunk indices to read for a variable based on
/// the primary chunk being processed.
fn compute_var_chunk_indices(
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
fn compute_actual_chunk_shape(
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

fn should_include_column(
    name: &IStr,
    with_columns: Option<&BTreeSet<IStr>>,
) -> bool {
    with_columns
        .map(|s| s.contains(name))
        .unwrap_or(true)
}

/// Build a coordinate column for the DataFrame.
fn build_coord_column(
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
    if let Some(te) = time_encoding {
        let mut out_i64: Vec<i64> =
            Vec::with_capacity(keep.len());
        for &row in keep {
            let local = (row as u64
                / strides[dim_idx])
                % chunk_shape[dim_idx];
            let raw_value = coord_data
                .and_then(|c| {
                    c.get_i64(local as usize)
                })
                .unwrap_or(
                    (origin[dim_idx] + local)
                        as i64,
                );
            let ns = if te.is_duration {
                raw_value
                    .saturating_mul(te.unit_ns)
            } else {
                raw_value
                    .saturating_mul(te.unit_ns)
                    .saturating_add(te.epoch_ns)
            };
            out_i64.push(ns);
        }
        let series = if te.is_duration {
            Series::new(dim_name.into(), &out_i64)
                .cast(&DataType::Duration(
                    TimeUnit::Nanoseconds,
                ))
                .unwrap_or_else(|_| {
                    Series::new(
                        dim_name.into(),
                        out_i64,
                    )
                })
        } else {
            Series::new(dim_name.into(), &out_i64)
                .cast(&DataType::Datetime(
                    TimeUnit::Nanoseconds,
                    None,
                ))
                .unwrap_or_else(|_| {
                    Series::new(
                        dim_name.into(),
                        out_i64,
                    )
                })
        };
        series.into()
    } else if let Some(coord) = coord_data {
        if coord.is_float() {
            let mut out_f64: Vec<f64> =
                Vec::with_capacity(keep.len());
            for &row in keep {
                let local = (row as u64
                    / strides[dim_idx])
                    % chunk_shape[dim_idx];
                out_f64.push(
                    coord
                        .get_f64(local as usize)
                        .unwrap_or(0.0),
                );
            }
            Series::new(dim_name.into(), out_f64)
                .into()
        } else {
            let mut out_i64: Vec<i64> =
                Vec::with_capacity(keep.len());
            for &row in keep {
                let local = (row as u64
                    / strides[dim_idx])
                    % chunk_shape[dim_idx];
                out_i64.push(
                    coord
                        .get_i64(local as usize)
                        .unwrap_or(
                            (origin[dim_idx]
                                + local)
                                as i64,
                        ),
                );
            }
            Series::new(dim_name.into(), out_i64)
                .into()
        }
    } else {
        let mut out_i64: Vec<i64> =
            Vec::with_capacity(keep.len());
        for &row in keep {
            let local = (row as u64
                / strides[dim_idx])
                % chunk_shape[dim_idx];
            out_i64.push(
                (origin[dim_idx] + local) as i64,
            );
        }
        Series::new(dim_name.into(), out_i64)
            .into()
    }
}

/// Build a variable column for the DataFrame.
fn build_var_column(
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

// =============================================================================
// Convenience Functions with ChunkGridSignature
// =============================================================================

/// Convert a chunk to DataFrame using signature and grid info.
///
/// This is a convenience wrapper that computes chunk geometry from
/// the signature and grid, then calls the main function.
pub async fn chunk_to_df_from_grid_with_backend<
    B: ChunkDataSourceAsync,
>(
    backend: &B,
    idx: Vec<u64>,
    sig: &ChunkGridSignature,
    array_shape: &[u64],
    vars: &[IStr],
    with_columns: Option<&BTreeSet<IStr>>,
) -> Result<DataFrame, PyErr> {
    let chunk_shape = sig.chunk_shape();
    let dims = sig.dims();

    // Compute origin from chunk indices
    let origin: Vec<u64> = idx
        .iter()
        .zip(chunk_shape.iter())
        .map(|(i, s)| i * s)
        .collect();

    chunk_to_df_with_backend(
        backend,
        idx,
        &chunk_shape,
        &origin,
        array_shape,
        dims,
        vars,
        with_columns,
    )
    .await
}
