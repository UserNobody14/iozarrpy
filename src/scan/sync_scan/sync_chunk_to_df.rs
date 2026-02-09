//! Synchronous chunk-to-DataFrame conversion using generic backend traits.
//!
//! This module provides sync functions to convert zarr chunks into Polars DataFrames,
//! using a generic backend interface that can work with any implementation.

use crate::IStr;
use crate::chunk_plan::ChunkGridSignature;
use crate::errors::BackendError;
use crate::meta::{ZarrDatasetMeta, ZarrMeta};
use crate::reader::{
    ColumnData, checked_chunk_len,
    compute_strides,
};
use crate::shared::{
    ChunkDataSourceSync, ChunkedDataBackendSync,
};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use std::collections::BTreeSet;

fn to_py_err<E: std::fmt::Display>(
    e: E,
) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(
        e.to_string(),
    )
}

use crate::scan::shared::{
    build_coord_column, build_var_column,
    compute_actual_chunk_shape,
    compute_in_bounds_mask,
    compute_var_chunk_indices,
    should_include_column,
};

// =============================================================================
// Chunked Coordinate Reading (replaces retrieve_1d_subset)
// =============================================================================

/// Read coordinate values for a range using chunked access.
fn read_coord_range_chunked<
    B: ChunkedDataBackendSync,
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
            .read_chunk_sync(
                coord_path,
                &[chunk_idx],
            )?;

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

/// Read coordinate chunks for all dimensions.
fn read_coord_chunks<B: ChunkDataSourceSync>(
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
    let mut coord_slices: std::collections::BTreeMap<
        IStr,
        ColumnData,
    > = Default::default();

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

        // Use the full path from metadata for array access
        let data = read_coord_range_chunked(
            backend,
            dim_name,
            // &coord_meta.path,
            coord_chunk_shape,
            dim_start,
            dim_len,
        )
        .map_err(to_py_err)?;

        if data.len() != dim_len as usize {
            return Err(PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(format!(
                "coord '{}' length mismatch: expected {}, got {}",
                dim_name,
                dim_len,
                data.len()
            )));
        }
        coord_slices
            .insert(dim_name.clone(), data);
    }

    Ok(coord_slices)
}

/// Read variable chunks for all requested variables.
fn read_var_chunks<B: ChunkDataSourceSync>(
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
    let mut var_chunks = Vec::new();

    for name in vars.iter() {
        // Dimension columns are always produced via `build_coord_column`.
        // If a dataset also has 1D coord arrays named the same as dims (e.g. "x", "y", "a"),
        // they can appear in `vars` (e.g. from `pl.col(["x","y",...])`). Reading them as
        // "variables" would create duplicate DataFrame columns (two "x" columns, etc.).
        if dims.iter().any(|d| d == name) {
            continue;
        }
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

        let var_dims: Vec<IStr> = var_meta
            .dims
            .iter()
            .cloned()
            .collect();

        // Compute which chunk to read for this variable
        let (var_chunk_indices, var_offsets) =
            compute_var_chunk_indices(
                idx,
                chunk_shape,
                dims,
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

        // Read the chunk using the full path from metadata
        let data = backend
            .read_chunk_sync(
                &name,
                &var_chunk_indices,
            )
            .map_err(to_py_err)?;

        var_chunks.push((
            name.clone(),
            data,
            var_dims,
            var_chunk_shape,
            var_offsets,
        ));
    }

    Ok(var_chunks)
}

// =============================================================================
// Convenience Functions with ChunkGridSignature
// =============================================================================

/// Convert a chunk to DataFrame using signature and grid info (sync).
pub fn chunk_to_df_from_grid_with_backend<
    B: ChunkDataSourceSync,
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

    let meta =
        backend.metadata().map_err(to_py_err)?;
    let planning_meta = meta.planning_meta();

    let chunk_len =
        checked_chunk_len(chunk_shape)?;
    let strides = compute_strides(chunk_shape);

    // In-bounds mask (handles edge chunks).
    // Returns KeepMask::All for interior chunks
    // (O(ndim) check), avoiding O(chunk_len) work.
    let keep = compute_in_bounds_mask(
        chunk_len,
        chunk_shape,
        &origin,
        array_shape,
        &strides,
    );

    // Read coordinate chunks
    let coord_slices = read_coord_chunks(
        backend,
        &meta,
        dims,
        &origin,
        chunk_shape,
        with_columns,
    )?;

    // Read variable chunks
    let var_chunks = read_var_chunks(
        backend,
        &planning_meta,
        &idx,
        chunk_shape,
        dims,
        vars,
        with_columns,
    )?;

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
            &origin,
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
            data,
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
