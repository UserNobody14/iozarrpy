//! Chunk-to-DataFrame conversion using generic backend traits.
//!
//! This module provides functions to convert zarr chunks into Polars DataFrames,
//! using a generic backend interface that can work with any implementation
//! (zarr, icechunk, gribberish, etc.).

pub(crate) use std::collections::BTreeSet;

pub(crate) use futures::stream::{
    FuturesUnordered, StreamExt,
};
pub(crate) use polars::prelude::*;
use snafu::ResultExt;

use crate::errors::{
    BackendError, BackendResult, PolarsSnafu,
};
pub(crate) use crate::meta::ZarrMeta;
pub(crate) use crate::reader::{
    ColumnData, checked_chunk_len,
    compute_strides,
};

use crate::IStr;
use crate::chunk_plan::{
    ChunkGridSignature, ChunkSubset,
};
use crate::scan::shared::{
    build_coord_column, build_var_column,
    compute_actual_chunk_shape,
    compute_in_bounds_mask,
    compute_var_chunk_indices,
    should_include_column,
};
use crate::shared::ChunkedDataBackendAsync;

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

    let first_chunk = start / coord_chunk_shape;
    let last_pos = start + len - 1;
    let last_chunk = last_pos / coord_chunk_shape;

    // Fetch all needed chunks concurrently (typically 1-3 chunks)
    let chunk_indices: Vec<[u64; 1]> =
        (first_chunk..=last_chunk)
            .map(|i| [i])
            .collect();
    let chunk_futs: Vec<_> = chunk_indices
        .iter()
        .map(|idx| {
            backend
                .read_chunk_async(coord_path, idx)
        })
        .collect();

    let chunks =
        futures::future::try_join_all(chunk_futs)
            .await?;

    // Assemble the result in order from the fetched chunks
    let range_end = start + len;
    let mut result_data: Option<ColumnData> =
        None;

    for (i, chunk_data) in
        chunks.into_iter().enumerate()
    {
        let chunk_idx = first_chunk + i as u64;
        let chunk_start_pos =
            chunk_idx * coord_chunk_shape;
        let chunk_end_pos =
            chunk_start_pos + coord_chunk_shape;

        let local_start =
            if start > chunk_start_pos {
                (start - chunk_start_pos) as usize
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
async fn read_coord_chunks<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    meta: &ZarrMeta,
    dims: &[IStr],
    origin: &[u64],
    chunk_shape: &[u64],
    with_columns: Option<&BTreeSet<IStr>>,
) -> BackendResult<
    std::collections::BTreeMap<IStr, ColumnData>,
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
            meta.array_by_path(dim_name.clone())
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
        let coord = res?;
        if coord.len() != expected_len {
            return Err(BackendError::CoordLengthMismatch {
                name: name.clone(),
                expected_len: expected_len as u64,
                coord_len: coord.len() as u64,
            });
        }
        coord_slices.insert(name, coord);
    }

    Ok(coord_slices)
}

/// Read variable chunks for all requested variables.
async fn read_var_chunks<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    meta: &ZarrMeta,
    idx: &[u64],
    chunk_shape: &[u64],
    dims: &[IStr],
    vars: &[IStr],
    with_columns: Option<&BTreeSet<IStr>>,
) -> BackendResult<
    Vec<(
        IStr,
        Arc<ColumnData>,
        Vec<IStr>,
        Vec<u64>,
        Vec<u64>,
    )>,
> {
    let mut var_reads = FuturesUnordered::new();

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

        let var_meta = meta
            .array_by_path(name.clone())
            .ok_or(
                BackendError::UnknownDataVar {
                    name: name.clone(),
                    available_vars: meta
                        .all_data_var_paths(),
                },
            )?;

        let name = name.clone();
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
                .await?;

            Ok::<_, BackendError>((
                name,
                data.clone(),
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

// =============================================================================
// Convenience Functions with ChunkGridSignature
// =============================================================================

/// Convert a chunk to DataFrame using signature and grid info.
///
/// This is a convenience wrapper that computes chunk geometry from
/// the signature and grid, then calls the main function.
/// An optional `chunk_subset` constrains which elements within the
/// chunk are included, avoiding unnecessary column-building work.
pub async fn chunk_to_df_from_grid_with_backend<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    idx: Vec<u64>,
    sig: &ChunkGridSignature,
    array_shape: &[u64],
    vars: &[IStr],
    with_columns: Option<&BTreeSet<IStr>>,
    chunk_subset: Option<&ChunkSubset>,
    meta: &ZarrMeta,
) -> BackendResult<DataFrame> {
    let chunk_shape = sig.chunk_shape();
    let dims = sig.dims();

    // Compute origin from chunk indices
    let origin: Vec<u64> = idx
        .iter()
        .zip(chunk_shape.iter())
        .map(|(i, s)| i * s)
        .collect();

    let chunk_len =
        checked_chunk_len(chunk_shape)?;
    let strides = compute_strides(chunk_shape);

    let keep = compute_in_bounds_mask(
        chunk_len,
        chunk_shape,
        &origin,
        array_shape,
        &strides,
        chunk_subset,
    );

    // Perform both reads concurrently
    let (coord_slices, var_chunks) = futures::try_join!(
        read_coord_chunks(
            backend,
            meta,
            dims,
            &origin,
            chunk_shape,
            with_columns,
        ),
        read_var_chunks(
            backend,
            meta,
            &idx,
            chunk_shape,
            dims,
            vars,
            with_columns,
        )
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

        let encoding = meta
            .array_by_path(dim_name.clone())
            .and_then(|m| m.encoding.as_ref());

        let col = build_coord_column(
            dim_name.as_ref(),
            d,
            &keep,
            &strides,
            chunk_shape,
            &origin,
            coord_slices.get(dim_name),
            encoding,
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
        let encoding = meta
            .array_by_path(name.clone())
            .and_then(|m| m.encoding.as_ref());

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
            encoding,
        );
        cols.push(col);
    }

    Ok(DataFrame::new(height, cols)
        .context(PolarsSnafu)?)
}
