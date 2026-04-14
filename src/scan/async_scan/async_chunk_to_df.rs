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
use crate::scan::column_policy::{
    DimMaterialization, ReadSpec,
    build_chunk_physical_plan,
};
use crate::scan::shared::{
    build_coord_column, build_var_column,
    compute_in_bounds_mask,
};
use crate::shared::ChunkedDataBackendAsync;

use std::collections::BTreeMap;
use std::sync::Arc;

// =============================================================================
// Physical reads (slice + chunk)
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

async fn execute_read_async<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    path: IStr,
    spec: ReadSpec,
) -> BackendResult<(IStr, Arc<ColumnData>)> {
    let data: Arc<ColumnData> = match spec {
        ReadSpec::Slice1d {
            coord_chunk_shape,
            start,
            len,
        } => Arc::new(
            read_coord_range_chunked(
                backend,
                &path,
                coord_chunk_shape,
                start,
                len,
            )
            .await?,
        ),
        ReadSpec::Chunk { indices } => backend
            .read_chunk_async(&path, &indices)
            .await?,
    };
    Ok((path, data))
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

    let plan = build_chunk_physical_plan(
        meta,
        dims,
        &idx,
        chunk_shape,
        &origin,
        vars,
        with_columns,
    )?;

    let mut read_futs = FuturesUnordered::new();
    for (path, spec) in &plan.reads {
        read_futs.push(execute_read_async(
            backend,
            *path,
            spec.clone(),
        ));
    }

    let mut loaded: BTreeMap<
        IStr,
        Arc<ColumnData>,
    > = BTreeMap::new();
    while let Some(res) = read_futs.next().await {
        let (path, data) = res?;
        loaded.insert(path, data);
    }

    // Build DataFrame columns
    let mut cols: Vec<Column> = Vec::new();
    let height = keep.len();

    for dim_step in &plan.dims {
        let coord_data: Option<&ColumnData> =
            match &dim_step.mat {
                DimMaterialization::Synthetic => None,
                DimMaterialization::FromArray {
                    path,
                } => {
                    let col = loaded
                        .get(path)
                        .ok_or_else(|| {
                            BackendError::Other {
                                msg: format!(
                                    "internal: missing read for coord path {}",
                                    path
                                ),
                            }
                        })?
                        .as_ref();
                    let expected_len =
                        chunk_shape[dim_step.dim_idx]
                            as usize;
                    if col.len() != expected_len {
                        return Err(
                            BackendError::CoordLengthMismatch {
                                name: dim_step.dim_name,
                                expected_len: chunk_shape
                                    [dim_step.dim_idx],
                                coord_len: col.len()
                                    as u64,
                            },
                        );
                    }
                    Some(col)
                }
            };

        let encoding = meta
            .array_by_path(dim_step.dim_name)
            .and_then(|m| m.encoding.as_ref());

        cols.push(build_coord_column(
            dim_step.dim_name.as_ref(),
            dim_step.dim_idx,
            &keep,
            &strides,
            chunk_shape,
            &origin,
            coord_data,
            encoding,
        ));
    }

    for vs in &plan.vars {
        let data = loaded
            .get(&vs.path)
            .ok_or_else(|| BackendError::Other {
                msg: format!(
                    "internal: missing read for variable path {}",
                    vs.path
                ),
            })?
            .clone();

        let encoding = meta
            .array_by_path(vs.name)
            .and_then(|m| m.encoding.as_ref());

        cols.push(build_var_column(
            &vs.name,
            data,
            &vs.var_dims,
            &vs.var_chunk_shape,
            &vs.offsets,
            dims,
            chunk_shape,
            &strides,
            &keep,
            encoding,
        ));
    }

    DataFrame::new(height, cols).context(
        PolarsSnafu {
            message: "Error creating DataFrame"
                .to_string(),
        },
    )
}
