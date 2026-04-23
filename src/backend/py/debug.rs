use crate::chunk_plan::ChunkGridSignature;
use crate::meta::ZarrMeta;
use crate::shared::{
    ChunkedDataBackendAsync,
    ChunkedExpressionCompilerSync,
};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::shared::ChunkedExpressionCompilerAsync;
use crate::shared::{
    ChunkedDataBackendSync,
    HasMetadataBackendAsync,
    HasMetadataBackendSync,
};

#[derive(Clone)]
pub(crate) struct ShardInfo {
    indices: Vec<u64>,
    origin: Vec<u64>,
    shape: Vec<u64>,
}

#[derive(Clone)]
pub(crate) struct ChunkInfo {
    indices: Vec<u64>,
    origin: Vec<u64>,
    shape: Vec<u64>,
    shards: Vec<ShardInfo>,
}

#[derive(Clone)]
pub(crate) struct GridInfo {
    dims: Vec<String>,
    variables: Vec<String>,
    chunks: Vec<ChunkInfo>,
}

fn shard_info_from_outer_grid(
    outer_idx: &[u64],
    outer_shape: &[u64],
    array_shape: Option<&Vec<u64>>,
) -> ShardInfo {
    let indices = outer_idx.to_vec();
    let origin: Vec<u64> = outer_idx
        .iter()
        .zip(outer_shape.iter())
        .map(|(&i, &sz)| i.saturating_mul(sz))
        .collect();
    let shape: Vec<u64> = match array_shape {
        Some(a_shape)
            if a_shape.len()
                == outer_shape.len() =>
        {
            origin
                .iter()
                .zip(a_shape.iter())
                .zip(outer_shape.iter())
                .map(|((&o, &a), &s)| {
                    if o >= a {
                        0
                    } else {
                        (a - o).min(s)
                    }
                })
                .collect()
        }
        _ => outer_shape.to_vec(),
    };
    ShardInfo {
        indices,
        origin,
        shape,
    }
}

fn build_group_chunks(
    inner_chunk_indices: &[Vec<u64>],
    inner_chunk_shape: &[u64],
    sig: &ChunkGridSignature,
    array_shape: &Option<Vec<u64>>,
) -> Vec<ChunkInfo> {
    let mut chunks: Vec<ChunkInfo> = Vec::new();
    let is_sharded = sig.is_sharded();
    for inner_idx in inner_chunk_indices {
        if is_sharded {
            let outer_shape =
                sig.outer_chunk_shape();
            let outer_idx: Vec<u64> = inner_idx
                .iter()
                .zip(inner_chunk_shape.iter())
                .zip(outer_shape.iter())
                .map(
                    |(
                        (&i, &inner_sz),
                        &outer_sz,
                    )| {
                        if outer_sz == 0 {
                            0
                        } else {
                            i.saturating_mul(
                                inner_sz,
                            ) / outer_sz
                        }
                    },
                )
                .collect();

            let shard =
                shard_info_from_outer_grid(
                    &outer_idx,
                    outer_shape,
                    array_shape.as_ref(),
                );

            let origin: Vec<u64> = inner_idx
                .iter()
                .zip(inner_chunk_shape.iter())
                .map(|(&i, &s)| i * s)
                .collect();

            chunks.push(ChunkInfo {
                indices: inner_idx.clone(),
                origin,
                shape: inner_chunk_shape.to_vec(),
                shards: vec![shard],
            });
        } else {
            let origin: Vec<u64> = inner_idx
                .iter()
                .zip(inner_chunk_shape.iter())
                .map(|(&i, &s)| i * s)
                .collect();
            chunks.push(ChunkInfo {
                indices: inner_idx.clone(),
                origin,
                shape: inner_chunk_shape.to_vec(),
                shards: vec![],
            });
        }
    }
    chunks
}

// Helper for async grid extraction logic
pub(crate) async fn extract_grids<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync,
>(
    backend: Arc<B>,
    expr: polars::prelude::Expr,
) -> PyResult<(Vec<GridInfo>, u64)> {
    // Compile expression to grouped chunk plan using backend-based resolver
    let (grouped_plan, stats) = backend
        .clone()
        .compile_expression_async(&expr)
        .await?;

    let mut grids: Vec<GridInfo> = Vec::new();

    // Use iter_consolidated_chunks() to mirror the actual I/O path exactly.
    // This function already deduplicates chunk indices across overlapping subsets
    // using the same BTreeSet logic as the real chunk readers.
    for result in
        grouped_plan.iter_consolidated_chunks()
    {
        let group = result.map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e.to_string())
        })?;
        let sig = group.sig;
        let vars = group.vars;
        // chunk_indices are already deduplicated by iter_consolidated_chunks()
        let inner_chunk_indices =
            group.chunk_indices;

        let dims: Vec<String> = sig
            .dims()
            .iter()
            .map(|d| d.to_string())
            .collect();
        let variables: Vec<String> = vars
            .iter()
            .map(|v| v.to_string())
            .collect();

        let inner_chunk_shape: Vec<u64> =
            sig.retrieval_shape().to_vec();

        let chunks = build_group_chunks(
            &inner_chunk_indices,
            &inner_chunk_shape,
            &sig,
            &Some(group.array_shape),
        );

        grids.push(GridInfo {
            dims,
            variables,
            chunks,
        });
    }

    Ok((grids, stats.coord_reads))
}

// Helper for sync grid extraction logic
pub(crate) fn extract_grids_sync<
    B: HasMetadataBackendSync<ZarrMeta>
        + ChunkedDataBackendSync,
>(
    backend: Arc<B>,
    expr: polars::prelude::Expr,
) -> PyResult<(Vec<GridInfo>, u64)> {
    let (grouped_plan, stats) =
        backend
            .clone()
            .compile_expression_sync(&expr)?;

    let mut grids: Vec<GridInfo> = Vec::new();

    // Use iter_consolidated_chunks() to mirror the actual I/O path exactly.
    // This function already deduplicates chunk indices across overlapping subsets
    // using the same BTreeSet logic as the real chunk readers.
    for result in
        grouped_plan.iter_consolidated_chunks()
    {
        let group = result.map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e.to_string())
        })?;
        let sig = group.sig;
        let vars = group.vars;
        // chunk_indices are already deduplicated by iter_consolidated_chunks()
        let inner_chunk_indices =
            group.chunk_indices;

        let dims: Vec<String> = sig
            .dims()
            .iter()
            .map(|d| d.to_string())
            .collect();
        let variables: Vec<String> = vars
            .iter()
            .map(|v| v.to_string())
            .collect();

        let inner_chunk_shape: Vec<u64> =
            sig.retrieval_shape().to_vec();

        let chunks = build_group_chunks(
            &inner_chunk_indices,
            &inner_chunk_shape,
            &sig,
            &Some(group.array_shape),
        );

        grids.push(GridInfo {
            dims,
            variables,
            chunks,
        });
    }

    Ok((grids, stats.coord_reads))
}

pub(crate) fn grids_to_python<'py>(
    py: Python<'py>,
    grids: Vec<GridInfo>,
    coord_reads: u64,
) -> PyResult<Py<PyAny>> {
    let py_grids = pyo3::types::PyList::empty(py);

    for grid in grids {
        let grid_dict =
            pyo3::types::PyDict::new(py);
        grid_dict.set_item("dims", &grid.dims)?;
        grid_dict.set_item(
            "variables",
            &grid.variables,
        )?;

        let chunks_list =
            pyo3::types::PyList::empty(py);
        for chunk in grid.chunks {
            let chunk_dict =
                pyo3::types::PyDict::new(py);
            chunk_dict.set_item(
                "indices",
                &chunk.indices,
            )?;
            chunk_dict.set_item(
                "origin",
                &chunk.origin,
            )?;
            chunk_dict.set_item(
                "shape",
                &chunk.shape,
            )?;

            let shards_list =
                pyo3::types::PyList::empty(py);
            for shard in chunk.shards {
                let shard_dict =
                    pyo3::types::PyDict::new(py);
                shard_dict.set_item(
                    "indices",
                    &shard.indices,
                )?;
                shard_dict.set_item(
                    "origin",
                    &shard.origin,
                )?;
                shard_dict.set_item(
                    "shape",
                    &shard.shape,
                )?;
                shards_list.append(shard_dict)?;
            }
            chunk_dict.set_item(
                "shards",
                shards_list,
            )?;

            chunks_list.append(chunk_dict)?;
        }
        grid_dict
            .set_item("chunks", chunks_list)?;

        py_grids.append(grid_dict)?;
    }

    let result = pyo3::types::PyDict::new(py);
    result.set_item("grids", py_grids)?;
    result
        .set_item("coord_reads", coord_reads)?;
    Ok(result.into_any().unbind())
}
