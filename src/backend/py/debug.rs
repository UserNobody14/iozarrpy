use crate::meta::ZarrMeta;
use crate::shared::{
    ChunkedDataBackendAsync,
    ChunkedExpressionCompilerSync, HasStore,
};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use std::collections::BTreeSet;
use zarrs::array::Array;

use crate::shared::ChunkedExpressionCompilerAsync;
use crate::shared::{
    ChunkedDataBackendSync, HasAsyncStore,
    HasMetadataBackendAsync,
    HasMetadataBackendSync,
};

#[derive(Clone)]
pub(crate) struct ChunkInfo {
    indices: Vec<u64>,
    origin: Vec<u64>,
    shape: Vec<u64>,
}

#[derive(Clone)]
pub(crate) struct GridInfo {
    dims: Vec<String>,
    variables: Vec<String>,
    chunks: Vec<ChunkInfo>,
}

// Helper for async grid extraction logic
pub(crate) async fn extract_grids<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync
        + HasAsyncStore,
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
    for result in grouped_plan.iter_consolidated_chunks() {
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
            sig.chunk_shape().to_vec();

        // Sharding detection/metadata
        let store = backend.async_store().clone();
        let mut outer_chunk_shape: Option<
            Vec<u64>,
        > = None;
        let mut array_shape: Option<Vec<u64>> =
            None;
        if let Some(var) = vars.first() {
            let path =
                crate::shared::normalize_path(
                    var,
                );
            if let Ok(arr) = Array::async_open(
                store.clone(),
                &path,
            )
            .await
            {
                let zero = vec![
                        0u64;
                        arr.dimensionality()
                    ];
                outer_chunk_shape = arr
                    .chunk_grid()
                    .chunk_shape_u64(&zero)
                    .ok()
                    .flatten();
                array_shape =
                    Some(arr.shape().to_vec());
            }
        }
        let is_sharded = outer_chunk_shape
            .as_ref()
            .is_some_and(|outer| {
                outer != &inner_chunk_shape
            });

        let mut chunks: Vec<ChunkInfo> =
            Vec::new();
        // For sharded arrays, multiple inner chunk indices can map to the same
        // outer shard; deduplicate at the shard level.
        let mut seen_shards: BTreeSet<Vec<u64>> =
            BTreeSet::new();

        for inner_idx in &inner_chunk_indices {
            if is_sharded {
                let Some(outer_shape) =
                    outer_chunk_shape.as_ref()
                else {
                    continue;
                };
                // Map inner chunk index -> shard index
                let outer_idx: Vec<u64> =
                    inner_idx
                        .iter()
                        .zip(
                            inner_chunk_shape
                                .iter(),
                        )
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

                if !seen_shards
                    .insert(outer_idx.clone())
                {
                    continue;
                }

                let origin: Vec<u64> = outer_idx
                    .iter()
                    .zip(outer_shape.iter())
                    .map(|(&i, &sz)| {
                        i.saturating_mul(sz)
                    })
                    .collect();

                // Compute actual shard shape at boundaries
                let shape: Vec<u64> =
                    match array_shape.as_ref() {
                        Some(a_shape)
                            if a_shape.len()
                                == outer_shape
                                    .len() =>
                        {
                            origin
                                .iter()
                                .zip(a_shape.iter())
                                .zip(
                                    outer_shape
                                        .iter(),
                                )
                                .map(
                                    |(
                                        (&o, &a),
                                        &s,
                                    )| {
                                        if o >= a {
                                            0
                                        } else {
                                            (a - o)
                                                .min(s)
                                        }
                                    },
                                )
                                .collect()
                        }
                        _ => outer_shape.clone(),
                    };

                chunks.push(ChunkInfo {
                    indices: outer_idx,
                    origin,
                    shape,
                });
            } else {
                // inner_chunk_indices already deduplicated; compute origin directly.
                let chunk_shape =
                    sig.chunk_shape();
                let origin: Vec<u64> = inner_idx
                    .iter()
                    .zip(chunk_shape.iter())
                    .map(|(&i, &s)| i * s)
                    .collect();
                chunks.push(ChunkInfo {
                    indices: inner_idx.clone(),
                    origin,
                    shape: chunk_shape.to_vec(),
                });
            }
        }

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
        + ChunkedDataBackendSync
        + HasStore,
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
    for result in grouped_plan.iter_consolidated_chunks() {
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
            sig.chunk_shape().to_vec();

        // Sharding detection/metadata
        let store = backend.store().clone();
        let mut outer_chunk_shape: Option<
            Vec<u64>,
        > = None;
        let mut array_shape: Option<Vec<u64>> =
            None;
        if let Some(var) = vars.first() {
            let path =
                crate::shared::normalize_path(
                    var,
                );
            if let Ok(arr) =
                Array::open(store.clone(), &path)
            {
                let zero = vec![
                        0u64;
                        arr.dimensionality()
                    ];
                outer_chunk_shape = arr
                    .chunk_grid()
                    .chunk_shape_u64(&zero)
                    .ok()
                    .flatten();
                array_shape =
                    Some(arr.shape().to_vec());
            }
        }
        let is_sharded = outer_chunk_shape
            .as_ref()
            .is_some_and(|outer| {
                outer != &inner_chunk_shape
            });

        let mut chunks: Vec<ChunkInfo> =
            Vec::new();
        // For sharded arrays, multiple inner chunk indices can map to the same
        // outer shard; deduplicate at the shard level.
        let mut seen_shards: BTreeSet<Vec<u64>> =
            BTreeSet::new();

        for inner_idx in &inner_chunk_indices {
            if is_sharded {
                let Some(outer_shape) =
                    outer_chunk_shape.as_ref()
                else {
                    continue;
                };
                // Map inner chunk index -> shard index
                let outer_idx: Vec<u64> =
                    inner_idx
                        .iter()
                        .zip(
                            inner_chunk_shape
                                .iter(),
                        )
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

                if !seen_shards
                    .insert(outer_idx.clone())
                {
                    continue;
                }

                let origin: Vec<u64> = outer_idx
                    .iter()
                    .zip(outer_shape.iter())
                    .map(|(&i, &sz)| {
                        i.saturating_mul(sz)
                    })
                    .collect();

                // Compute actual shard shape at boundaries
                let shape: Vec<u64> =
                    match array_shape.as_ref() {
                        Some(a_shape)
                            if a_shape.len()
                                == outer_shape
                                    .len() =>
                        {
                            origin
                                .iter()
                                .zip(a_shape.iter())
                                .zip(
                                    outer_shape
                                        .iter(),
                                )
                                .map(
                                    |(
                                        (&o, &a),
                                        &s,
                                    )| {
                                        if o >= a {
                                            0
                                        } else {
                                            (a - o)
                                                .min(s)
                                        }
                                    },
                                )
                                .collect()
                        }
                        _ => outer_shape.clone(),
                    };

                chunks.push(ChunkInfo {
                    indices: outer_idx,
                    origin,
                    shape,
                });
            } else {
                // inner_chunk_indices already deduplicated; compute origin directly.
                let chunk_shape =
                    sig.chunk_shape();
                let origin: Vec<u64> = inner_idx
                    .iter()
                    .zip(chunk_shape.iter())
                    .map(|(&i, &s)| i * s)
                    .collect();
                chunks.push(ChunkInfo {
                    indices: inner_idx.clone(),
                    origin,
                    shape: chunk_shape.to_vec(),
                });
            }
        }

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

