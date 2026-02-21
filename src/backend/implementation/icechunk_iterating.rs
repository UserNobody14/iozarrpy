//! Streaming iterator for Icechunk-backed zarr datasets.
//!
//! Provides a sync iterator that yields DataFrames in batches by blocking on
//! async chunk reads. Uses tokio concurrency within each batch for efficient
//! I/O when scanning large time-chunked datasets.

use std::collections::BTreeSet;
use std::sync::Arc;

use futures::stream::{
    FuturesUnordered, StreamExt,
};
use polars::prelude::*;
use pyo3::PyErr;
use pyo3::prelude::*;
use tokio::sync::Semaphore;

use crate::chunk_plan::ChunkSubset;
use crate::errors::BackendError;
use crate::IStr;
use crate::meta::ZarrMeta;
use crate::scan::async_scan::chunk_to_df_from_grid_with_backend;
use crate::shared::ChunkedExpressionCompilerAsync;
use crate::shared::HasMetadataBackendAsync;
use crate::shared::{
    combine_chunk_dataframes,
    expand_projection_to_flat_paths,
    restructure_to_structs,
};

use super::FullyCachedIcechunkBackendAsync;

const DEFAULT_MAX_CONCURRENCY: usize = 32;
const DEFAULT_BATCH_SIZE: usize = 10_000;

/// Owned version of ConsolidatedGridGroup for storage in iterator state
struct OwnedGridGroup {
    sig: Arc<
        crate::chunk_plan::ChunkGridSignature,
    >,
    vars: Vec<IStr>,
    chunk_indices: Vec<Vec<u64>>,
    chunk_subsets: Vec<Option<ChunkSubset>>,
    array_shape: Vec<u64>,
}

struct IteratorState {
    grid_groups: Vec<OwnedGridGroup>,
    current_group_idx: usize,
    current_chunk_idx: usize,

    current_batch: Vec<DataFrame>,
    current_batch_rows: usize,

    total_rows_yielded: usize,
    num_rows_limit: Option<usize>,

    meta: Arc<ZarrMeta>,
    expanded_with_columns: Option<BTreeSet<IStr>>,

    predicate: Expr,
}

/// Streaming iterator for Icechunk-backed zarr that yields DataFrames in batches.
///
/// Blocks on async I/O internally, using tokio concurrency for chunk reads
/// within each batch. Enables memory-efficient streaming when scanning
/// time-chunked data (e.g., a single point across a year).
#[pyclass]
pub struct IcechunkIterator {
    backend: Arc<FullyCachedIcechunkBackendAsync>,
    expr: Expr,
    with_columns: Option<BTreeSet<IStr>>,
    max_chunks_to_read: Option<usize>,
    num_rows: Option<usize>,
    batch_size: usize,
    max_concurrency: usize,

    runtime: tokio::runtime::Runtime,
    state: Option<IteratorState>,
}

impl IcechunkIterator {
    pub(crate) fn new(
        backend: Arc<
            FullyCachedIcechunkBackendAsync,
        >,
        expr: Expr,
        with_columns: Option<BTreeSet<IStr>>,
        max_chunks_to_read: Option<usize>,
        num_rows: Option<usize>,
        batch_size: Option<usize>,
        max_concurrency: Option<usize>,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to create tokio runtime: {}",
                    e
                ))
            })?;

        Ok(Self {
            backend,
            expr,
            with_columns,
            max_chunks_to_read,
            num_rows,
            batch_size: batch_size
                .unwrap_or(DEFAULT_BATCH_SIZE),
            max_concurrency: max_concurrency
                .filter(|&v| v > 0)
                .unwrap_or(
                    DEFAULT_MAX_CONCURRENCY,
                ),
            runtime,
            state: None,
        })
    }

    fn initialize(
        &mut self,
    ) -> Result<(), PyErr> {
        let backend = self.backend.clone();
        let expr = self.expr.clone();
        let with_columns =
            self.with_columns.clone();
        let max_chunks_to_read =
            self.max_chunks_to_read;

        let (grid_groups, meta, expanded_with_columns) = self.runtime.block_on(async {
            let meta = backend.metadata().await?;

            let expanded_with_columns =
                with_columns.as_ref().map(|cols| expand_projection_to_flat_paths(cols, &meta));

            let (grouped_plan, _stats) = backend.compile_expression_async(&expr).await.map_err(
                |e: BackendError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()),
            )?;

            if let Some(max_chunks) = max_chunks_to_read {
                let total_chunks = grouped_plan.total_unique_chunks().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
                })?;
                if total_chunks > max_chunks {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "max_chunks_to_read exceeded: {} chunks needed, limit is {}",
                        total_chunks, max_chunks
                    )));
                }
            }

            let grid_groups: Vec<OwnedGridGroup> = grouped_plan
                .iter_consolidated_chunks()
                .map(|result| {
                    result.map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
                    })
                })
                .collect::<Result<Vec<_>, PyErr>>()?
                .into_iter()
                .map(|group| OwnedGridGroup {
                    sig: Arc::new(group.sig.clone()),
                    vars: group.vars.iter().map(|&v| v.clone()).collect(),
                    chunk_indices: group.chunk_indices,
                    chunk_subsets: group.chunk_subsets,
                    array_shape: group.array_shape,
                })
                .collect();

            Ok::<_, PyErr>((grid_groups, meta, expanded_with_columns))
        })?;

        self.state = Some(IteratorState {
            grid_groups,
            current_group_idx: 0,
            current_chunk_idx: 0,
            current_batch: Vec::new(),
            current_batch_rows: 0,
            total_rows_yielded: 0,
            num_rows_limit: self.num_rows,
            meta,
            expanded_with_columns,
            predicate: self.expr.clone(),
        });

        Ok(())
    }

    fn next_batch(
        &mut self,
    ) -> Result<Option<DataFrame>, PyErr> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Iterator not initialized"))?;

        if let Some(limit) = state.num_rows_limit
        {
            if state.total_rows_yielded >= limit {
                return Ok(None);
            }
        }

        while state.current_group_idx
            < state.grid_groups.len()
        {
            let group = &state.grid_groups
                [state.current_group_idx];

            let mut chunks_to_read: Vec<(Vec<u64>, Option<ChunkSubset>)> = Vec::new();
            while state.current_chunk_idx < group.chunk_indices.len()
                && state.current_batch_rows < self.batch_size
            {
                let ci = state.current_chunk_idx;
                chunks_to_read.push((
                    group.chunk_indices[ci].clone(),
                    group.chunk_subsets[ci].clone(),
                ));
                state.current_chunk_idx += 1;

                if chunks_to_read.len() >= 100 {
                    break;
                }
            }

            if !chunks_to_read.is_empty() {
                let vars = group.vars.clone();
                let backend =
                    self.backend.clone();
                let sig = group.sig.clone();
                let array_shape =
                    group.array_shape.clone();
                let expanded_with_columns = state
                    .expanded_with_columns
                    .clone();
                let max_concurrency =
                    self.max_concurrency;

                let chunk_dfs: Result<Vec<DataFrame>, PyErr> = self.runtime.block_on(async {
                    let semaphore = Arc::new(Semaphore::new(max_concurrency));
                    let mut futs = FuturesUnordered::new();

                    for (idx, subset) in chunks_to_read {
                        let sem = semaphore.clone();
                        let backend = backend.clone();
                        let sig = sig.clone();
                        let array_shape = array_shape.clone();
                        let vars = vars.clone();
                        let expanded = expanded_with_columns.clone();

                        futs.push(async move {
                            let _permit = sem.acquire().await.expect("semaphore closed");
                            chunk_to_df_from_grid_with_backend(
                                backend.as_ref(),
                                idx,
                                &sig,
                                &array_shape,
                                &vars,
                                expanded.as_ref(),
                                subset.as_ref(),
                            )
                            .await
                        });
                    }

                    let mut dfs = Vec::new();
                    while let Some(r) = futs.next().await {
                        dfs.push(r?);
                    }
                    Ok(dfs)
                });

                for df in chunk_dfs? {
                    state.current_batch_rows +=
                        df.height();
                    state.current_batch.push(df);
                }
            }

            if state.current_chunk_idx
                >= group.chunk_indices.len()
            {
                state.current_group_idx += 1;
                state.current_chunk_idx = 0;
            }

            if state.current_batch_rows
                >= self.batch_size
            {
                break;
            }
        }

        if !state.current_batch.is_empty() {
            let batch_dfs = std::mem::take(
                &mut state.current_batch,
            );
            state.current_batch_rows = 0;

            let combined = if batch_dfs.is_empty()
            {
                let keys: Vec<IStr> = state
                    .meta
                    .path_to_array
                    .keys()
                    .cloned()
                    .collect();
                let planning_meta =
                    state.meta.planning_meta();
                DataFrame::empty_with_schema(
                    &planning_meta.tidy_schema(
                        Some(keys.as_slice()),
                    ),
                )
            } else if batch_dfs.len() == 1 {
                batch_dfs
                    .into_iter()
                    .next()
                    .unwrap()
            } else {
                combine_chunk_dataframes(
                    batch_dfs,
                    &state.meta,
                )?
            };

            let result = combined
                .lazy()
                .filter(state.predicate.clone())
                .collect()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to filter data: {}",
                        e
                    ))
                })?;

            let result = if let Some(limit) =
                state.num_rows_limit
            {
                let remaining = limit
                    .saturating_sub(
                        state.total_rows_yielded,
                    );
                if remaining < result.height() {
                    result.slice(0, remaining)
                } else {
                    result
                }
            } else {
                result
            };

            state.total_rows_yielded +=
                result.height();

            let result =
                if state.meta.is_hierarchical() {
                    restructure_to_structs(
                        &result,
                        &state.meta,
                    )?
                } else {
                    result
                };

            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
}

#[pymethods]
impl IcechunkIterator {
    fn __iter__(
        slf: PyRef<'_, Self>,
    ) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(
        mut slf: PyRefMut<'_, Self>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if slf.state.is_none() {
            slf.initialize()?;
        }

        match slf.next_batch()? {
            Some(df) => {
                let py = slf.py();
                let py_df =
                    pyo3_polars::PyDataFrame(df);
                Ok(Some(
                    py_df
                        .into_pyobject(py)?
                        .unbind(),
                ))
            }
            None => Ok(None),
        }
    }
}
