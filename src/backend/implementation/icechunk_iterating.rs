//! Streaming iterator for Icechunk-backed zarr datasets.
//!
//! Sync iterator that blocks on async chunk reads. Uses the unified
//! [`crate::chunk_plan::indexing::grid_join_reader`] for batching and joining,
//! and tokio concurrency within each batch for I/O parallelism.

use std::collections::BTreeSet;
use std::sync::Arc;

use futures::stream::{
    FuturesUnordered, StreamExt,
};
use polars::prelude::*;
use pyo3::PyErr;
use pyo3::prelude::*;
use snafu::ResultExt;
use snafu::ensure;
use tokio::sync::Semaphore;

use crate::chunk_plan::GridJoinTree;
use crate::chunk_plan::indexing::grid_join_reader::{
    assemble_batch_dataframe, flatten_reads,
};
use crate::errors::BackendError;
use crate::errors::CreateTokioRuntimeForSyncStoreSnafu;
use crate::errors::MaxChunksToReadExceededSnafu;
use crate::scan::async_scan::chunk_to_df_from_grid_with_backend;
use crate::scan::column_policy::ResolvedColumnPolicy;
use crate::shared::ChunkedExpressionCompilerAsync;
use crate::shared::HasMetadataBackendAsync;
use crate::shared::IStr;

use super::FullyCachedIcechunkBackendAsync;
use super::iterating_common::{
    DEFAULT_BATCH_SIZE, IteratorState,
    build_batches, distinct_chunks_in_batches,
    empty_streaming_schema_batch,
    expr_top_literal_bool,
    output_columns_for_streaming_batch,
    postprocess_batch,
};

const DEFAULT_MAX_CONCURRENCY: usize = 32;

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
            .context(CreateTokioRuntimeForSyncStoreSnafu {
                store: "icechunk_iterating".to_string(),
                prefix: "".to_string(),
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
        let batch_size = self.batch_size;

        let (tree, batches, meta, expanded_with_columns, emit_empty_schema_once) =
            self.runtime.block_on(async {
                let meta = backend.metadata().await?;
                let effective_with_columns = with_columns.clone().or_else(|| {
                    Some(meta.tidy_column_order(None).into_iter().collect())
                });

                let policy = ResolvedColumnPolicy::new(
                    effective_with_columns.clone(),
                    &expr,
                    &meta,
                );
                let expanded_with_columns = policy.physical_superset().cloned();

                let (grouped_plan, _stats) =
                    backend.compile_expression_async(&expr).await?;

                let literal_false =
                    expr_top_literal_bool(&expr) == Some(false);

                let groups = grouped_plan
                    .owned_grid_groups_for_io(literal_false, meta.as_ref())?;

                let tree = GridJoinTree::build(groups);
                let batches = match &tree {
                    Some(t) => build_batches(t, batch_size),
                    None => Vec::new(),
                };
                // Emit one empty-schema batch when we have nothing to read so
                // Polars can still resolve downstream projection / filter expressions.
                let emit_empty_schema_once = literal_false || batches.is_empty();

                if let Some(max_chunks) = max_chunks_to_read {
                    let total_chunks = distinct_chunks_in_batches(&batches);
                    ensure!(
                        total_chunks <= max_chunks,
                        MaxChunksToReadExceededSnafu {
                            total_chunks,
                            max_chunks,
                        }
                    );
                }

                Ok::<_, BackendError>((
                    tree,
                    batches,
                    meta,
                    expanded_with_columns,
                    emit_empty_schema_once,
                ))
            })?;

        let output_columns =
            output_columns_for_streaming_batch(
                meta.as_ref(),
                with_columns.as_ref(),
                expanded_with_columns.as_ref(),
            );

        self.state = Some(IteratorState {
            tree,
            batches,
            cursor: 0,
            total_rows_yielded: 0,
            num_rows_limit: self.num_rows,
            meta,
            output_columns,
            expanded_with_columns,
            predicate: self.expr.clone(),
            emit_empty_schema_once,
        });

        Ok(())
    }

    fn next_batch(
        &mut self,
    ) -> Result<Option<DataFrame>, PyErr> {
        let state = self.state.as_mut().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Iterator not initialized",
            )
        })?;

        if let Some(limit) = state.num_rows_limit
            && state.total_rows_yielded >= limit
        {
            return Ok(None);
        }

        if state.emit_empty_schema_once {
            state.emit_empty_schema_once = false;
            return empty_streaming_schema_batch(state)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "{e:?}"
                    ))
                })
                .map(Some);
        }

        loop {
            if state.cursor >= state.batches.len()
            {
                return Ok(None);
            }
            let plan = state.batches
                [state.cursor]
                .clone();
            state.cursor += 1;

            let leaves = state
                .tree
                .as_ref()
                .expect("tree present when batches non-empty")
                .leaves();
            let reads =
                flatten_reads(&plan, &leaves);
            if reads.is_empty() {
                continue;
            }

            let backend = self.backend.clone();
            let expanded_with_columns = state
                .expanded_with_columns
                .clone();
            let max_concurrency =
                self.max_concurrency;
            let meta = state.meta.clone();

            let chunk_dfs: Result<Vec<(usize, DataFrame)>, PyErr> =
                self.runtime.block_on(async {
                    let semaphore = Arc::new(Semaphore::new(max_concurrency));
                    let mut futs = FuturesUnordered::new();

                    for r in reads {
                        let sem = semaphore.clone();
                        let backend = backend.clone();
                        let expanded = expanded_with_columns.clone();
                        let meta = meta.clone();
                        let leaf_idx = r.leaf_idx;
                        let sig = r.sig.clone();
                        let array_shape = r.array_shape.clone();
                        let vars = r.vars.clone();
                        let idx = r.idx.clone();
                        let subset = r.subset.clone();

                        futs.push(async move {
                            let _permit =
                                sem.acquire().await.expect("semaphore closed");
                            let df = chunk_to_df_from_grid_with_backend(
                                backend.as_ref(),
                                idx,
                                sig.as_ref(),
                                &array_shape,
                                &vars,
                                expanded.as_ref(),
                                subset.as_ref(),
                                &meta,
                            )
                            .await?;
                            Ok::<_, BackendError>((leaf_idx, df))
                        });
                    }

                    let mut dfs = Vec::new();
                    while let Some(r) = futs.next().await {
                        dfs.push(r?);
                    }
                    Ok(dfs)
                });

            let combined =
                assemble_batch_dataframe(
                    &plan, chunk_dfs?,
                )?;
            let Some(combined) = combined else {
                continue;
            };
            let result = postprocess_batch(
                combined, state,
            )?;
            return Ok(Some(result));
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
