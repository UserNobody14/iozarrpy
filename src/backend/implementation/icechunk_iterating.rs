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
use snafu::ResultExt;
use snafu::ensure;
use tokio::sync::Semaphore;

use crate::IStr;
use crate::errors::BackendError;
use crate::errors::CreateTokioRuntimeForSyncStoreSnafu;
use crate::errors::MaxChunksToReadExceededSnafu;
use crate::scan::async_scan::chunk_to_df_from_grid_with_backend;
use crate::scan::column_policy::ResolvedColumnPolicy;
use crate::shared::ChunkedExpressionCompilerAsync;
use crate::shared::HasMetadataBackendAsync;

use super::FullyCachedIcechunkBackendAsync;
use super::iterating_common::{
    DEFAULT_BATCH_SIZE, IteratorState,
    OwnedGridGroup, collect_chunks_for_batch,
    combine_and_postprocess_batch,
    output_columns_for_streaming_batch,
    sort_grid_groups_for_streaming_join,
    streaming_batch_row_cap,
};

const DEFAULT_MAX_CONCURRENCY: usize = 32;

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
            .context(
                CreateTokioRuntimeForSyncStoreSnafu {
                    store: "icechunk_iterating".to_string(),
                    prefix: "".to_string(),
                },
            )?;

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

        let (grid_groups, meta, expanded_with_columns, _effective_with_columns) = self.runtime.block_on(async {
            let meta = backend.metadata().await?;
            let effective_with_columns = with_columns.clone().or_else(|| {
                Some(meta.tidy_column_order(None).into_iter().collect())
            });

            let policy = ResolvedColumnPolicy::new(
                effective_with_columns.clone(),
                &expr,
                &meta,
            );
            let expanded_with_columns =
                policy.physical_superset().cloned();

            let (mut grouped_plan, _stats) =
                backend.compile_expression_async(&expr).await?;

            if let Some(max_chunks) = max_chunks_to_read {
                let total_chunks =
                    grouped_plan.total_unique_chunks()?;

                ensure!(total_chunks <= max_chunks, MaxChunksToReadExceededSnafu {
                    total_chunks,
                    max_chunks,
                });
            }

            if let Some(ref superset) =
                expanded_with_columns
            {
                grouped_plan.augment_with_physical_vars(
                    superset,
                    meta.as_ref(),
                )?;
            }

            let mut grid_groups: Vec<OwnedGridGroup> =
                grouped_plan
                    .iter_consolidated_chunks()
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .map(OwnedGridGroup::from_consolidated)
                    .collect();
            sort_grid_groups_for_streaming_join(
                &mut grid_groups,
            );

            Ok::<_, BackendError>((
                grid_groups,
                meta,
                expanded_with_columns,
                effective_with_columns,
            ))
        })?;

        let output_columns =
            output_columns_for_streaming_batch(
                meta.as_ref(),
                with_columns.as_ref(),
                expanded_with_columns.as_ref(),
            );

        let batch_row_cap =
            streaming_batch_row_cap(
                grid_groups.len(),
                self.batch_size,
            );

        self.state = Some(IteratorState {
            grid_groups,
            current_group_idx: 0,
            current_chunk_idx: 0,
            current_batch: Vec::new(),
            current_batch_rows: 0,
            total_rows_yielded: 0,
            num_rows_limit: self.num_rows,
            meta,
            output_columns,
            expanded_with_columns,
            predicate: self.expr.clone(),
            batch_row_cap,
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

            let chunks_to_read =
                collect_chunks_for_batch(
                    group,
                    &mut state.current_chunk_idx,
                    state.current_batch_rows,
                    state.batch_row_cap,
                );

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
                let meta = state.meta.clone();

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
                        let meta = meta.clone();

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
                                &meta,
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
                >= state.batch_row_cap
            {
                break;
            }
        }

        if !state.current_batch.is_empty() {
            let batch_dfs = std::mem::take(
                &mut state.current_batch,
            );
            state.current_batch_rows = 0;

            // let meta = state.meta.clone();
            // let wc = state
            //     .expanded_with_columns
            //     .clone();
            // let backend = self.backend.clone();
            // let runtime = &self.runtime;
            let result =
                combine_and_postprocess_batch(
                    batch_dfs,
                    state,
                    // move |df| {
                    //     runtime.block_on(
                    //         crate::scan::enrich_df_vars::enrich_df_missing_requested_vars_async(
                    //             backend.as_ref(),
                    //             df,
                    //             &meta,
                    //             wc.as_ref(),
                    //         ),
                    //     )
                    // },
                )?;
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
