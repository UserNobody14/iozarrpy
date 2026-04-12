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
    LegacyBatchState, StreamingSchedule,
    collect_chunks_for_batch,
    combine_and_postprocess_batch,
    empty_streaming_schema_batch,
    expr_top_literal_bool,
    output_columns_for_streaming_batch,
};
use crate::chunk_plan::{
    GridGroupExecutionOpts, ScheduleBuilt,
    apply_streaming_batch_io_cut,
    build_streaming_schedule,
    distinct_chunk_slots_in_batches,
    streaming_grid_chunk_read_count,
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
        let batch_size = self.batch_size;

        let (
            grid_groups,
            meta,
            expanded_with_columns,
            emit_empty_schema_once,
            schedule,
        ) = self.runtime.block_on(async {
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

            let (grouped_plan, _stats) = backend.compile_expression_async(&expr).await?;

            let emit_empty_schema_once =
                expr_top_literal_bool(&expr) == Some(false);

            let groups_uncut = grouped_plan.owned_grid_groups_for_io(
                meta.as_ref(),
                GridGroupExecutionOpts {
                    literal_false_clear: emit_empty_schema_once,
                    drop_redundant_1d_coords: !emit_empty_schema_once,
                    streaming_batch_io_cut: None,
                },
            )?;

            let grid_groups = if emit_empty_schema_once {
                groups_uncut
            } else {
                apply_streaming_batch_io_cut(
                    groups_uncut,
                    policy.predicate_refs(),
                    with_columns.as_ref(),
                    meta.as_ref(),
                )
            };

            let built = build_streaming_schedule(
                &grid_groups,
                meta.as_ref(),
                batch_size,
            );

            let schedule = match built {
                ScheduleBuilt::JoinClosed { batches } => {
                    if let Some(max_chunks) = max_chunks_to_read {
                        let total_chunks =
                            distinct_chunk_slots_in_batches(
                                &batches,
                            );
                        ensure!(
                            total_chunks <= max_chunks,
                            MaxChunksToReadExceededSnafu {
                                total_chunks,
                                max_chunks,
                            }
                        );
                    }
                    StreamingSchedule::JoinClosed {
                        batches,
                        cursor: 0,
                    }
                }
                ScheduleBuilt::Legacy => {
                    if let Some(max_chunks) = max_chunks_to_read {
                        let total_chunks =
                            streaming_grid_chunk_read_count(
                                &grid_groups,
                            );
                        ensure!(
                            total_chunks <= max_chunks,
                            MaxChunksToReadExceededSnafu {
                                total_chunks,
                                max_chunks,
                            }
                        );
                    }
                    StreamingSchedule::Legacy(
                        LegacyBatchState {
                            current_group_idx: 0,
                            current_chunk_idx: 0,
                            current_batch: Vec::new(),
                            current_batch_rows: 0,
                        },
                    )
                }
            };

            Ok::<_, BackendError>((
                grid_groups,
                meta,
                expanded_with_columns,
                emit_empty_schema_once,
                schedule,
            ))
        })?;

        let output_columns =
            output_columns_for_streaming_batch(
                meta.as_ref(),
                with_columns.as_ref(),
                expanded_with_columns.as_ref(),
            );

        self.state = Some(IteratorState {
            grid_groups,
            schedule,
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
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Iterator not initialized"))?;

        if let Some(limit) = state.num_rows_limit
            && state.total_rows_yielded >= limit
        {
            return Ok(None);
        }

        if state.emit_empty_schema_once {
            state.emit_empty_schema_once = false;
            return empty_streaming_schema_batch(state)
                .map_err(|e| {
                    PyErr::new::<
                        pyo3::exceptions::PyRuntimeError,
                        _,
                    >(format!("{e:?}"))
                })
                .map(Some);
        }

        match &mut state.schedule {
            StreamingSchedule::JoinClosed {
                batches,
                cursor,
            } => {
                if *cursor >= batches.len() {
                    return Ok(None);
                }
                let mut reads = batches[*cursor]
                    .reads
                    .clone();
                *cursor += 1;
                reads.sort_by_key(|r| {
                    (r.group_idx, r.chunk_slot)
                });

                let backend =
                    self.backend.clone();
                let expanded_with_columns = state
                    .expanded_with_columns
                    .clone();
                let max_concurrency =
                    self.max_concurrency;
                let meta = state.meta.clone();

                let chunk_dfs: Result<Vec<DataFrame>, PyErr> =
                    self.runtime.block_on(async {
                        let semaphore =
                            Arc::new(Semaphore::new(
                                max_concurrency,
                            ));
                        let mut futs =
                            FuturesUnordered::new();

                        for r in reads {
                            let g = &state.grid_groups
                                [r.group_idx];
                            let idx = g.chunk_indices
                                [r.chunk_slot]
                                .clone();
                            let subset = g.chunk_subsets
                                [r.chunk_slot]
                                .clone();
                            let vars: Vec<IStr> =
                                g.vars.clone();
                            let sig = g.sig.clone();
                            let array_shape =
                                g.array_shape.clone();
                            let sem = semaphore.clone();
                            let backend =
                                backend.clone();
                            let expanded = expanded_with_columns.clone();
                            let meta = meta.clone();

                            futs.push(async move {
                                let _permit = sem
                                    .acquire()
                                    .await
                                    .expect("semaphore closed");
                                chunk_to_df_from_grid_with_backend(
                                    backend.as_ref(),
                                    idx,
                                    sig.as_ref(),
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
                        while let Some(r) =
                            futs.next().await
                        {
                            dfs.push(r?);
                        }
                        Ok(dfs)
                    });

                let result =
                    combine_and_postprocess_batch(
                        chunk_dfs?, state,
                    )?;
                Ok(Some(result))
            }
            StreamingSchedule::Legacy(leg) => {
                while leg.current_group_idx
                    < state.grid_groups.len()
                {
                    let group = &state
                        .grid_groups
                        [leg.current_group_idx];

                    let chunks_to_read =
                        collect_chunks_for_batch(
                            group,
                            &mut leg.current_chunk_idx,
                            leg.current_batch_rows,
                            self.batch_size,
                        );

                    if !chunks_to_read.is_empty()
                    {
                        let vars =
                            group.vars.clone();
                        let backend =
                            self.backend.clone();
                        let sig =
                            group.sig.clone();
                        let array_shape = group
                            .array_shape
                            .clone();
                        let expanded_with_columns = state
                            .expanded_with_columns
                            .clone();
                        let max_concurrency =
                            self.max_concurrency;
                        let meta =
                            state.meta.clone();

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
                            leg.current_batch_rows +=
                                df.height();
                            leg.current_batch
                                .push(df);
                        }
                    }

                    if leg.current_chunk_idx
                        >= group
                            .chunk_indices
                            .len()
                    {
                        leg.current_group_idx +=
                            1;
                        leg.current_chunk_idx = 0;
                    }

                    if leg.current_batch_rows
                        >= self.batch_size
                    {
                        break;
                    }
                }

                if !leg.current_batch.is_empty() {
                    let batch_dfs =
                        std::mem::take(
                            &mut leg
                                .current_batch,
                        );
                    leg.current_batch_rows = 0;

                    let result =
                        combine_and_postprocess_batch(
                            batch_dfs,
                            state,
                        )?;
                    Ok(Some(result))
                } else {
                    Ok(None)
                }
            }
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
