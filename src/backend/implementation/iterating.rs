use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::PyErr;
use pyo3::prelude::*;
use rayon::prelude::*;
use snafu::ensure;

use crate::IStr;
use crate::errors::MaxChunksToReadExceededSnafu;
use crate::scan::chunk_to_df_from_grid_with_backend_sync;
use crate::scan::column_policy::ResolvedColumnPolicy;
use crate::shared::HasMetadataBackendSync;
use crate::shared::{
    ChunkedExpressionCompilerSync,
    FullyCachedZarrBackendSync,
};

use super::iterating_common::{
    DEFAULT_BATCH_SIZE, IteratorState, LegacyBatchState,
    StreamingSchedule, collect_chunks_for_batch,
    combine_and_postprocess_batch,
    empty_streaming_schema_batch,
    expr_top_literal_bool,
    output_columns_for_streaming_batch,
};
use crate::chunk_plan::{
    GridGroupExecutionOpts, ScheduleBuilt, apply_streaming_batch_io_cut,
    build_streaming_schedule, distinct_chunk_slots_in_batches,
    streaming_grid_chunk_read_count,
};

/// Version of scan zarr sync that returns an iterator
///
/// This iterator yields DataFrames in batches, allowing for memory-efficient
/// streaming of large Zarr datasets.
#[pyclass]
pub struct ZarrIterator {
    inner: ZarrIteratorInner,
}

impl ZarrIterator {
    pub(crate) fn new(
        backend: Arc<FullyCachedZarrBackendSync>,
        expr: polars::prelude::Expr,
        with_columns: Option<BTreeSet<IStr>>,
        max_chunks_to_read: Option<usize>,
        num_rows: Option<usize>,
        batch_size: Option<usize>,
    ) -> Self {
        Self {
            inner: ZarrIteratorInner::new(
                backend,
                expr,
                with_columns,
                max_chunks_to_read,
                num_rows,
                batch_size,
            ),
        }
    }
}

struct ZarrIteratorInner {
    backend: Arc<FullyCachedZarrBackendSync>,
    expr: polars::prelude::Expr,
    with_columns: Option<BTreeSet<IStr>>,
    max_chunks_to_read: Option<usize>,
    batch_size: usize,
    num_rows: Option<usize>,

    // State for iteration (lazy initialized)
    state: Option<IteratorState>,
}

impl ZarrIteratorInner {
    fn new(
        backend: Arc<FullyCachedZarrBackendSync>,
        expr: polars::prelude::Expr,
        with_columns: Option<BTreeSet<IStr>>,
        max_chunks_to_read: Option<usize>,
        num_rows: Option<usize>,
        batch_size: Option<usize>,
    ) -> Self {
        Self {
            backend,
            expr,
            with_columns,
            max_chunks_to_read,
            batch_size: batch_size
                .unwrap_or(DEFAULT_BATCH_SIZE),
            num_rows,
            state: None,
        }
    }

    /// Lazy initialization - compile the expression and prepare iteration state
    fn initialize(
        &mut self,
    ) -> Result<(), PyErr> {
        // Get metadata from backend
        let meta = self.backend.metadata()?;
        let effective_with_columns = self
            .with_columns
            .clone()
            .or_else(|| {
                Some(
                    meta.tidy_column_order(None)
                        .into_iter()
                        .collect(),
                )
            });
        // Expand struct column names to flat paths for chunk reading
        let with_set: Option<BTreeSet<IStr>> =
            effective_with_columns.as_ref().map(
                |cols| {
                    cols.iter().cloned().collect()
                },
            );
        let policy = ResolvedColumnPolicy::new(
            with_set, &self.expr, &meta,
        );
        let expanded_with_columns =
            policy.physical_superset().cloned();

        // Compile grouped chunk plan
        let (grouped_plan, _stats) = self
            .backend
            .compile_expression_sync(
                &self.expr,
            )?;

        let emit_empty_schema_once =
            expr_top_literal_bool(&self.expr)
                == Some(false);

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
                self.with_columns.as_ref(),
                meta.as_ref(),
            )
        };

        let built = build_streaming_schedule(
            &grid_groups,
            meta.as_ref(),
            self.batch_size,
        );

        let schedule = match built {
            ScheduleBuilt::JoinClosed { batches } => {
                if let Some(max_chunks) =
                    self.max_chunks_to_read
                {
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
                if let Some(max_chunks) =
                    self.max_chunks_to_read
                {
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

        let output_columns =
            output_columns_for_streaming_batch(
                meta.as_ref(),
                self.with_columns.as_ref(),
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

    /// Get the next batch of data
    fn next_batch(
        &mut self,
    ) -> Result<Option<DataFrame>, PyErr> {
        // Lazy initialization on first call
        if self.state.is_none() {
            return Err(PyErr::new::<
                pyo3::exceptions::PyRuntimeError,
                _,
            >(
                "Iterator not initialized",
            ));
        }

        let state = self.state.as_mut().unwrap();

        // Check if we've reached the row limit
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
                let mut reads =
                    batches[*cursor].reads.clone();
                *cursor += 1;
                reads.sort_by_key(|r| {
                    (r.group_idx, r.chunk_slot)
                });

                let backend = self.backend.clone();
                let expanded_with_columns = state
                    .expanded_with_columns
                    .clone();
                let meta = state.meta.clone();

                let dfs: Vec<DataFrame> = reads
                    .par_iter()
                    .map(|r| {
                        let g =
                            &state.grid_groups[r.group_idx];
                        let idx = g.chunk_indices[r.chunk_slot]
                            .clone();
                        let subset = g.chunk_subsets
                            [r.chunk_slot]
                            .clone();
                        let vars: Vec<IStr> =
                            g.vars.clone();
                        chunk_to_df_from_grid_with_backend_sync(
                            backend.as_ref(),
                            idx,
                            g.sig.as_ref(),
                            &g.array_shape,
                            &vars,
                            expanded_with_columns.as_ref(),
                            subset.as_ref(),
                            &meta,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let result =
                    combine_and_postprocess_batch(
                        dfs, state,
                    )?;
                Ok(Some(result))
            }
            StreamingSchedule::Legacy(leg) => {
                while leg.current_group_idx
                    < state.grid_groups.len()
                {
                    let group = &state.grid_groups
                        [leg.current_group_idx];

                    let chunks_to_read =
                        collect_chunks_for_batch(
                            group,
                            &mut leg.current_chunk_idx,
                            leg.current_batch_rows,
                            self.batch_size,
                        );

                    if !chunks_to_read.is_empty() {
                        let vars: Vec<IStr> =
                            group.vars.clone();

                        let backend =
                            self.backend.clone();
                        let sig = group.sig.clone();
                        let array_shape =
                            group.array_shape.clone();
                        let expanded_with_columns = state
                            .expanded_with_columns
                            .clone();
                        let meta = state.meta.clone();

                        let dfs: Vec<DataFrame> =
                            chunks_to_read
                                .par_iter()
                                .map(|(idx, subset)| {
                                    chunk_to_df_from_grid_with_backend_sync(
                                        backend.as_ref(),
                                        idx.clone(),
                                        &sig,
                                        &array_shape,
                                        &vars,
                                        expanded_with_columns.as_ref(),
                                        subset.as_ref(),
                                        &meta,
                                    )
                                })
                                .collect::<Result<
                                    Vec<_>,
                                    _,
                                >>()?;

                        for df in dfs {
                            leg.current_batch_rows +=
                                df.height();
                            leg.current_batch.push(df);
                        }
                    }

                    if leg.current_chunk_idx
                        >= group.chunk_indices.len()
                    {
                        leg.current_group_idx += 1;
                        leg.current_chunk_idx = 0;
                    }

                    if leg.current_batch_rows
                        >= self.batch_size
                    {
                        break;
                    }
                }

                if !leg.current_batch.is_empty() {
                    let batch_dfs = std::mem::take(
                        &mut leg.current_batch,
                    );
                    leg.current_batch_rows = 0;

                    let result =
                        combine_and_postprocess_batch(
                            batch_dfs, state,
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
impl ZarrIterator {
    fn __iter__(
        slf: PyRef<'_, Self>,
    ) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(
        mut slf: PyRefMut<'_, Self>,
    ) -> PyResult<Option<Py<PyAny>>> {
        // Lazy initialization on first call
        if slf.inner.state.is_none() {
            slf.inner.initialize()?;
        }

        match slf.inner.next_batch()? {
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
