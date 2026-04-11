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
    DEFAULT_BATCH_SIZE, IteratorState,
    OwnedGridGroup, collect_chunks_for_batch,
    combine_and_postprocess_batch,
    output_columns_for_streaming_batch,
    sort_grid_groups_for_streaming_join,
    streaming_batch_row_cap,
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
        let (mut grouped_plan, _stats) = self
            .backend
            .compile_expression_sync(
                &self.expr,
            )?;

        // See `scan_zarr_with_backend_sync`: limit counts predicate plan only.
        if let Some(max_chunks) =
            self.max_chunks_to_read
        {
            let total_chunks = grouped_plan
                .total_unique_chunks()?;
            ensure!(
                total_chunks <= max_chunks,
                MaxChunksToReadExceededSnafu {
                    total_chunks,
                    max_chunks,
                }
            );
        }

        if let Some(ref superset) =
            expanded_with_columns
        {
            grouped_plan
                .augment_with_physical_vars(
                    superset,
                    meta.as_ref(),
                )?;
        }

        // Collect all grid groups upfront and convert to owned data
        // We need to own the data rather than borrow it because GroupedChunkPlan
        // will be dropped after this function returns
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
        let batch_row_cap =
            streaming_batch_row_cap(
                grid_groups.len(),
                self.batch_size,
            );

        let output_columns =
            output_columns_for_streaming_batch(
                meta.as_ref(),
                self.with_columns.as_ref(),
                expanded_with_columns.as_ref(),
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

    /// Get the next batch of data
    fn next_batch(
        &mut self,
    ) -> Result<Option<DataFrame>, PyErr> {
        // Lazy initialization on first call
        if self.state.is_none() {
            // We need to pass num_rows here, but it's not stored in self anymore
            // We'll need to refactor to store it
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
        {
            if state.total_rows_yielded >= limit {
                return Ok(None);
            }
        }

        // Accumulate chunks until we have enough rows for a batch
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

                let dfs: Vec<DataFrame> = chunks_to_read
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
                    .collect::<Result<Vec<_>, _>>()?;

                // Add to batch and count rows
                for df in dfs {
                    state.current_batch_rows +=
                        df.height();
                    state.current_batch.push(df);
                }
            }

            // Move to next group if current group is exhausted
            // IMPORTANT: Do this BEFORE checking batch size, so we don't re-read
            if state.current_chunk_idx
                >= group.chunk_indices.len()
            {
                state.current_group_idx += 1;
                state.current_chunk_idx = 0;
            }

            // Check if we have enough rows for a batch AFTER advancing position
            if state.current_batch_rows
                >= state.batch_row_cap
            {
                break;
            }
        }

        // If we have accumulated any data, return it
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
            let result =
                combine_and_postprocess_batch(
                    batch_dfs,
                    state,
                    // move |df| {
                    //     crate::scan::enrich_df_vars::enrich_df_missing_requested_vars(
                    //         backend.as_ref(),
                    //         df,
                    //         &meta,
                    //         wc.as_ref(),
                    //     )
                    // },
                )?;
            Ok(Some(result))
        } else {
            // No more data
            Ok(None)
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
