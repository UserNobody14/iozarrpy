//! Streaming iterator for sync zarr backends.
//!
//! Thin wrapper around the tree-driven [`BatchPlanner`]: the join tree is built
//! once at iterator construction; each `__next__` reads one [`BatchPlan`] in
//! parallel via rayon, joins the per-leaf DataFrames, applies predicate /
//! restructuring, and yields.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::PyErr;
use pyo3::prelude::*;
use snafu::ensure;

use crate::chunk_plan::indexing::grid_join_reader::{
    assemble_batch_dataframe, flatten_reads,
};
use crate::errors::MaxChunksToReadExceededSnafu;
use crate::scan::chunk_to_df_from_grid_with_backend_sync;
use crate::scan::column_policy::ResolvedColumnPolicy;
use crate::shared::HasMetadataBackendSync;
use crate::shared::IStr;
use crate::shared::MaybeParIter;
use crate::shared::{
    ChunkedExpressionCompilerSync, FullyCachedZarrBackendSync,
};

/// Below this many chunk reads per batch the rayon scheduling overhead exceeds
/// the gain from parallel decode (single-chunk batches in particular regress
/// noticeably when forced through `par_iter`). Tune from benchmarks if decode
/// cost per chunk changes substantially.
const PARALLEL_CHUNK_READS: usize = 2;

use super::iterating_common::{
    DEFAULT_BATCH_SIZE, IteratorState,
    build_batches, distinct_chunks_in_batches,
    empty_streaming_schema_batch,
    expr_top_literal_bool,
    output_columns_for_streaming_batch,
    postprocess_batch,
};

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

    fn initialize(
        &mut self,
    ) -> Result<(), PyErr> {
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

        let (tree, _stats) = self
            .backend
            .compile_expression_to_tree_sync(
                &self.expr,
            )?;

        let literal_false =
            expr_top_literal_bool(&self.expr)
                == Some(false);

        let batches = match &tree {
            Some(t) => {
                build_batches(t, self.batch_size)
            }
            None => Vec::new(),
        };
        // Always emit a single empty-schema batch when there's no actual data
        // to scan (literal-false predicate or pushdown trimmed every grid).
        // Polars needs at least one batch with the correct columns to resolve
        // downstream filter / projection expressions.
        let emit_empty_schema_once =
            literal_false || batches.is_empty();

        if let Some(max_chunks) =
            self.max_chunks_to_read
        {
            let total_chunks =
                distinct_chunks_in_batches(
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

        let output_columns =
            output_columns_for_streaming_batch(
                meta.as_ref(),
                self.with_columns.as_ref(),
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
        if self.state.is_none() {
            return Err(PyErr::new::<
                pyo3::exceptions::PyRuntimeError,
                _,
            >(
                "Iterator not initialized",
            ));
        }

        let state = self.state.as_mut().unwrap();

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
            let meta = state.meta.clone();

            let chunk_dfs: Vec<(usize, DataFrame)> = reads
                .maybe_par_iter(PARALLEL_CHUNK_READS)
                .map_collect(|r| {
                    let df = chunk_to_df_from_grid_with_backend_sync(
                        backend.as_ref(),
                        r.idx.clone(),
                        r.sig.as_ref(),
                        &r.array_shape,
                        &r.vars,
                        expanded_with_columns.as_ref(),
                        r.subset.as_ref(),
                        &meta,
                    )?;
                    Ok::<_, crate::errors::BackendError>((r.leaf_idx, df))
                })?;

            let combined =
                assemble_batch_dataframe(
                    &plan, chunk_dfs,
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
impl ZarrIterator {
    fn __iter__(
        slf: PyRef<'_, Self>,
    ) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(
        mut slf: PyRefMut<'_, Self>,
    ) -> PyResult<Option<Py<PyAny>>> {
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
