use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::PyErr;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::chunk_plan::ChunkSubset;
use crate::meta::ZarrMeta;
use crate::scan::chunk_to_df_from_grid_with_backend_sync;
use crate::shared::HasMetadataBackendSync;
use crate::shared::{
    ChunkedExpressionCompilerSync,
    FullyCachedZarrBackendSync,
    combine_chunk_dataframes,
    expand_projection_to_flat_paths,
    restructure_to_structs,
};
use crate::IStr;

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
    // Owned grid groups (converted from borrowed ConsolidatedGridGroup)
    grid_groups: Vec<OwnedGridGroup>,
    current_group_idx: usize,
    current_chunk_idx: usize,

    // Batch accumulation
    current_batch: Vec<DataFrame>,
    current_batch_rows: usize,

    // Limits
    total_rows_yielded: usize,
    num_rows_limit: Option<usize>,

    // Metadata (cached for efficiency)
    meta: Arc<ZarrMeta>,
    expanded_with_columns: Option<BTreeSet<IStr>>,

    // Predicate for filtering rows (not just chunks)
    predicate: Expr,
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
                .unwrap_or(10_000),
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
        // Expand struct column names to flat paths for chunk reading
        let expanded_with_columns = self
            .with_columns
            .as_ref()
            .map(|cols| {
                expand_projection_to_flat_paths(
                    cols, &meta,
                )
            });

        // Compile grouped chunk plan
        let (grouped_plan, _stats) = self
            .backend
            .compile_expression_sync(
                &self.expr,
            )?;

        // Check max_chunks_to_read limit before doing any I/O
        if let Some(max_chunks) =
            self.max_chunks_to_read
        {
            let total_chunks = grouped_plan
                .total_unique_chunks()
                .map_err(|e| {
                    PyErr::new::<
                        pyo3::exceptions::PyValueError,
                        _,
                    >(e)
                })?;
            if total_chunks > max_chunks {
                return Err(PyErr::new::<
                    pyo3::exceptions::PyRuntimeError,
                    _,
                >(format!(
                    "max_chunks_to_read exceeded: {} chunks needed, limit is {}",
                    total_chunks, max_chunks
                )));
            }
        }

        // Collect all grid groups upfront and convert to owned data
        // We need to own the data rather than borrow it because GroupedChunkPlan
        // will be dropped after this function returns
        let grid_groups: Vec<OwnedGridGroup> = grouped_plan
            .iter_consolidated_chunks()
            .map(|result| {
                result.map_err(|e| {
                    PyErr::new::<
                        pyo3::exceptions::PyValueError,
                        _,
                    >(e)
                })
            })
            .collect::<Result<Vec<_>, PyErr>>()?
            .into_iter()
            .map(|group| {
                OwnedGridGroup {
                    sig: Arc::new(group.sig.clone()),
                    vars: group.vars.iter().map(|&v| v.clone()).collect(),
                    chunk_indices: group.chunk_indices,
                    chunk_subsets: group.chunk_subsets,
                    array_shape: group.array_shape,
                }
            })
            .collect();

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

            let mut chunks_to_read: Vec<(
                Vec<u64>,
                Option<ChunkSubset>,
            )> = Vec::new();
            while state.current_chunk_idx
                < group.chunk_indices.len()
                && state.current_batch_rows
                    < self.batch_size
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

                let chunk_dfs: Result<Vec<DataFrame>, PyErr> = chunks_to_read
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
                        )
                    })
                    .collect();

                let dfs = chunk_dfs?;

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
                >= self.batch_size
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

            // Combine all chunk DataFrames
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

            // Apply the predicate filter to the actual rows
            // Chunk planning only prunes chunks, but chunks may contain rows
            // that don't match the predicate
            let result = combined
                .lazy()
                .filter(state.predicate.clone())
                .collect()
                .map_err(|e| {
                    PyErr::new::<
                        pyo3::exceptions::PyRuntimeError,
                        _,
                    >(format!("Failed to filter data: {}", e))
                })?;

            // Apply row limit
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

            // For hierarchical data, convert flat path columns to struct columns
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
