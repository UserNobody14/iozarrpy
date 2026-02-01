//! Python bindings for ZarrBackend.
//!
//! Exposes the caching backend to Python with scan methods.

use std::fmt::Pointer;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_polars::PySchema;

use crate::backend::compile::ChunkedExpressionCompilerWithBackendAsync;
use crate::backend::traits::{
    EvictableChunkCacheAsync,
    HasMetadataBackendAsync,
};
use crate::backend::zarr::{
    FullyCachedZarrBackendAsync,
    ZarrBackendAsync, to_fully_cached_async,
};
use crate::py::expr_extract::extract_expr;
use crate::scan::chunk_to_df::chunk_to_df_from_grid_with_backend;
use crate::store::StoreInput;
use crate::{IStr, IntoIStr};

/// Python-exposed Zarr backend with caching and scan methods.
///
/// The backend owns the store and caches coordinate array chunks and metadata
/// across multiple scan operations.
#[pyclass(name = "ZarrBackend")]
pub struct PyZarrBackend {
    inner: Arc<FullyCachedZarrBackendAsync>,
}

#[pymethods]
impl PyZarrBackend {
    /// Create a backend from a URL string.
    ///
    /// # Arguments
    /// * `url` - URL to the zarr store (e.g., "s3://bucket/path.zarr")
    /// * `max_cache_entries` - Maximum cached coord chunks (0 = unlimited)
    #[staticmethod]
    #[pyo3(signature = (url, max_cache_entries=0))]
    fn from_url(
        url: String,
        max_cache_entries: usize,
    ) -> PyResult<Self> {
        let backend = ZarrBackendAsync::new(
            StoreInput::Url(url),
        )?;

        let backend =
            to_fully_cached_async(backend)?;
        Ok(Self {
            inner: Arc::new(backend),
        })
    }

    /// Create a backend from an ObjectStore instance.
    ///
    /// # Arguments
    /// * `store` - ObjectStore instance (from rainbear.store or obstore)
    /// * `prefix` - Optional path prefix within the store
    /// * `max_cache_entries` - Maximum cached coord chunks (0 = unlimited)
    #[staticmethod]
    #[pyo3(signature = (store, prefix=None, max_cache_entries=0))]
    fn from_store(
        store: &Bound<'_, PyAny>,
        prefix: Option<String>,
        max_cache_entries: usize,
    ) -> PyResult<Self> {
        let store_input =
            StoreInput::from_py(store, prefix)?;
        let backend =
            ZarrBackendAsync::new(store_input)?;
        let backend =
            to_fully_cached_async(backend)?;
        Ok(Self {
            inner: Arc::new(backend),
        })
    }

    /// Async scan the zarr store and return a DataFrame.
    ///
    /// Uses the backend's cached coordinates for efficient predicate pushdown.
    ///
    /// # Arguments
    /// * `predicate` - Polars expression for filtering
    /// * `variables` - Optional list of variable names to read
    /// * `max_concurrency` - Maximum concurrent chunk reads
    /// * `with_columns` - Optional list of columns to include
    /// * `max_chunks_to_read` - Maximum number of chunks to read (safety limit)
    #[pyo3(signature = (predicate, variables=None, max_concurrency=None, with_columns=None, max_chunks_to_read=None))]
    fn scan_zarr_async<'py>(
        &self,
        py: Python<'py>,
        predicate: &Bound<'_, PyAny>,
        variables: Option<Vec<String>>,
        max_concurrency: Option<usize>,
        with_columns: Option<Vec<String>>,
        max_chunks_to_read: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use polars::prelude::IntoLazy;
        use pyo3_polars::PyDataFrame;
        use std::collections::BTreeSet;

        let expr = extract_expr(predicate)?;
        let expr2 = expr.clone();

        // Combine with_columns and variables into a single projection set
        let with_columns_set: Option<
            BTreeSet<IStr>,
        > = match (with_columns, variables) {
            (Some(cols), Some(vars)) => {
                let mut set: BTreeSet<IStr> =
                    cols.into_iter()
                        .map(|s| s.istr())
                        .collect();
                for v in vars {
                    set.insert(v.istr());
                }
                Some(set)
            }
            (Some(cols), None) => Some(
                cols.into_iter()
                    .map(|s| s.istr())
                    .collect(),
            ),
            (None, Some(vars)) => Some(
                vars.into_iter()
                    .map(|s| s.istr())
                    .collect(),
            ),
            (None, None) => None,
        };

        let backend = self.inner.clone();

        future_into_py(py, async move {
            let df =
                scan_zarr_with_backend_async(
                    backend,
                    expr,
                    max_concurrency,
                    with_columns_set,
                    max_chunks_to_read,
                )
                .await?;

            let filtered = df
                .lazy()
                .filter(expr2)
                .collect()
                .map_err(|e| {
                    PyErr::new::<
                        pyo3::exceptions::PyRuntimeError,
                        _,
                    >(e.to_string())
                })?;

            Python::attach(|py| {
                Ok(PyDataFrame(filtered)
                    .into_pyobject(py)?
                    .unbind())
            })
        })
    }

    /// Get the schema for the zarr dataset.
    ///
    /// # Arguments
    /// * `variables` - Optional list of variable names to include
    #[pyo3(signature = (variables=None))]
    fn schema(
        &self,
        variables: Option<Vec<String>>,
    ) -> PyResult<PySchema> {
        use crate::IntoIStr;

        // Create a runtime to block on async metadata load
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| {
                PyErr::new::<
                    pyo3::exceptions::PyRuntimeError,
                    _,
                >(e.to_string())
            })?;

        let meta = runtime
            .block_on(self.inner.metadata())?;

        let vars: Option<Vec<crate::IStr>> =
            variables.map(|v| {
                v.into_iter()
                    .map(|s| s.istr())
                    .collect()
            });
        let schema =
            meta.tidy_schema(vars.as_deref());

        Ok(PySchema(Arc::new(schema)))
    }

    /// Debug function that returns per-variable chunk selections.
    ///
    /// Returns a dict with:
    /// - grids: List of grid info dicts, each containing:
    ///   - dims: List of dimension names
    ///   - variables: List of variable names in this grid
    ///   - chunks: List of chunk dicts with indices/origin/shape
    /// - coord_reads: Number of coordinate array reads performed
    #[pyo3(signature = (predicate))]
    fn selected_chunks_debug<'py>(
        &self,
        py: Python<'py>,
        predicate: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let backend = self.inner.clone();
        let expr = extract_expr(predicate)?;

        // Collect chunk info per grid
        #[derive(Clone)]
        struct ChunkInfo {
            indices: Vec<u64>,
            origin: Vec<u64>,
            shape: Vec<u64>,
        }

        #[derive(Clone)]
        struct GridInfo {
            dims: Vec<String>,
            variables: Vec<String>,
            chunks: Vec<ChunkInfo>,
        }

        // Create a runtime to block on async operations (same pattern as schema())
        let runtime =
            tokio::runtime::Runtime::new().map_err(
                |e| {
                    PyErr::new::<
                        pyo3::exceptions::PyRuntimeError,
                        _,
                    >(e.to_string())
                },
            )?;

        let (grids, coord_reads): (Vec<GridInfo>, u64) =
            runtime.block_on(async {
                // Compile expression to grouped chunk plan using backend-based resolver
                let (grouped_plan, stats) = backend
                    .clone()
                    .compile_expression_with_backend_async(&expr)
                    .await?;

                let mut grids: Vec<GridInfo> =
                    Vec::new();

                for (sig, vars, subsets, chunkgrid) in
                    grouped_plan.iter_grids()
                {
                    let dims: Vec<String> = sig
                        .dims()
                        .iter()
                        .map(|d| d.to_string())
                        .collect();
                    let variables: Vec<String> = vars
                        .iter()
                        .map(|v| v.to_string())
                        .collect();

                    let mut chunks: Vec<ChunkInfo> =
                        Vec::new();

                    for subset in subsets.subsets_iter()
                    {
                        let chunk_indices = chunkgrid
                            .chunks_in_array_subset(
                                subset,
                            )
                            .map_err(|e| {
                                PyErr::new::<
                                pyo3::exceptions::PyValueError,
                                _,
                            >(e.to_string())
                            })?;

                        if let Some(indices) =
                            chunk_indices
                        {
                            for idx in indices.indices()
                            {
                                let chunk_shape =
                                    sig.chunk_shape();
                                let origin = chunkgrid
                                    .chunk_origin(&idx)
                                    .map_err(|e| {
                                        PyErr::new::<
                                        pyo3::exceptions::PyValueError,
                                        _,
                                    >(e.to_string())
                                    })?
                                    .unwrap_or_else(
                                        || {
                                            vec![
                                            0;
                                            chunk_shape
                                                .len()
                                        ]
                                        },
                                    );

                                chunks.push(ChunkInfo {
                                    indices: idx
                                        .to_vec(),
                                    origin,
                                    shape: chunk_shape
                                        .to_vec(),
                                });
                            }
                        }
                    }

                    grids.push(GridInfo {
                        dims,
                        variables,
                        chunks,
                    });
                }

                Ok::<_, PyErr>((
                    grids,
                    stats.coord_reads,
                ))
            })?;

        // Convert to Python objects (with GIL held)
        let py_grids =
            pyo3::types::PyList::empty(py);

        for grid in grids {
            let grid_dict =
                pyo3::types::PyDict::new(py);
            grid_dict
                .set_item("dims", &grid.dims)?;
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
            grid_dict.set_item(
                "chunks",
                chunks_list,
            )?;

            py_grids.append(grid_dict)?;
        }

        let result = pyo3::types::PyDict::new(py);
        result.set_item("grids", py_grids)?;
        result.set_item(
            "coord_reads",
            coord_reads,
        )?;

        Ok(result.into_any().unbind())
    }

    // / Get the store root path.
    fn root(&self) -> String {
        return '/'.to_string();
    }

    /// Clear the coordinate cache.
    fn clear_coord_cache<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.inner.clone();
        future_into_py(py, async move {
            backend.clear().await;
            Ok(())
        })
    }

    /// Clear all caches (metadata and coordinates).
    fn clear_all_caches<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.inner.clone();
        future_into_py(py, async move {
            backend.clear().await;
            Ok(())
        })
    }

    /// Get cache statistics.
    fn cache_stats<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.inner.clone();
        future_into_py(py, async move {
            let stats =
                backend.cache_stats().await;
            Python::attach(|py| {
                let dict =
                    pyo3::types::PyDict::new(py);
                dict.set_item(
                    "coord_entries",
                    stats.chunk_entries,
                )?;
                dict.set_item(
                    "has_metadata",
                    stats.chunk_entries > 0,
                )?;
                Ok(dict.into_any().unbind())
            })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ZarrBackend(root='{}')",
            self.inner
        )
    }
}

/// Internal: Async scan using the backend.
///
/// This uses the backend's cached metadata and chunk reading directly.
async fn scan_zarr_with_backend_async(
    backend: Arc<FullyCachedZarrBackendAsync>,
    expr: polars::prelude::Expr,
    max_concurrency: Option<usize>,
    with_columns: Option<
        std::collections::BTreeSet<IStr>,
    >,
    max_chunks_to_read: Option<usize>,
) -> Result<polars::prelude::DataFrame, PyErr> {
    use futures::stream::{
        FuturesUnordered, StreamExt,
    };
    use pyo3_polars::error::PyPolarsErr;
    use std::sync::Arc as StdArc;

    const DEFAULT_MAX_CONCURRENCY: usize = 32;
    let meta = backend.metadata().await?;

    let planning_meta =
        StdArc::new(meta.planning_meta());

    // Expand struct column names to flat paths for chunk reading
    let expanded_with_columns = with_columns.as_ref().map(|cols| {
        crate::backend::lazy::expand_projection_to_flat_paths(cols, &meta)
    });

    // Compile grouped chunk plan using backend-based resolver
    let (grouped_plan, _stats) = backend
        .clone()
        .compile_expression_with_backend_async(
            &expr,
        )
        .await?;

    // Count total chunks to read if max_chunks_to_read is set
    if let Some(max_chunks) = max_chunks_to_read {
        let mut total_chunks = 0usize;
        for (_sig, _vars, subsets, chunkgrid) in
            grouped_plan.iter_grids()
        {
            for subset in subsets.subsets_iter() {
                if let Ok(Some(indices)) =
                    chunkgrid
                        .chunks_in_array_subset(
                            subset,
                        )
                {
                    total_chunks += indices
                        .num_elements_usize();
                }
            }
        }
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

    let max_conc = max_concurrency
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CONCURRENCY);
    let semaphore = StdArc::new(
        tokio::sync::Semaphore::new(max_conc),
    );

    // Wrap in Arc for sharing across tasks
    let expanded_with_columns =
        expanded_with_columns.map(StdArc::new);

    let mut futs = FuturesUnordered::new();
    for (sig, vars, subsets, chunkgrid) in
        grouped_plan.iter_grids()
    {
        let vars: Vec<IStr> = vars
            .into_iter()
            .map(|v| v.istr())
            .collect();
        let array_shape =
            chunkgrid.array_shape().to_vec();

        for subset in subsets.subsets_iter() {
            let chunk_indices = chunkgrid
                .chunks_in_array_subset(subset)
                .map_err(|e| {
                    PyErr::new::<
                        pyo3::exceptions::PyValueError,
                        _,
                    >(e.to_string())
                })?
                .ok_or(PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >("no chunks found"))?;

            for idx in chunk_indices.indices() {
                let permit = semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .unwrap();
                let backend = backend.clone();
                let sig = sig.clone();
                let array_shape =
                    array_shape.clone();
                let vars = vars.clone();
                let expanded_cols =
                    expanded_with_columns.clone();

                futs.push(async move {
                    let _permit = permit;
                    chunk_to_df_from_grid_with_backend(
                        backend.as_ref(),
                        idx.into(),
                        &sig,
                        &array_shape,
                        &vars,
                        expanded_cols
                            .as_ref()
                            .map(|a| a.as_ref()),
                    )
                    .await
                });
            }
        }
    }

    // Collect all chunk DataFrames
    let mut dfs: Vec<polars::prelude::DataFrame> =
        Vec::new();
    while let Some(r) = futs.next().await {
        let df = r?;
        dfs.push(df);
    }

    // Combine all chunk DataFrames
    let result = if dfs.is_empty() {
        let keys: Vec<IStr> = grouped_plan
            .var_to_grid()
            .keys()
            .cloned()
            .collect();
        polars::prelude::DataFrame::empty_with_schema(
            &planning_meta.tidy_schema(Some(
                keys.as_slice(),
            )),
        )
    } else if dfs.len() == 1 {
        dfs.into_iter().next().unwrap()
    } else {
        crate::backend::lazy::combine_chunk_dataframes(
            dfs, &meta,
        )?
    };

    // For hierarchical data, convert flat path columns to struct columns
    if meta.is_hierarchical() {
        crate::backend::lazy::restructure_to_structs(
            &result, &meta,
        )
    } else {
        Ok(result)
    }
}
