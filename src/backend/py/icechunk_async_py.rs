//! Python bindings for IcechunkBackend.
//!
//! Exposes the async-only Icechunk backend to Python with scan methods.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use icechunk::repository::VersionInfo;
use icechunk::session::Session;
use icechunk::storage::new_local_filesystem_storage;
use icechunk::{Repository, RepositoryConfig};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_polars::PySchema;
use zarrs::array::Array;

use crate::backend::implementation::{
    FullyCachedIcechunkBackendAsync,
    IcechunkBackendAsync,
    IcechunkIterator,
    to_fully_cached_icechunk_async,
};
use crate::meta::ZarrMeta;
use crate::py::expr_extract::extract_expr;
use crate::scan::async_scan::chunk_to_df_from_grid_with_backend;
use crate::shared::ChunkedExpressionCompilerAsync;
use crate::shared::normalize_path;
use crate::shared::{
    ChunkedDataBackendAsync,
    EvictableChunkCacheAsync, HasAsyncStore,
    HasMetadataBackendAsync,
};
use crate::{IStr, IntoIStr};

use crate::shared::{
    combine_chunk_dataframes,
    restructure_to_structs,
};

/// Extract session bytes from a Python session object.
///
/// Supports:
/// - icechunk-python Session: extracts via `._session.as_bytes()`
/// - rainbear PySession: extracts via `.as_bytes()`
/// - Raw bytes: uses directly
fn extract_session_bytes(
    session: &Bound<'_, PyAny>,
) -> PyResult<Vec<u8>> {
    // Try 1: Check if it's raw bytes
    if let Ok(bytes) =
        session.extract::<Vec<u8>>()
    {
        return Ok(bytes);
    }

    // Try 2: icechunk-python Session - access `._session.as_bytes()`
    if let Ok(inner_session) =
        session.getattr("_session")
    {
        if let Ok(as_bytes_method) =
            inner_session.getattr("as_bytes")
        {
            if let Ok(bytes_obj) =
                as_bytes_method.call0()
            {
                if let Ok(bytes) =
                    bytes_obj.extract::<Vec<u8>>()
                {
                    return Ok(bytes);
                }
            }
        }
    }

    // Try 3: Object has .as_bytes() method directly (e.g., PySession from icechunk internal)
    if let Ok(as_bytes_method) =
        session.getattr("as_bytes")
    {
        if let Ok(bytes_obj) =
            as_bytes_method.call0()
        {
            if let Ok(bytes) =
                bytes_obj.extract::<Vec<u8>>()
            {
                return Ok(bytes);
            }
        }
    }

    Err(PyErr::new::<
        pyo3::exceptions::PyTypeError,
        _,
    >(
        "Expected an icechunk Session, rainbear Session, or bytes. \
         Could not extract session bytes from the provided object.",
    ))
}

/// Python-exposed Icechunk backend with caching and scan methods.
///
/// The backend owns the Icechunk session and caches coordinate array chunks
/// and metadata across multiple scan operations.
#[pyclass(name = "IcechunkBackend")]
pub struct PyIcechunkBackend {
    inner: Arc<FullyCachedIcechunkBackendAsync>,
}

#[pymethods]
impl PyIcechunkBackend {
    /// Create a backend from a filesystem path to an Icechunk repository.
    ///
    /// Opens a readonly session on the specified branch.
    ///
    /// # Arguments
    /// * `path` - Path to the Icechunk repository
    /// * `branch` - Branch name to read from (default: "main")
    /// * `root` - Optional root path within the store (default: "/")
    ///
    /// # Example
    /// ```python
    /// backend = await IcechunkBackend.from_filesystem("/path/to/icechunk/repo")
    /// ```
    #[staticmethod]
    #[pyo3(signature = (path, branch=None, root=None))]
    fn from_filesystem<'py>(
        py: Python<'py>,
        path: String,
        branch: Option<String>,
        root: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let branch =
            branch.unwrap_or_else(|| {
                "main".to_string()
            });
        let root_clone = root.clone();

        future_into_py(py, async move {
            // Create filesystem storage
            let path = Path::new(&path);
            let storage = new_local_filesystem_storage(path)
                .await
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to create local storage: {}",
                        e
                    ))
                })?;

            // Open repository
            let repo = Repository::open(
                Some(RepositoryConfig::default()),
                storage,
                HashMap::new(),
            )
            .await
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to open repository: {}",
                    e
                ))
            })?;

            // Open readonly session
            let version =
                VersionInfo::BranchTipRef(branch);
            let session = repo.readonly_session(&version).await.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to open session: {}",
                    e
                ))
            })?;

            let backend =
                IcechunkBackendAsync::from_session(session, root_clone);
            let backend =
                to_fully_cached_icechunk_async(
                    backend, 20,
                )?;

            Python::attach(|py| {
                let py_backend =
                    PyIcechunkBackend {
                        inner: Arc::new(backend),
                    };
                Ok(Py::new(py, py_backend)?
                    .into_any())
            })
        })
    }

    /// Create a backend from an Icechunk session.
    ///
    /// Accepts either:
    /// - An icechunk-python Session object directly
    /// - A rainbear Session object  
    /// - Raw bytes from session serialization
    ///
    /// # Arguments
    /// * `session` - An icechunk Session object (from icechunk-python or rainbear)
    /// * `root` - Optional root path within the store (default: "/")
    ///
    /// # Example
    /// ```python
    /// from icechunk import Repository, local_filesystem_storage
    /// import rainbear
    ///
    /// # Get session from icechunk-python
    /// storage = local_filesystem_storage("/path/to/repo")
    /// repo = Repository.open(storage)
    /// session = repo.readonly_session("main")
    ///
    /// # Directly create backend from icechunk-python session
    /// backend = await rainbear.IcechunkBackend.from_session(session)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (session, root=None))]
    fn from_session<'py>(
        py: Python<'py>,
        session: &Bound<'py, PyAny>,
        root: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let root_clone = root.clone();

        // Extract session bytes from the Python object
        let session_bytes =
            extract_session_bytes(session)?;

        // Deserialize to Rust Session
        let inner_session = Session::from_bytes(session_bytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to deserialize session: {}",
                e
            ))
        })?;

        future_into_py(py, async move {
            let backend =
                IcechunkBackendAsync::from_session(inner_session, root_clone);
            let backend =
                to_fully_cached_icechunk_async(
                    backend, 20,
                )?;

            Python::attach(|py| {
                let py_backend =
                    PyIcechunkBackend {
                        inner: Arc::new(backend),
                    };
                Ok(Py::new(py, py_backend)?
                    .into_any())
            })
        })
    }

    /// Async scan the Icechunk store and return a DataFrame.
    ///
    /// Uses the backend's cached coordinates for efficient predicate pushdown.
    ///
    /// # Arguments
    /// * `predicate` - Polars expression for filtering
    /// * `max_concurrency` - Maximum concurrent chunk reads
    /// * `max_chunks_to_read` - Maximum number of chunks to read (safety limit)
    #[pyo3(signature = (predicate, max_concurrency=None, max_chunks_to_read=None))]
    fn scan_zarr_async<'py>(
        &self,
        py: Python<'py>,
        predicate: &Bound<'_, PyAny>,
        max_concurrency: Option<usize>,
        max_chunks_to_read: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use polars::prelude::IntoLazy;
        use pyo3_polars::PyDataFrame;

        let expr = extract_expr(predicate)?;
        let expr2 = expr.clone();

        let backend = self.inner.clone();

        future_into_py(py, async move {
            let df = scan_with_backend_async(
                backend,
                expr,
                max_concurrency,
                max_chunks_to_read,
            )
            .await?;

            // `predicate` is used for chunk planning but may also include projection,
            // e.g. `pl.col(["y","x","var"]).filter(pred)`.
            let lf = df.lazy();
            let filtered =                         lf
                            .select([expr2])
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

    /// Streaming scan the Icechunk store and return an iterator over DataFrames.
    ///
    /// Blocks on async I/O internally, using tokio concurrency for chunk reads.
    /// Enables memory-efficient streaming when scanning time-chunked data
    /// (e.g., a single point across a year).
    ///
    /// # Arguments
    /// * `predicate` - Polars expression for filtering
    /// * `with_columns` - Optional list of columns to include
    /// * `max_chunks_to_read` - Maximum number of chunks to read (safety limit)
    /// * `n_rows` - Number of rows to read total
    /// * `batch_size` - Batch size for reading
    /// * `max_concurrency` - Maximum concurrent chunk reads per batch
    #[pyo3(signature = (predicate=None, with_columns=None, max_chunks_to_read=None, n_rows=None, batch_size=None, max_concurrency=None))]
    fn scan_zarr_streaming_sync<'py>(
        &self,
        py: Python<'py>,
        predicate: Option<&Bound<'_, PyAny>>,
        with_columns: Option<Vec<String>>,
        max_chunks_to_read: Option<usize>,
        n_rows: Option<usize>,
        batch_size: Option<usize>,
        max_concurrency: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use polars::prelude::lit;
        use pyo3::IntoPyObjectExt;

        let prd = if let Some(predicate) = predicate {
            extract_expr(predicate)?
        } else {
            lit(true)
        };
        let with_cols_set: Option<BTreeSet<IStr>> =
            with_columns.map(|cols| cols.into_iter().map(|c| c.istr()).collect());

        IcechunkIterator::new(
            self.inner.clone(),
            prd,
            with_cols_set,
            max_chunks_to_read,
            n_rows,
            batch_size,
            max_concurrency,
        )?
        .into_bound_py_any(py)
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
        // Create a runtime to block on async metadata load
        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        let meta = runtime
            .block_on(self.inner.metadata())?;

        let vars: Option<Vec<IStr>> = variables
            .map(|v| {
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

        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        let (grids, coord_reads): (Vec<GridInfo>, u64) =
            runtime.block_on(async {
                let (grouped_plan, stats) = backend
                    .clone()
                    .compile_expression_async(&expr)
                    .await?;

                let mut grids: Vec<GridInfo> = Vec::new();

                for (sig, vars, subsets, chunkgrid) in
                    grouped_plan.iter_grids()
                {
                    let dims: Vec<String> =
                        sig.dims().iter().map(|d| d.to_string()).collect();
                    let variables: Vec<String> =
                        vars.iter().map(|v| v.to_string()).collect();

                    // For sharded arrays, the planner produces *inner* chunk indices
                    // but remote access happens at the shard-file level.
                    let inner_chunk_shape: Vec<u64> =
                        sig.chunk_shape().to_vec();

                    let mut outer_chunk_shape: Option<Vec<u64>> = None;
                    let mut array_shape: Option<Vec<u64>> = None;
                    if let Some(var) = vars.first() {
                        if let Ok(arr) = Array::async_open(
                            backend.async_store().clone(),
                            &normalize_path(*var),
                        )
                        .await
                        {
                            let zero =
                                vec![0u64; arr.dimensionality()];
                            outer_chunk_shape = arr
                                .chunk_grid()
                                .chunk_shape_u64(&zero)
                                .ok()
                                .flatten();
                            array_shape = Some(arr.shape().to_vec());
                        }
                    }

                    let is_sharded = outer_chunk_shape
                        .as_ref()
                        .is_some_and(|outer| outer != &inner_chunk_shape);

                    let mut chunks: Vec<ChunkInfo> = Vec::new();
                    let mut seen_outer: BTreeSet<Vec<u64>> = BTreeSet::new();

                    for subset in subsets.subsets_iter() {
                        let chunk_indices = chunkgrid
                            .chunks_in_array_subset(subset)
                            .map_err(|e| {
                                PyErr::new::<
                                    pyo3::exceptions::PyValueError,
                                    _,
                                >(e.to_string())
                            })?;

                        if let Some(indices) = chunk_indices {
                            for idx in indices.indices() {
                                let inner_idx = idx.to_vec();

                                if is_sharded {
                                    let Some(outer_shape) =
                                        outer_chunk_shape.as_ref()
                                    else {
                                        continue;
                                    };
                                    let outer_idx: Vec<u64> =
                                        inner_idx
                                            .iter()
                                            .zip(inner_chunk_shape.iter())
                                            .zip(outer_shape.iter())
                                            .map(
                                                |((&i, &inner_sz), &outer_sz)| {
                                                    if outer_sz == 0 {
                                                        0
                                                    } else {
                                                        i.saturating_mul(inner_sz)
                                                            / outer_sz
                                                    }
                                                },
                                            )
                                            .collect();

                                    if !seen_outer.insert(outer_idx.clone()) {
                                        continue;
                                    }

                                    let origin: Vec<u64> = outer_idx
                                        .iter()
                                        .zip(outer_shape.iter())
                                        .map(|(&i, &sz)| i.saturating_mul(sz))
                                        .collect();

                                    let shape: Vec<u64> = match array_shape.as_ref() {
                                        Some(a_shape) if a_shape.len() == outer_shape.len() => {
                                            origin
                                                .iter()
                                                .zip(a_shape.iter())
                                                .zip(outer_shape.iter())
                                                .map(|((&o, &a), &s)| {
                                                    if o >= a {
                                                        0
                                                    } else {
                                                        (a - o).min(s)
                                                    }
                                                })
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
                                    let chunk_shape = sig.chunk_shape();
                                    let origin = chunkgrid
                                        .chunk_origin(&idx)
                                        .map_err(|e| {
                                            PyErr::new::<
                                                pyo3::exceptions::PyValueError,
                                                _,
                                            >(e.to_string())
                                        })?
                                        .unwrap_or_else(|| {
                                            vec![0; chunk_shape.len()]
                                        });

                                    chunks.push(ChunkInfo {
                                        indices: inner_idx,
                                        origin,
                                        shape: chunk_shape.to_vec(),
                                    });
                                }
                            }
                        }
                    }

                    grids.push(GridInfo {
                        dims,
                        variables,
                        chunks,
                    });
                }

                Ok::<_, PyErr>((grids, stats.coord_reads))
            })?;

        // Convert to Python objects
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

    /// Get the store root path.
    fn root(&self) -> String {
        "/".to_string()
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
            backend.clear_all_caches().await;
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
            let has_metadata = backend
                .has_metadata_cached()
                .await;
            Python::attach(|py| {
                let dict =
                    pyo3::types::PyDict::new(py);
                dict.set_item(
                    "coord_entries",
                    stats.chunk_entries,
                )?;
                dict.set_item(
                    "has_metadata",
                    has_metadata,
                )?;
                Ok(dict.into_any().unbind())
            })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "IcechunkBackend(root='{}')",
            self.inner
        )
    }
}

/// Internal: Async scan using any backend that implements the required traits.
async fn scan_with_backend_async<B>(
    backend: Arc<B>,
    expr: polars::prelude::Expr,
    max_concurrency: Option<usize>,
    max_chunks_to_read: Option<usize>,
) -> Result<polars::prelude::DataFrame, PyErr>
where
    B: ChunkedDataBackendAsync
        + HasMetadataBackendAsync<ZarrMeta>
        + ChunkedExpressionCompilerAsync
        + Send
        + Sync
        + 'static,
{
    use futures::stream::{
        FuturesUnordered, StreamExt,
    };
    use std::sync::Arc as StdArc;

    const DEFAULT_MAX_CONCURRENCY: usize = 32;
    let meta = backend.metadata().await?;

    let planning_meta =
        StdArc::new(meta.planning_meta());

    // // Expand struct column names to flat paths for chunk reading
    // let expanded_with_columns =
    //     with_columns.as_ref().map(|cols| {
    //         expand_projection_to_flat_paths(
    //             cols, &meta,
    //         )
    //     });

    // Compile grouped chunk plan using backend-based resolver
    let (grouped_plan, _stats) = backend
        .clone()
        .compile_expression_async(&expr)
        .await?;

    // Check max_chunks_to_read limit before doing any I/O
    if let Some(max_chunks) = max_chunks_to_read {
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

    let max_conc = max_concurrency
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CONCURRENCY);
    let semaphore = StdArc::new(
        tokio::sync::Semaphore::new(max_conc),
    );

    // Read chunks using consolidated (deduplicated) iteration
    let mut futs = FuturesUnordered::new();
    for group in
        grouped_plan.iter_consolidated_chunks()
    {
        let group = group.map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e)
        })?;
        let vars: Vec<IStr> = group
            .vars
            .iter()
            .map(|v| v.istr())
            .collect();

        for (idx, subset) in group
            .chunk_indices
            .into_iter()
            .zip(group.chunk_subsets)
        {
            let sem = semaphore.clone();
            let backend = backend.clone();
            let sig = group.sig.clone();
            let array_shape =
                group.array_shape.clone();
            let vars = vars.clone();

            futs.push(async move {
                let _permit = sem
                    .acquire_owned()
                    .await
                    .expect("semaphore closed");
                chunk_to_df_from_grid_with_backend(
                    backend.as_ref(),
                    idx.into(),
                    &sig,
                    &array_shape,
                    &vars,
                    None,
                    subset.as_ref(),
                )
                .await
            });
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
            &planning_meta.tidy_schema(Some(keys.as_slice())),
        )
    } else if dfs.len() == 1 {
        dfs.into_iter().next().unwrap()
    } else {
        combine_chunk_dataframes(dfs, &meta)?
    };

    // For hierarchical data, convert flat path columns to struct columns
    if meta.is_hierarchical() {
        restructure_to_structs(&result, &meta)
    } else {
        Ok(result)
    }
}
