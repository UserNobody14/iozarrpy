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

use crate::backend::implementation::{
    FullyCachedIcechunkBackendAsync,
    IcechunkBackendAsync, IcechunkIterator,
    scan_with_backend_async,
    to_fully_cached_icechunk_async,
};
use crate::backend::py::debug::extract_grids;
use crate::backend::py::debug::grids_to_python;
use crate::py::expr_extract::extract_expr;
use crate::shared::{
    EvictableChunkCacheAsync,
    HasMetadataBackendAsync,
};
use crate::{IStr, IntoIStr};

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

        let prd =
            if let Some(predicate) = predicate {
                extract_expr(predicate)?
            } else {
                lit(true)
            };
        let with_cols_set: Option<
            BTreeSet<IStr>,
        > = with_columns.map(|cols| {
            cols.into_iter()
                .map(|c| c.istr())
                .collect()
        });

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

        // Create a runtime to block on async operations (same pattern as schema())
        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        let (grids, coord_reads) = runtime
            .block_on(extract_grids(
                backend, expr,
            ))?;

        grids_to_python(py, grids, coord_reads)
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
