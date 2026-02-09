//! Python bindings for ZarrBackendSync.
//!
//! Exposes the sync caching backend to Python with scan methods.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_polars::{PyDataFrame, PySchema};

use crate::IntoIStr;
use crate::backend::implementation::scan_zarr_with_backend_sync;
use crate::py::expr_extract::extract_expr;
use crate::shared::{
    EvictableChunkCacheSync,
    HasMetadataBackendSync,
};
use crate::shared::{
    FullyCachedZarrBackendSync, ZarrBackendSync,
    to_fully_cached_sync,
};
use crate::store::StoreInput;

/// Python-exposed Zarr backend with caching and scan methods.
///
/// The backend owns the store and caches coordinate array chunks and metadata
/// across multiple scan operations.
#[pyclass(name = "ZarrBackendSync")]
pub struct PyZarrBackendSync {
    inner: Arc<FullyCachedZarrBackendSync>,
}

#[pymethods]
impl PyZarrBackendSync {
    /// Create a backend from a URL string.
    ///
    /// # Arguments
    /// * `url` - URL to the zarr store (e.g., "s3://bucket/path.zarr")
    /// * `max_cache_entries` - Maximum cached coord chunks (0 = unlimited)
    #[staticmethod]
    #[pyo3(signature = (url, max_cache_entries=5))]
    fn from_url(
        url: String,
        max_cache_entries: u64,
    ) -> PyResult<Self> {
        let backend = ZarrBackendSync::new(
            StoreInput::Url(url),
        )?;

        let backend = to_fully_cached_sync(
            backend,
            max_cache_entries,
        )?;
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
    #[pyo3(signature = (store, prefix=None, max_cache_entries=5))]
    fn from_store(
        store: &Bound<'_, PyAny>,
        prefix: Option<String>,
        max_cache_entries: u64,
    ) -> PyResult<Self> {
        let store_input =
            StoreInput::from_py(store, prefix)?;
        let backend =
            ZarrBackendSync::new(store_input)?;
        let backend = to_fully_cached_sync(
            backend,
            max_cache_entries,
        )?;
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
    /// * `n_rows` - Maximum number of rows to read
    /// * `batch_size` - Batch size for reading
    #[pyo3(signature = (predicate=None, with_columns=None, max_chunks_to_read=None, n_rows=None, batch_size=None))]
    fn scan_zarr_sync<'py>(
        &self,
        py: Python<'py>,
        predicate: Option<&Bound<'_, PyAny>>,
        with_columns: Option<Vec<String>>,
        max_chunks_to_read: Option<usize>,
        n_rows: Option<usize>,
        batch_size: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let prd =
            if let Some(predicate) = predicate {
                extract_expr(predicate)?
            } else {
                // Filter that will pass all chunks
                // Use a filter expression that evaluates to true for all rows.
                lit(true)
            };
        let with_cols_set: Option<
            BTreeSet<crate::IStr>,
        > = with_columns.map(|cols| {
            cols.into_iter()
                .map(|c| c.istr())
                .collect()
        });
        let df = scan_zarr_with_backend_sync(
            &self.inner,
            prd.clone(),
            with_cols_set,
            max_chunks_to_read,
        )?;

        let filtered = df
            .lazy()
            .filter(prd)
            .collect()
            .map_err(|e| {
                PyErr::new::<
                    pyo3::exceptions::PyRuntimeError,
                    _,
                >(e.to_string())
            })?;
        Ok(PyDataFrame(filtered)
            .into_pyobject(py)
            .map_err(|e| {
                PyErr::new::<
                    pyo3::exceptions::PyRuntimeError,
                    _,
                >(e.to_string())
            })?
            .into_any())
        // if let Some(variables) = variables {
        //     asc =
        //         asc.select(Expr::Selector(Selector::ByName {
        //             names: variables
        //                 .into_iter()
        //                 .map(|s| s.istr())
        //                 .collect::<Vec<_>>(),
        //         })?;
        // }
    }
    /// # Arguments
    /// * `variables` - Optional list of variable names to include
    #[pyo3(signature = (variables=None))]
    fn schema(
        &self,
        variables: Option<Vec<String>>,
    ) -> PyResult<PySchema> {
        use crate::IntoIStr;

        let meta = self.inner.metadata()?;

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
            backend.clear();
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
            backend.clear_all_caches();
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
            let stats = backend.cache_stats();
            let has_metadata =
                backend.has_metadata_cached();
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
            "ZarrBackend(root='{}')",
            self.inner
        )
    }
}
