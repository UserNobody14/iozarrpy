//! Python bindings for ZarrBackend.
//!
//! Exposes the caching backend to Python with scan methods.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_polars::PySchema;

use super::caching::CachingAsyncBackend;
use super::traits::ZarrBackendAsync;
use crate::py::expr_extract::extract_expr;
use crate::store::StoreInput;

/// Python-exposed Zarr backend with caching and scan methods.
///
/// The backend owns the store and caches coordinate array chunks and metadata
/// across multiple scan operations.
#[pyclass(name = "ZarrBackend")]
pub struct PyZarrBackend {
    inner: Arc<CachingAsyncBackend>,
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
        let opened = crate::store::open_store_async(&url)
            .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                e,
            )
        })?;

        let backend = CachingAsyncBackend::new(
            opened.store,
            opened.root,
            max_cache_entries,
        );

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
        let opened =
            store_input.open_async().map_err(|e| {
                PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >(e)
            })?;

        let backend = CachingAsyncBackend::new(
            opened.store,
            opened.root,
            max_cache_entries,
        );

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
    #[pyo3(signature = (predicate, variables=None, max_concurrency=None, with_columns=None))]
    fn scan_zarr_async<'py>(
        &self,
        py: Python<'py>,
        predicate: &Bound<'_, PyAny>,
        variables: Option<Vec<String>>,
        max_concurrency: Option<usize>,
        with_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use polars::prelude::IntoLazy;
        use pyo3_polars::PyDataFrame;
        use std::collections::BTreeSet;

        let expr = extract_expr(predicate)?;
        let expr2 = expr.clone();

        let with_columns_set: Option<
            BTreeSet<String>,
        > = with_columns.map(|v| {
            v.into_iter().collect::<BTreeSet<_>>()
        });

        let backend = self.inner.clone();

        future_into_py(py, async move {
            let df =
                scan_zarr_with_backend_async(
                    backend,
                    expr,
                    variables,
                    max_concurrency,
                    with_columns_set,
                )
                .await?;

            let filtered =
                df.lazy().filter(expr2).collect().map_err(
                    |e| {
                        PyErr::new::<
                        pyo3::exceptions::PyRuntimeError,
                        _,
                    >(e.to_string())
                    },
                )?;

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
            .block_on(self.inner.load_metadata())
            .map_err(|e| {
                PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >(e.to_string())
            })?;

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

    /// Get the store root path.
    fn root(&self) -> String {
        self.inner.root().to_string()
    }

    /// Clear the coordinate cache.
    fn clear_coord_cache<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.inner.clone();
        future_into_py(py, async move {
            backend.clear_coord_cache().await;
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
            Python::attach(|py| {
                let dict =
                    pyo3::types::PyDict::new(py);
                dict.set_item(
                    "coord_entries",
                    stats.coord_entries,
                )?;
                dict.set_item(
                    "has_metadata",
                    stats.has_metadata,
                )?;
                Ok(dict.into_any().unbind())
            })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ZarrBackend(root='{}')",
            self.inner.root()
        )
    }
}

/// Internal: Async scan using the backend.
///
/// This reuses the existing scan infrastructure but uses the backend's cached metadata.
async fn scan_zarr_with_backend_async(
    backend: Arc<CachingAsyncBackend>,
    expr: polars::prelude::Expr,
    variables: Option<Vec<String>>,
    max_concurrency: Option<usize>,
    with_columns: Option<
        std::collections::BTreeSet<String>,
    >,
) -> Result<polars::prelude::DataFrame, PyErr> {
    use crate::IntoIStr;
    use crate::chunk_plan::compile_expr_to_grouped_chunk_plan_async;
    use crate::scan::chunk_to_df::chunk_to_df;
    use crate::scan::open_arrays::open_arrays_async;
    use futures::stream::{
        FuturesUnordered, StreamExt,
    };
    use pyo3_polars::error::PyPolarsErr;
    use std::collections::BTreeSet;
    use std::sync::Arc as StdArc;

    const DEFAULT_MAX_CONCURRENCY: usize = 32;

    // Load metadata from cache
    let meta = backend
        .load_metadata()
        .await
        .map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e.to_string())
        })?;

    // Convert variables to IStr
    let vars: Vec<crate::IStr> = variables
        .map(|v| {
            v.into_iter()
                .map(|s| s.istr())
                .collect()
        })
        .unwrap_or_else(|| {
            meta.data_vars.clone()
        });

    if vars.is_empty() {
        return Err(PyErr::new::<
            pyo3::exceptions::PyValueError,
            _,
        >(
            "no variables found/selected",
        ));
    }

    // Use dataset dims directly
    let dims = meta.dims.clone();

    let store = backend.async_store();

    // Open arrays for reading
    let (var_arrays, coord_arrays) =
        open_arrays_async(
            store.clone(),
            &meta,
            &vars,
            &dims,
        )
        .await
        .map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e)
        })?;

    // Pick a reference variable for chunk iteration geometry
    // TODO: Eventually iterate per-grid, but for now pick the first variable
    let ref_var = &vars[0];
    let ref_meta = meta
        .arrays
        .get(ref_var)
        .ok_or_else(|| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >("unknown variable")
        })?;
    let ref_array =
        zarrs::array::Array::async_open(
            store.clone(),
            ref_meta.path.as_ref(),
        )
        .await
        .map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e.to_string())
        })?;
    let ref_array = StdArc::new(ref_array);

    // Compile grouped chunk plan
    let (grouped_plan, _stats) =
        match compile_expr_to_grouped_chunk_plan_async(
            &expr,
            &meta,
            store.clone(),
        )
        .await
        {
            Ok(x) => x,
            Err(_) => {
                // Fall back to empty plan
                (
                crate::chunk_plan::GroupedChunkPlan::new(),
                crate::chunk_plan::PlannerStats { coord_reads: 0 },
            )
            }
        };

    // Convert array subsets to chunk indices for the reference variable
    let mut chunk_indices: Vec<Vec<u64>> =
        Vec::new();

    if let Some(subsets) =
        grouped_plan.get_plan(ref_var.as_ref())
    {
        for subset in subsets.subsets_iter() {
            if let Ok(Some(chunks)) = ref_array
                .chunks_in_array_subset(subset)
            {
                for chunk_idx in
                    chunks.indices().iter()
                {
                    chunk_indices.push(
                        chunk_idx
                            .iter()
                            .copied()
                            .collect(),
                    );
                }
            }
        }
    } else if grouped_plan.is_empty() {
        // No selection made - scan all chunks
        let grid_shape =
            ref_array.chunk_grid().grid_shape();
        let mut idx =
            vec![0u64; grid_shape.len()];
        loop {
            chunk_indices.push(idx.clone());
            let mut carry = true;
            for d in (0..idx.len()).rev() {
                if carry {
                    idx[d] += 1;
                    if idx[d] < grid_shape[d] {
                        carry = false;
                    } else {
                        idx[d] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
    }

    let max_conc = max_concurrency
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CONCURRENCY);
    let semaphore = StdArc::new(
        tokio::sync::Semaphore::new(max_conc),
    );

    let meta = StdArc::new((*meta).clone());
    let dims = StdArc::new(dims);
    let vars = StdArc::new(vars);
    let var_arrays = StdArc::new(var_arrays);
    let coord_arrays = StdArc::new(coord_arrays);
    let with_columns: StdArc<
        Option<BTreeSet<crate::IStr>>,
    > = StdArc::new(with_columns.map(|s| {
        s.into_iter().map(|c| c.istr()).collect()
    }));

    let mut futs = FuturesUnordered::new();
    for idx in chunk_indices {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();
        let ref_array = ref_array.clone();
        let meta = StdArc::clone(&meta);
        let dims = StdArc::clone(&dims);
        let vars = StdArc::clone(&vars);
        let var_arrays =
            StdArc::clone(&var_arrays);
        let coord_arrays =
            StdArc::clone(&coord_arrays);
        let with_columns =
            StdArc::clone(&with_columns);
        futs.push(async move {
            let _permit = permit;
            chunk_to_df(
                idx,
                ref_array,
                meta,
                dims,
                vars,
                var_arrays,
                coord_arrays,
                with_columns,
            )
            .await
        });
    }

    let mut out: Option<
        polars::prelude::DataFrame,
    > = None;
    while let Some(r) = futs.next().await {
        let df = r?;
        if let Some(acc) = &mut out {
            acc.vstack_mut(&df)
                .map_err(PyPolarsErr::from)?;
        } else {
            out = Some(df);
        }
    }

    Ok(out.unwrap_or_else(|| {
        polars::prelude::DataFrame::empty_with_schema(
            &meta.tidy_schema(Some(&vars)),
        )
    }))
}
