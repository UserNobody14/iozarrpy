//! Python bindings for ZarrBackend.
//!
//! Exposes the caching backend to Python with scan methods.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_polars::PySchema;
use pyo3_polars::error::PyPolarsErr;
use snafu::ResultExt;
use std::collections::BTreeSet;
use zarrs::array::Array;

use crate::backend::implementation::scan_zarr_with_backend_async;
use crate::errors::IncompatibleDimensionalitySnafu;
use crate::py::expr_extract::extract_expr;
use crate::shared::ChunkedExpressionCompilerAsync;
use crate::shared::{
    EvictableChunkCacheAsync,
    FullyCachedZarrBackendAsync, HasAsyncStore,
    HasMetadataBackendAsync, ZarrBackendAsync,
    to_fully_cached_async,
};
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
    #[pyo3(signature = (url, max_cache_entries=5))]
    fn from_url(
        url: String,
        max_cache_entries: u64,
    ) -> PyResult<Self> {
        let backend = ZarrBackendAsync::new(
            StoreInput::Url(url),
        )?;

        let backend = to_fully_cached_async(
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
            ZarrBackendAsync::new(store_input)?;
        let backend = to_fully_cached_async(
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
            let df =
                scan_zarr_with_backend_async(
                    backend,
                    expr,
                    max_concurrency,
                    max_chunks_to_read,
                )
                .await?;

            // `predicate` is used for chunk planning but may also include projection,
            // e.g. `pl.col(["y","x","var"]).filter(pred)`.
            let lf = df.lazy();
            let filtered = lf
                .select([expr2])
                .collect()
                .map_err(PyPolarsErr::from)?;

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
                    .compile_expression_async(&expr)
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

                    // For sharded arrays, the planner produces *inner* chunk indices
                    // but remote access happens at the shard-file level (one remote file
                    // can contain many inner chunks). For debug output we want to reflect
                    // which remote shard files are touched.
                    let inner_chunk_shape: Vec<u64> =
                        sig.chunk_shape().to_vec();

                    // Try to open one representative array to detect sharding and obtain
                    // the shard (outer chunk) shape and array shape.
                    let store = backend.async_store().clone();
                    let mut outer_chunk_shape: Option<Vec<u64>> = None;
                    let mut array_shape: Option<Vec<u64>> = None;
                    if let Some(var) = vars.first() {
                        let path =
                            crate::shared::normalize_path(var);
                        if let Ok(arr) = Array::async_open(
                            store.clone(),
                            &path,
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

                    for subset in subsets.subsets_iter()
                    {
                        let chunk_indices = chunkgrid
                            .chunks_in_array_subset(
                                subset,
                            ).context(
                                IncompatibleDimensionalitySnafu {
                                    dims: sig.dims().to_vec(),
                                    shape: chunkgrid.array_shape().to_vec(),
                                    paths: vars.iter().cloned().collect::<Vec<IStr>>(),
                                }
                            )?;

                        if let Some(indices) =
                            chunk_indices
                        {
                            for idx in indices.indices()
                            {
                                let inner_idx = idx.to_vec();

                                if is_sharded {
                                    let Some(outer_shape) =
                                        outer_chunk_shape.as_ref()
                                    else {
                                        continue;
                                    };
                                    // Map inner chunk index -> shard index
                                    let outer_idx: Vec<u64> =
                                        inner_idx
                                            .iter()
                                            .zip(
                                                inner_chunk_shape.iter(),
                                            )
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

                                    // Compute actual shard shape at boundaries
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
            "ZarrBackend(root='{}')",
            self.inner
        )
    }
}
