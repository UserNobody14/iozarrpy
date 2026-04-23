//! Python binding for [`BackendOptions`].

use pyo3::prelude::*;

use crate::shared::BackendOptions;

/// Python-exposed bundle of backend construction options.
///
/// All knobs are keyword-only and have sensible defaults; pass an
/// instance as `options=` to any backend constructor (`from_url`,
/// `from_store`, `from_filesystem`, `from_session`).
#[pyclass(name = "BackendOptions", frozen)]
#[derive(Clone, Copy)]
pub struct PyBackendOptions {
    pub inner: BackendOptions,
}

impl PyBackendOptions {
    /// Resolve an `Option<&PyBackendOptions>` to the inner Rust struct,
    /// falling back to [`BackendOptions::default`].
    pub fn resolve(
        opt: Option<&PyBackendOptions>,
    ) -> BackendOptions {
        opt.map(|o| o.inner).unwrap_or_default()
    }
}

#[pymethods]
impl PyBackendOptions {
    #[new]
    #[pyo3(signature = (
        *,
        coord_cache_max_entries = BackendOptions::default().coord_cache_max_entries,
        var_cache_max_entries = BackendOptions::default().var_cache_max_entries,
    ))]
    fn new(
        coord_cache_max_entries: u64,
        var_cache_max_entries: u64,
    ) -> Self {
        Self {
            inner: BackendOptions {
                coord_cache_max_entries,
                var_cache_max_entries,
            },
        }
    }

    #[getter]
    fn coord_cache_max_entries(&self) -> u64 {
        self.inner.coord_cache_max_entries
    }

    #[getter]
    fn var_cache_max_entries(&self) -> u64 {
        self.inner.var_cache_max_entries
    }

    fn __repr__(&self) -> String {
        format!(
            "BackendOptions(coord_cache_max_entries={}, var_cache_max_entries={})",
            self.inner.coord_cache_max_entries,
            self.inner.var_cache_max_entries,
        )
    }
}
