//! Python binding for [`BackendOptions`].

use std::collections::HashSet;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};

use crate::shared::{
    AssumeSortedDims, BackendOptions, IntoIStr,
};

/// Python-exposed bundle of backend construction options.
///
/// All knobs are keyword-only and have sensible defaults; pass an
/// instance as `options=` to any backend constructor (`from_url`,
/// `from_store`, `from_filesystem`, `from_session`).
#[pyclass(
    name = "BackendOptions",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyBackendOptions {
    pub inner: BackendOptions,
}

impl PyBackendOptions {
    /// Resolve an `Option<PyBackendOptions>` to the inner Rust struct,
    /// falling back to [`BackendOptions::default`]. Consumes the wrapper
    /// so callers' `Option<PyBackendOptions>` ABI args don't trip
    /// `clippy::needless_pass_by_value`.
    pub fn resolve(
        opt: Option<PyBackendOptions>,
    ) -> BackendOptions {
        opt.map(|o| o.inner).unwrap_or_default()
    }
}

fn parse_assume_sorted(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<AssumeSortedDims> {
    let Some(obj) = obj else {
        return Ok(AssumeSortedDims::None);
    };
    if obj.is_none() {
        return Ok(AssumeSortedDims::None);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        let s = s.to_str()?;
        return match s {
            "all" => Ok(AssumeSortedDims::All),
            "none" => Ok(AssumeSortedDims::None),
            other => Err(PyTypeError::new_err(format!(
                "assume_sorted: expected 'all', 'none', or a list of dim names, got {other:?}",
            ))),
        };
    }
    let mut set = HashSet::new();
    let iter = if let Ok(list) =
        obj.cast::<PyList>()
    {
        list.iter().collect::<Vec<_>>()
    } else if let Ok(tup) = obj.cast::<PyTuple>() {
        tup.iter().collect::<Vec<_>>()
    } else {
        return Err(PyTypeError::new_err(
            "assume_sorted: expected str, list[str], or tuple[str]",
        ));
    };
    for item in iter {
        let s = item.cast::<PyString>()?;
        set.insert(s.to_str()?.istr());
    }
    Ok(AssumeSortedDims::Only(set))
}

#[pymethods]
impl PyBackendOptions {
    #[new]
    #[pyo3(signature = (
        *,
        coord_cache_max_entries = BackendOptions::default().coord_cache_max_entries,
        var_cache_max_entries = BackendOptions::default().var_cache_max_entries,
        assume_sorted = None,
    ))]
    fn new(
        coord_cache_max_entries: u64,
        var_cache_max_entries: u64,
        assume_sorted: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: BackendOptions {
                coord_cache_max_entries,
                var_cache_max_entries,
                assume_sorted:
                    parse_assume_sorted(
                        assume_sorted,
                    )?,
            },
        })
    }

    #[getter]
    fn coord_cache_max_entries(&self) -> u64 {
        self.inner.coord_cache_max_entries
    }

    #[getter]
    fn var_cache_max_entries(&self) -> u64 {
        self.inner.var_cache_max_entries
    }

    /// Returns `"all"`, `"none"`, or the list of asserted dims.
    #[getter]
    fn assume_sorted<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.assume_sorted {
            AssumeSortedDims::None => Ok(
                PyString::new(py, "none").into_any()
            ),
            AssumeSortedDims::All => Ok(
                PyString::new(py, "all").into_any()
            ),
            AssumeSortedDims::Only(set) => {
                let names: Vec<&str> = set
                    .iter()
                    .map(|s| {
                        AsRef::<str>::as_ref(s)
                    })
                    .collect();
                Ok(PyList::new(py, names)?
                    .into_any())
            }
        }
    }

    fn __repr__(&self) -> String {
        let assume = match &self.inner.assume_sorted {
            AssumeSortedDims::None => {
                "none".to_string()
            }
            AssumeSortedDims::All => {
                "all".to_string()
            }
            AssumeSortedDims::Only(set) => {
                let mut names: Vec<&str> = set
                    .iter()
                    .map(|s| s.as_ref())
                    .collect();
                names.sort();
                format!("{names:?}")
            }
        };
        format!(
            "BackendOptions(coord_cache_max_entries={}, var_cache_max_entries={}, assume_sorted={})",
            self.inner.coord_cache_max_entries,
            self.inner.var_cache_max_entries,
            assume,
        )
    }
}
