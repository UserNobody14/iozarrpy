use std::collections::BTreeSet;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_polars::{PyDataFrame, PyExpr};

#[pyfunction]
#[pyo3(signature = (zarr_url, predicate, variables=None, max_concurrency=None, with_columns=None))]
pub(crate) fn scan_zarr_async(
    py: Python<'_>,
    zarr_url: String,
    predicate: &Bound<'_, PyAny>,
    variables: Option<Vec<String>>,
    max_concurrency: Option<usize>,
    with_columns: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    // Extract Expr under the GIL (and guard against panics during conversion).
    let expr = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let pyexpr: PyExpr = predicate.extract()?;
        Ok::<polars::prelude::Expr, PyErr>(pyexpr.0.clone())
    }))
    .map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "panic while converting predicate Expr",
        )
    })??;

    let with_columns: Option<BTreeSet<String>> =
        with_columns.map(|v| v.into_iter().collect::<BTreeSet<_>>());

    let awaitable = future_into_py(py, async move {
        let df = crate::scan::scan_zarr_df_async(zarr_url, expr, variables, max_concurrency, with_columns).await?;
        Python::attach(|py| Ok(PyDataFrame(df).into_pyobject(py)?.unbind()))
    })?;

    Ok(awaitable.unbind())
}
