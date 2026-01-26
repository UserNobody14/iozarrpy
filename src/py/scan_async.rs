use std::collections::BTreeSet;

use polars::prelude::{IntoLazy, LazyFrame};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_polars::PyDataFrame;

use crate::py::expr_extract::extract_expr;
use crate::store::StoreInput;

#[pyfunction]
#[pyo3(signature = (store, predicate, variables=None, max_concurrency=None, with_columns=None, prefix=None))]
pub(crate) fn scan_zarr_async(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    predicate: &Bound<'_, PyAny>,
    variables: Option<Vec<String>>,
    max_concurrency: Option<usize>,
    with_columns: Option<Vec<String>>,
    prefix: Option<String>,
) -> PyResult<Py<PyAny>> {
    // Extract store input under the GIL
    let store_input = StoreInput::from_py(store, prefix)?;

    // Extract Expr under the GIL (and guard against panics during conversion).
    let expr = extract_expr(predicate)?;
    let expr2 = expr.clone();

    let with_columns: Option<BTreeSet<String>> =
        with_columns.map(|v| v.into_iter().collect::<BTreeSet<_>>());

    let awaitable = future_into_py(py, async move {
        let df = crate::scan::scan_zarr_df_async(store_input, expr, variables, max_concurrency, with_columns).await?;
        let filtered = df.lazy().filter(expr2).collect().map_err(
            |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        )?;
        Python::attach(|py| Ok(PyDataFrame(filtered).into_pyobject(py)?.unbind()))
    })?;

    Ok(awaitable.unbind())
}
