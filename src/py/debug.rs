use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_polars::PyExpr;

#[pyfunction]
pub(crate) fn print_extension_info() -> String {
    "Rainbear extension module loaded successfully".to_string()
}

#[pyfunction]
pub(crate) fn _debug_expr_ast(predicate: &Bound<'_, PyAny>) -> PyResult<String> {
    let expr = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let pyexpr: PyExpr = predicate.extract()?;
        Ok::<polars::prelude::Expr, PyErr>(pyexpr.0.clone())
    }))
    .map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("panic while converting predicate Expr")
    })??;
    Ok(format!("{expr:?}"))
}
