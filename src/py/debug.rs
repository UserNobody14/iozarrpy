use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::py::expr_extract::extract_expr;

#[pyfunction]
pub(crate) fn print_extension_info() -> String {
    "Rainbear extension module loaded successfully".to_string()
}

#[pyfunction]
pub(crate) fn _debug_expr_ast(predicate: &Bound<'_, PyAny>) -> PyResult<String> {
    let expr = extract_expr(predicate)?;
    Ok(format!("{expr:?}"))
}
