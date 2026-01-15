use polars::prelude::Expr;
use polars::polars_utils::pl_serialize;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_polars::PyExpr;

fn panic_to_py_err(msg: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg.to_string())
}

/// Extract a Rust `polars::Expr` from a Python `polars.Expr`.
///
/// We try two wire formats:
/// - **Pickle state** (`__getstate__`): fast, but can be brittle if the Python wheel and this
///   extension were built with different Polars feature sets (enum layout drift). This is where
///   `FfiPlugin` expressions were failing.
/// - **Versioned binary** (`expr.meta.serialize()`): slower but more robust, and what Polars
///   documents for `pl.Expr.deserialize(io.BytesIO(bytes))`.
pub(crate) fn extract_expr(predicate: &Bound<'_, PyAny>) -> PyResult<Expr> {
    // Fast path: pyo3-polars `PyExpr` uses `__getstate__` under the hood.
    let fast = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let pyexpr: PyExpr = predicate.extract()?;
        Ok::<Expr, PyErr>(pyexpr.0.clone())
    }))
    .map_err(|_| panic_to_py_err("panic while converting predicate Expr"))?;

    match fast {
        Ok(expr) => return Ok(expr),
        Err(_err) => {
            // Fall back to the documented Expr binary serialization.
            // This is particularly important for `FfiPlugin` expressions, whose pickle encoding
            // is sensitive to feature-gated enum layouts.
            let meta = predicate.getattr("meta")?;
            let bytes_any = meta.call_method0("serialize")?;
            let b: Bound<'_, PyBytes> = bytes_any.extract()?;

            pl_serialize::SerializeOptions::default()
                .deserialize_from_reader::<Expr, &[u8], true>(b.as_bytes())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))
        }
    }
}

