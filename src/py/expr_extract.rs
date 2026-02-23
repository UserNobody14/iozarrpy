use polars::polars_utils::pl_serialize;
use polars::prelude::Expr;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_polars::PyExpr;

fn panic_to_py_err(msg: &str) -> PyErr {
    PyErr::new::<
        pyo3::exceptions::PyRuntimeError,
        _,
    >(msg.to_string())
}

/// Extract a Rust `polars::Expr` from a Python `polars.Expr`.
///
/// We try two wire formats:
/// - **Versioned binary** (`expr.meta.serialize()`): robust, and what Polars documents for
///   `pl.Expr.deserialize(io.BytesIO(bytes))`. Tried first because the pickle path can silently
///   corrupt expressions when feature flags differ (e.g. `rank()` → `cum_sum()`, lost `over()`
///   partition columns).
/// - **Pickle state** (`__getstate__`): fast fallback via `pyo3-polars`, used only when the
///   versioned binary path fails.
pub(crate) fn extract_expr(
    predicate: &Bound<'_, PyAny>,
) -> PyResult<Expr> {
    // Robust path: documented Expr binary serialization.
    let robust = (|| -> PyResult<Expr> {
        let meta = predicate.getattr("meta")?;
        let bytes_any =
            meta.call_method0("serialize")?;
        let b: Bound<'_, PyBytes> =
            bytes_any.extract()?;
        pl_serialize::SerializeOptions::default()
            .deserialize_from_reader::<Expr, &[u8], true>(
                b.as_bytes(),
            )
            .map_err(|e| {
                PyErr::new::<
                    pyo3::exceptions::PyRuntimeError,
                    _,
                >(format!("{e}"))
            })
    })();

    match robust {
        Ok(expr) => Ok(expr),
        Err(_) => {
            // Fall back to pyo3-polars pickle extraction.
            std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| {
                    let pyexpr: PyExpr =
                        predicate.extract()?;
                    Ok::<Expr, PyErr>(pyexpr.0.clone())
                }),
            )
            .map_err(|_| {
                panic_to_py_err(
                    "panic while converting predicate Expr",
                )
            })?
        }
    }
}
