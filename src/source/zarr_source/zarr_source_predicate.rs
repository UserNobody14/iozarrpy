use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::chunk_plan::compile_expr_to_grouped_chunk_plan;
use crate::IntoIStr;

use super::{panic_to_py_err, to_py_err, ZarrSource};

impl ZarrSource {
    pub(super) fn try_set_predicate_impl(&mut self, predicate: &Bound<'_, PyAny>) -> PyResult<()> {
        // IMPORTANT: Taking `PyExpr` directly in the signature can abort the whole
        // Python process if the Python->Rust Expr conversion panics.
        //
        // By accepting `PyAny` and extracting inside a `catch_unwind`, we can turn
        // those panics into a normal Python exception which the Python wrapper can
        // safely ignore (disabling pushdown but keeping correctness).
        let expr = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::py::expr_extract::extract_expr(predicate)
        }))
        .map_err(|e| panic_to_py_err(e, "panic while converting predicate Expr"))??;

        // Compile Expr -> GroupedChunkPlan (per-grid subsets)
        let plan_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compile_expr_to_grouped_chunk_plan(&expr, &self.meta, self.store.clone())
        }))
        .map_err(|e| panic_to_py_err(e, "panic while compiling predicate chunk plan"))?;

        let (plan, _stats) = plan_result.map_err(to_py_err)?;

        // Rebuild grid iteration states from the plan
        self.set_grid_states_from_plan(plan);
        
        self.predicate = Some(expr);
        Ok(())
    }

    pub(super) fn set_with_columns_impl(&mut self, columns: Vec<String>) {
        self.with_columns = Some(columns.into_iter().map(|s| s.istr()).collect());
    }
}
