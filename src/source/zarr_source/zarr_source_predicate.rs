use pyo3::prelude::*;
use pyo3::types::PyAny;
use zarrs::array::Array;

use crate::chunk_plan::{compile_expr_to_chunk_plan, ChunkPlan};

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

        // Compile Expr -> candidate-chunk plan (no full-grid scans).
        let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compile_expr_to_chunk_plan(&expr, &self.meta, self.store.clone(), &self.vars[0])
        }))
        .map_err(|e| panic_to_py_err(e, "panic while compiling predicate chunk plan"))?;

        match compiled {
            Ok((plan, _stats)) => {
                self.primary_grid_shape = plan.grid_shape().to_vec();
                self.chunk_iter = plan.into_index_iter();
                self.current_chunk_indices = None;
                self.chunk_offset = 0;
            }
            Err(_) => {
                // Fall back to scanning all chunks if planning fails.
                let primary_path = self.meta.arrays[&self.vars[0]].path.clone();
                let primary = Array::open(self.store.clone(), &primary_path).map_err(to_py_err)?;
                let grid_shape = primary.chunk_grid().grid_shape().to_vec();
                self.primary_grid_shape = grid_shape.clone();
                self.chunk_iter = ChunkPlan::all(grid_shape).into_index_iter();
                self.current_chunk_indices = None;
                self.chunk_offset = 0;
            }
        }

        self.predicate = Some(expr);
        Ok(())
    }

    pub(super) fn set_with_columns_impl(&mut self, columns: Vec<String>) {
        self.with_columns = Some(columns.into_iter().collect());
    }
}

