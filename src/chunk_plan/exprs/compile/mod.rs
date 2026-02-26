//! Lazy expression compilation - produces ExprPlan without resolving.
//!
//! This module compiles Polars expressions into `ExprPlan`, which separates
//! dimension constraints from variable tracking. The expensive `GroupedSelection`
//! construction is deferred to `ExprPlan::into_lazy_dataset_selection`.

mod boolean;
mod cmp;
mod expr;
mod interpolate;
mod selector;
mod utils;

pub(crate) use expr::compile_expr;
