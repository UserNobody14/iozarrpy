//! Expression compilation for chunk planning.
//!
//! Expressions are analyzed to produce `ExprPlan` containing unresolved
//! value-based constraints. These are resolved inline via the backend
//! in `lazy_materialize::resolve_expr_plan_sync/async`.

pub mod compile;
pub mod compile_ctx;
pub mod compile_node;
pub mod expr_plan;
pub mod expr_utils;
pub mod expr_walk;
pub mod literals;

pub use compile::compile_expr;
pub use compile_ctx::LazyCompileCtx;
pub use expr_plan::ExprPlan;
pub(crate) use literals::apply_time_encoding;
