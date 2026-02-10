//! Expression compilation for chunk planning.
//!
//! The compilation is lazy: expressions are analyzed to produce `LazyDatasetSelection`
//! containing unresolved value-based constraints. These are later batch-resolved
//! and materialized into concrete `DatasetSelection`.

pub mod compile_ctx;
pub mod compile_node;
pub mod compile_node_lazy;
pub mod expr_utils;
pub mod expr_walk;
pub mod literals;

pub(crate) use crate::chunk_plan::indexing::Emptyable;
pub(crate) use crate::chunk_plan::indexing::selection::SetOperations;
pub(crate) use compile_ctx::LazyCompileCtx;
pub(crate) use compile_node_lazy::compile_node_lazy;
pub(crate) use literals::apply_time_encoding;
