//! Chunk planning: compile a Polars predicate `Expr` into a conservative set of Zarr chunk indices.

mod prelude;
mod compile_entry;

mod indexing;
mod exprs;

pub(crate) use exprs::errors::CompileError;
pub(crate) use indexing::plan::{ChunkIndexIter, ChunkPlan};
pub(crate) use indexing::types::ChunkId;

pub(crate) use compile_entry::{
    compile_expr_to_chunk_plan, compile_expr_to_dataset_selection, PlannerStats,
    // Lazy compilation entry points
    compile_expr_to_chunk_plan_lazy, compile_expr_to_dataset_selection_lazy,
    compile_expr_to_lazy_selection, resolve_lazy_selection_sync,
    // Async entry points
    compile_expr_to_chunk_plan_async, compile_expr_to_dataset_selection_async,
};

pub(crate) use indexing::selection_to_chunks::plan_dataset_chunk_indices;

// Lazy selection types and resolver traits for advanced usage
pub(crate) use indexing::lazy_selection::LazyDatasetSelection;
pub(crate) use indexing::resolver_traits::{AsyncCoordResolver, SyncCoordResolver};
pub(crate) use indexing::monotonic_async::AsyncMonotonicResolver;