//! Chunk planning: compile a Polars predicate `Expr` into a conservative set of Zarr chunk indices.
//!
//! The compilation uses a lazy approach for efficient I/O batching:
//! 1. Compile expression to `LazyDatasetSelection` (no I/O)
//! 2. Batch-resolve value ranges to index ranges (sync or async)
//! 3. Materialize to concrete `DatasetSelection`

mod prelude;
mod compile_entry;

mod indexing;
mod exprs;

pub(crate) use exprs::errors::CompileError;
pub(crate) use indexing::plan::{ChunkIndexIter, ChunkPlan};
pub(crate) use indexing::types::ChunkId;

// Primary entry points (sync)
pub(crate) use compile_entry::{
    compile_expr_to_chunk_plan, compile_expr_to_dataset_selection, PlannerStats,
};

// Primary entry points (async)
pub(crate) use compile_entry::{
    compile_expr_to_chunk_plan_async, compile_expr_to_dataset_selection_async,
};

// Advanced: lazy compilation + manual resolution
pub(crate) use compile_entry::{
    compile_expr_to_lazy_selection, resolve_lazy_selection_sync, resolve_lazy_selection_async,
};

pub(crate) use indexing::selection_to_chunks::plan_dataset_chunk_indices;

// Lazy selection types and resolver traits for advanced usage
pub(crate) use indexing::lazy_selection::LazyDatasetSelection;
pub(crate) use indexing::resolver_traits::{AsyncCoordResolver, SyncCoordResolver};
pub(crate) use indexing::monotonic_async::AsyncMonotonicResolver;
