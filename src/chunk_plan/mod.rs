//! Chunk planning: compile a Polars predicate `Expr` into a conservative set of Zarr chunk indices.
//!
//! The compilation uses a lazy approach for efficient I/O batching:
//! 1. Compile expression to `LazyDatasetSelection` (no I/O)
//! 2. Batch-resolve value ranges to index ranges (sync or async)
//! 3. Materialize to concrete `DatasetSelection`

mod compile_entry;
mod prelude;

mod exprs;
mod indexing;

pub(crate) use exprs::errors::CompileError;

pub(crate) use indexing::selection_to_chunks::selection_to_grouped_chunk_plan;
pub(crate) use indexing::selection_to_chunks::selection_to_grouped_chunk_plan_unified;
pub(crate) use indexing::types::ChunkId;
// Primary entry points (sync)
pub(crate) use compile_entry::{
    PlannerStats,
    compile_expr_to_dataset_selection,
};

// Primary entry points (async)

// GroupedChunkPlan entry points (heterogeneous chunk grids)
pub(crate) use compile_entry::{
    compile_expr_to_grouped_chunk_plan,
    compile_expr_to_grouped_chunk_plan_async,
    compile_expr_to_grouped_chunk_plan_unified,
    compile_expr_to_grouped_chunk_plan_unified_async,
};
pub(crate) use indexing::plan::GroupedChunkPlan;
pub(crate) use indexing::types::ChunkGridSignature;

// Advanced: lazy compilation + manual resolution
pub(crate) use compile_entry::{
    compile_expr_to_lazy_selection,
    resolve_lazy_selection_async,
    resolve_lazy_selection_sync,
};

// Lazy selection types and resolver traits for advanced usage
pub(crate) use indexing::selection::DSelection;

// Debug utilities - expose additional internal types
pub(crate) use indexing::lazy_materialize::collect_requests_with_meta;
pub(crate) use indexing::selection::DatasetSelection;
