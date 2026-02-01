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

// Primary entry points (sync)
pub(crate) use compile_entry::PlannerStats;

// Primary entry points (async)

// GroupedChunkPlan entry points (heterogeneous chunk grids)
pub(crate) use compile_entry::{
    compile_expr_to_grouped_chunk_plan_unified,
    compile_expr_to_grouped_chunk_plan_unified_async,
    compile_expr_with_backend_async,
};
pub(crate) use indexing::plan::GroupedChunkPlan;
pub(crate) use indexing::types::ChunkGridSignature;
