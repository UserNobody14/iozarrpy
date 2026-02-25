//! Chunk planning: compile a Polars predicate `Expr` into a conservative set of Zarr chunk indices.
//!
//! The compilation flow:
//! 1. Compile expression to `ExprPlan` (no I/O)
//! 2. Resolve constraints inline via the backend (binary search on cached coordinate chunks)
//! 3. Convert to `GroupedChunkPlan`

mod compile_entry;
mod prelude;

mod exprs;
mod indexing;

mod selection;
pub(crate) use compile_entry::compute_dims_and_lengths_unified;
pub(crate) use exprs::LazyCompileCtx;
pub(crate) use exprs::apply_time_encoding;
pub(crate) use exprs::compile_expr;
pub use indexing::ChunkSubset;
pub(crate) use indexing::GroupedChunkPlan;
pub(crate) use indexing::lazy_materialize::{
    resolve_expr_plan_async,
    resolve_expr_plan_sync,
};
pub(crate) use indexing::plan::ConsolidatedGridGroup;
pub(crate) use indexing::resolver_traits::ResolutionError;
pub(crate) use indexing::selection_to_chunks::selection_to_grouped_chunk_plan_unified_from_meta;
pub(crate) use indexing::types::ChunkGridSignature;
