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
pub(crate) use compile_entry::compute_dims_and_lengths_unified;
pub(crate) use compile_entry::{
    compile_expr_to_grouped_chunk_plan_unified    
};
pub(crate) use exprs::LazyCompileCtx;
pub(crate) use exprs::compile_node_lazy;
pub(crate) use indexing::AsyncCoordResolver;
pub(crate) use indexing::GroupedChunkPlan;
pub(crate) use chunk_plan::indexing::resolver_traits::HashMapCache;
pub(crate) use indexing::lazy_materialize::{
    MergedCache, collect_requests_with_meta,
    materialize,
};
pub(crate) use indexing::selection_to_chunks::selection_to_grouped_chunk_plan_unified_from_meta;
pub(crate) use indexing::resolver_traits::ResolutionRequest;
pub(crate) use indexing::resolver_traits::ResolutionCache;
pub(crate) use indexing::types::ChunkGridSignature;
pub(crate) use indexing::types::IndexRange;
pub(crate) use indexing::types::ValueRange;
pub(crate) use indexing::types::CoordScalar;
pub(crate) use exprs::apply_time_encoding;
pub(crate) use indexing::types::BoundKind;
use crate::chunk_plan;
