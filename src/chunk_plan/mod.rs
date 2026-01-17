//! Chunk planning: compile a Polars predicate `Expr` into a conservative set of Zarr chunk indices.

mod prelude;
mod compile_entry;

mod indexing;
mod exprs;

#[allow(unused_imports)]
pub(crate) use compile_entry::{
    compile_expr_to_chunk_plan, compile_expr_to_dataset_selection, PlannerStats,
};
pub(crate) use exprs::errors::CompileError;
pub(crate) use indexing::plan::{ChunkIndexIter, ChunkPlan};
pub(crate) use indexing::types::ChunkId;
#[allow(unused_imports)]
pub(crate) use indexing::selection::{
    DataArraySelection, DatasetSelection, HyperRectangleSelection, RangeList, ScalarRange,
};
#[allow(unused_imports)]
pub(crate) use indexing::selection_to_chunks::plan_dataset_chunk_indices;

