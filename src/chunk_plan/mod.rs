//! Chunk planning: compile a Polars predicate `Expr` into a conservative set of Zarr chunk indices.

mod prelude;
mod types;
mod plan;
mod errors;
mod monotonic_scalar;
mod monotonic_bounds;
mod monotonic_impl;
mod literals;
mod compile_entry;
mod compile_node;
mod compile_boolean;
mod selector;
mod expr_utils;
mod compile_is_between;
mod compile_is_in;
mod compile_cmp;
mod index_ranges;

pub(crate) use compile_entry::{compile_expr_to_chunk_plan, PlannerStats};
pub(crate) use errors::CompileError;
pub(crate) use plan::{ChunkIndexIter, ChunkPlan};
pub(crate) use types::ChunkId;
