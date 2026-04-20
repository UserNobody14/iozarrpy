//! Chunk planning: compile a Polars predicate `Expr` into a [`GridJoinTree`].
//!
//! The compilation flow:
//! 1. Compile expression to `ExprPlan` (no I/O).
//! 2. Walk the plan in [`indexing::builder::GridJoinTreeBuilder`], resolving
//!    each per-dim constraint against the backend (binary search on cached
//!    coordinate chunks) and accumulating per-dim index ranges.
//! 3. `finalize` groups vars by [`ChunkGridSignature`], drops redundant
//!    1D dim-coord groups, and emits the [`GridJoinTree`] consumed by
//!    `grid_join_reader`.

mod compile_entry;
pub(crate) mod coord_resolve;
mod prelude;

pub(crate) mod exprs;
pub(crate) mod indexing;

mod selection;
pub use compile_entry::compute_dims_and_lengths_unified;
pub use exprs::LazyCompileCtx;
pub use exprs::compile_expr;
pub(crate) use exprs::compile_node::collect_column_refs;

pub use indexing::ChunkSubset;
pub use indexing::types::ChunkGridSignature;

pub use indexing::GridJoinTree;
pub use indexing::{
    PlannerStats, compile_to_tree_async,
    compile_to_tree_sync,
};
