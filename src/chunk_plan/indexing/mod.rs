//! Index range computation and chunk planning.
//!
//! This module contains types and functions for:
//! - Value range to index range resolution (eager, at compile time)
//! - The unified [`grid_join_tree::GridJoinTree`] that drives all batched reads,
//!   produced by [`builder::GridJoinTreeBuilder`].

pub mod grid_join_reader;
pub mod grid_join_tree;
pub mod index_set;
pub mod types;

pub mod builder;

pub use builder::{
    PlannerStats, compile_to_tree_async,
    compile_to_tree_sync,
};
pub use grid_join_tree::{
    ChunkSubset, GridJoinTree,
};
