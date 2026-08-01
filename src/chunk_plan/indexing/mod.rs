//! Index range computation and chunk planning.
//!
//! This module contains types and functions for:
//! - Value range to index range resolution
//! - Lazy selection types and inline resolution
//! - Chunk plan computation from selections
//! - The unified [`grid_join_tree::GridJoinTree`] that drives all batched reads

pub mod grid_join_reader;
pub mod grid_join_tree;
pub mod plan;
pub mod selection;
pub mod selection_base;
pub mod selection_to_chunks;
pub mod types;

pub mod grouped_selection;

pub mod lazy_selection;

pub mod lazy_materialize;

pub mod resolver_traits;

pub use grid_join_tree::GridJoinTree;
pub use plan::{ChunkSubset, GroupedChunkPlan};
pub use selection::DatasetSelection;
