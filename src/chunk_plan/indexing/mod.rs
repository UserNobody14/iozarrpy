//! Index range computation and chunk planning.
//!
//! This module contains types and functions for:
//! - Value range to index range resolution
//! - Lazy selection types and materialization
//! - Chunk plan computation from selections

pub mod plan;
pub mod selection;
pub mod selection_to_chunks;
pub mod types;

// Generic grouped selection types
pub mod grouped_selection;

// Lazy selection types
pub mod lazy_selection;

// Lazy resolution and materialization
pub mod lazy_materialize;

// Resolver traits
pub mod resolver_traits;
pub(crate) use resolver_traits::{
    AsyncCoordResolver, SyncCoordResolver,
};

// Core types re-exports
pub(crate) use plan::GroupedChunkPlan;
pub use plan::ChunkSubset;
pub(crate) use selection::{
    DatasetSelection, Emptyable,
};
