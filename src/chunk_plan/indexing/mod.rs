//! Index range computation and chunk planning.
//!
//! This module contains types and functions for:
//! - Value range to index range resolution
//! - Lazy selection types and inline resolution
//! - Chunk plan computation from selections

pub mod plan;
pub mod selection;
pub mod selection_base;
pub mod selection_to_chunks;
pub mod types;

// Generic grouped selection types
pub mod grouped_selection;

// Lazy selection types
pub mod lazy_selection;

// Direct resolution and materialization
pub mod lazy_materialize;

// Resolver traits (error types, dim context)
pub mod resolver_traits;

// Core types re-exports
pub use plan::ChunkSubset;
pub(crate) use plan::GroupedChunkPlan;
pub(crate) use selection::DatasetSelection;
