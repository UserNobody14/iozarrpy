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
pub mod index_ranges;

// Sync monotonic coordinate resolver
pub mod monotonic_scalar;
pub(crate) use monotonic_scalar::MonotonicCoordResolver;

// Async monotonic coordinate resolver
pub mod monotonic_async;
pub(crate) use monotonic_async::AsyncMonotonicResolver;

// Lazy selection types
pub mod lazy_selection;


// Lazy resolution and materialization
pub mod lazy_materialize;

// Resolver traits
pub mod resolver_traits;
pub(crate) use resolver_traits::{
    AsyncCoordResolver, SyncCoordResolver
};

// Core types re-exports
pub(crate) use selection::{
    DSelection, Emptyable, DatasetSelection
};
pub(crate) use types::DimChunkRange;
