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
pub(crate) use lazy_selection::{
    LazyArraySelection, LazyDatasetSelection, LazyDimConstraint, LazyHyperRectangle,
    lazy_dataset_all_for_vars, lazy_dataset_for_vars_with_selection,
};

// Lazy resolution and materialization
pub mod lazy_materialize;
pub(crate) use lazy_materialize::{
    collect_requests, collect_requests_with_meta, materialize, materialize_with_dim_lengths,
    MergedCache,
};

// Resolver traits
pub mod resolver_traits;
pub(crate) use resolver_traits::{
    AsyncCoordResolver, HashMapCache, ResolutionCache, ResolutionError,
    ResolutionRequest, SyncCoordResolver,
};

// Core types re-exports
pub(crate) use plan::{ChunkIndexIter, ChunkPlan};
pub(crate) use selection::{
    SetOperations,
    DataArraySelection, DatasetSelection, HyperRectangleSelection, RangeList, ScalarRange,
};
pub(crate) use selection_to_chunks::plan_dataset_chunk_indices;
pub(crate) use types::{BoundKind, ChunkId, CoordScalar, DimChunkRange, IndexRange, ValueRange};
