pub mod plan;
pub mod selection;
pub mod selection_to_chunks;
pub mod types;
pub mod index_ranges;
pub mod monotonic_scalar;
pub mod monotonic_bounds;
pub mod monotonic_impl;

pub(crate) use plan::{ChunkIndexIter, ChunkPlan};
pub(crate) use selection::{
    SetOperations,
    DataArraySelection, DatasetSelection, HyperRectangleSelection, RangeList, ScalarRange,
};
pub(crate) use selection_to_chunks::plan_dataset_chunk_indices;
pub(crate) use types::{BoundKind, ChunkId, CoordScalar, DimChunkRange, IndexRange, ValueRange};
pub mod lazy_selection;

pub(crate) use lazy_selection::{
    LazyArraySelection, LazyDatasetSelection, LazyDimConstraint, LazyHyperRectangle,
    lazy_dataset_all_for_vars, lazy_dataset_for_vars_with_selection,
};
pub mod resolver_traits;

pub(crate) use resolver_traits::{
    AsyncCoordResolver, HashMapCache, ResolutionCache, ResolutionError,
    ResolutionRequest, SyncCoordResolver,
};
pub mod lazy_materialize;

pub(crate) use lazy_materialize::{
    collect_requests, collect_requests_with_meta, materialize, materialize_with_dim_lengths,
    MergedCache,
};
pub mod monotonic_async;

pub(crate) use monotonic_async::AsyncMonotonicResolver;
