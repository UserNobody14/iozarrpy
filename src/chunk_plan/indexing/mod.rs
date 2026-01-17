pub mod plan;
pub mod selection;
pub mod selection_to_chunks;
pub mod types;
pub mod index_ranges;
pub mod monotonic_scalar;
pub mod monotonic_bounds;
pub mod monotonic_impl;
pub mod interpolate_selection_nd;

pub(crate) use plan::{ChunkIndexIter, ChunkPlan};
pub(crate) use selection::{
    DataArraySelection, DatasetSelection, HyperRectangleSelection, RangeList, ScalarRange,
};
pub(crate) use selection_to_chunks::plan_dataset_chunk_indices;
pub(crate) use types::{BoundKind, ChunkId, CoordScalar, DimChunkRange, IndexRange, ValueRange};
