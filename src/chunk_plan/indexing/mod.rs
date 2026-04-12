//! Index range computation and chunk planning.
//!
//! This module contains types and functions for:
//! - Value range to index range resolution
//! - Lazy selection types and inline resolution
//! - Chunk plan computation from selections

pub mod grid_execution;
pub mod plan;
pub mod selection;
pub mod selection_base;
pub mod selection_to_chunks;
pub mod streaming_batch_plan;
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
pub(crate) use grid_execution::{
    GridGroupExecutionOpts, OwnedGridGroup,
    apply_streaming_batch_io_cut,
    streaming_grid_chunk_read_count,
};
pub use plan::ChunkSubset;
pub use plan::GroupedChunkPlan;
pub use selection::DatasetSelection;
pub(crate) use streaming_batch_plan::{
    ScheduleBuilt, StreamingBatch,
    build_streaming_schedule,
    distinct_chunk_slots_in_batches,
};
