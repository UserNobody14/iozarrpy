mod compile;
mod stats;
mod structural;
mod traits;
mod zarr;

pub(crate) use stats::PlannerStats;

// Re-export commonly used traits
pub(crate) use traits::{
    BackendError, ChunkDataSourceAsync,
    ChunkDataSourceSync, ChunkedDataBackendAsync,
    ChunkedDataBackendSync,
    ChunkedDataCacheAsync, ChunkedDataCacheSync,
    EvictableChunkCacheAsync,
    EvictableChunkCacheSync, HasAsyncStore,
    HasMetadataBackendAsync,
    HasMetadataBackendCacheAsync,
    HasMetadataBackendCacheSync,
    HasMetadataBackendSync, HasStats,
};

// Re-export compile traits
pub(crate) use compile::{
    ChunkedExpressionCompilerAsync,
    ChunkedExpressionCompilerSync,
};

// Re-export zarr traits
pub(crate) use zarr::{
    FullyCachedZarrBackendAsync,
    FullyCachedZarrBackendSync, ZarrBackendAsync,
    ZarrBackendSync, normalize_path,
    to_fully_cached_async, to_fully_cached_sync,
};

// Re-export structural traits
pub(crate) use structural::{
    combine_chunk_dataframes,
    expand_projection_to_flat_paths,
    restructure_to_structs,
};
