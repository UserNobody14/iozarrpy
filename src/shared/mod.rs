mod compile;
mod intern;
mod options;
mod parallel;
mod structural;
mod traits;
mod zarr;

pub use options::BackendOptions;
pub(crate) use parallel::MaybeParIter;

// Re-export commonly used traits
pub use traits::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync,
    ChunkedDataCacheAsync,
    EvictableChunkCacheAsync,
    EvictableChunkCacheSync,
    HasMetadataBackendAsync,
    HasMetadataBackendCacheAsync,
    HasMetadataBackendSync,
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
    diagonal_concat_batches,
    expand_projection_to_flat_paths,
};

pub(crate) use intern::*;
