//! Backend traits for Zarr data access with caching support.
//!
//! These traits provide an abstraction layer over Zarr storage, enabling:
//! - Persistent caching of coordinate array chunks across scans
//! - Thread-safe access to cached metadata
//! - Extensibility for alternative backends (icechunk, gribberish, etc.)

use std::sync::Arc;

use zarrs::storage::{AsyncReadableWritableListableStorage, ReadableWritableListableStorage};

use crate::meta::ZarrDatasetMeta;
use crate::IStr;

/// Cached chunk data for a coordinate array.
///
/// Stores decoded coordinate values in the appropriate numeric type.
#[derive(Debug, Clone)]
pub enum CoordChunkData {
    F64(Vec<f64>),
    I64(Vec<i64>),
    U64(Vec<u64>),
}

impl CoordChunkData {
    /// Get the length of the chunk data.
    pub fn len(&self) -> usize {
        match self {
            CoordChunkData::F64(v) => v.len(),
            CoordChunkData::I64(v) => v.len(),
            CoordChunkData::U64(v) => v.len(),
        }
    }

    /// Check if the chunk data is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a scalar value at the given offset.
    ///
    /// For I64 data with time encoding, the caller should apply time encoding separately.
    pub fn get_raw(&self, offset: usize) -> Option<CoordScalarRaw> {
        match self {
            CoordChunkData::F64(v) => v.get(offset).copied().map(CoordScalarRaw::F64),
            CoordChunkData::I64(v) => v.get(offset).copied().map(CoordScalarRaw::I64),
            CoordChunkData::U64(v) => v.get(offset).copied().map(CoordScalarRaw::U64),
        }
    }
}

/// Raw scalar value from a coordinate chunk (before time encoding).
#[derive(Debug, Clone, Copy)]
pub enum CoordScalarRaw {
    F64(f64),
    I64(i64),
    U64(u64),
}

/// Error type for backend operations.
#[derive(Debug, Clone)]
pub enum BackendError {
    /// The requested coordinate array was not found.
    CoordNotFound(String),
    /// Failed to open the zarr array.
    ArrayOpenFailed(String),
    /// Failed to read chunk data.
    ChunkReadFailed(String),
    /// Metadata not yet loaded.
    MetadataNotLoaded,
    /// Other error.
    Other(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::CoordNotFound(dim) => write!(f, "coordinate array not found: {}", dim),
            BackendError::ArrayOpenFailed(msg) => write!(f, "failed to open array: {}", msg),
            BackendError::ChunkReadFailed(msg) => write!(f, "failed to read chunk: {}", msg),
            BackendError::MetadataNotLoaded => write!(f, "metadata not yet loaded"),
            BackendError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// Synchronous Zarr backend trait.
///
/// Provides access to Zarr data with optional caching of coordinate chunks
/// and dataset metadata.
pub trait ZarrBackendSync: Send + Sync {
    /// Get the cached dataset metadata.
    ///
    /// Returns `None` if metadata hasn't been loaded yet.
    fn metadata(&self) -> Option<Arc<ZarrDatasetMeta>>;

    /// Load and cache the dataset metadata.
    fn load_metadata(&self) -> Result<Arc<ZarrDatasetMeta>, BackendError>;

    /// Read a coordinate chunk, using cache if available.
    ///
    /// # Arguments
    /// * `dim` - The dimension/coordinate name
    /// * `chunk_idx` - The chunk index to read
    ///
    /// # Returns
    /// The chunk data, either from cache or freshly loaded.
    fn read_coord_chunk(&self, dim: &IStr, chunk_idx: u64) -> Result<CoordChunkData, BackendError>;

    /// Get the underlying sync store.
    fn sync_store(&self) -> ReadableWritableListableStorage;

    /// Get the root path within the store.
    fn root(&self) -> &str;
}

/// Asynchronous Zarr backend trait.
///
/// Provides async access to Zarr data with optional caching of coordinate chunks
/// and dataset metadata.
#[async_trait::async_trait]
pub trait ZarrBackendAsync: Send + Sync {
    /// Get the cached dataset metadata.
    ///
    /// Returns `None` if metadata hasn't been loaded yet.
    fn metadata(&self) -> Option<Arc<ZarrDatasetMeta>>;

    /// Load and cache the dataset metadata asynchronously.
    async fn load_metadata(&self) -> Result<Arc<ZarrDatasetMeta>, BackendError>;

    /// Read a coordinate chunk asynchronously, using cache if available.
    ///
    /// # Arguments
    /// * `dim` - The dimension/coordinate name
    /// * `chunk_idx` - The chunk index to read
    ///
    /// # Returns
    /// The chunk data, either from cache or freshly loaded.
    async fn read_coord_chunk(
        &self,
        dim: &IStr,
        chunk_idx: u64,
    ) -> Result<CoordChunkData, BackendError>;

    /// Get the underlying async store.
    fn async_store(&self) -> AsyncReadableWritableListableStorage;

    /// Get the root path within the store.
    fn root(&self) -> &str;
}

/// A type-erased async backend that can be shared across threads.
pub type DynAsyncBackend = Arc<dyn ZarrBackendAsync>;

/// A type-erased sync backend that can be shared across threads.
pub type DynSyncBackend = Arc<dyn ZarrBackendSync>;
