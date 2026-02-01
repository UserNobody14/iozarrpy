//! Backend traits for Zarr data access with caching support.
//!
//! These traits provide an abstraction layer over Zarr storage, enabling:
//! - Persistent caching of coordinate array chunks across scans
//! - Thread-safe access to cached metadata
//! - Extensibility for alternative backends (icechunk, gribberish, etc.)

use std::collections::BTreeMap;
use std::fmt::Display;
use std::sync::Arc;

use ambassador::{Delegate, delegatable_trait};
use pyo3::PyErr;
use tokio::sync::RwLock;
use zarrs::storage::{
    AsyncReadableWritableListableStorage,
    ReadableWritableListableStorage,
};

use crate::IStr;
use crate::chunk_plan::CompileError;
use crate::reader::ColumnData;

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
    /// Compile error.
    CompileError(Box<CompileError>),
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

impl From<CompileError> for BackendError {
    fn from(v: CompileError) -> Self {
        Self::CompileError(Box::new(v))
    }
}

impl Display for BackendError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            BackendError::CompileError(err) => {
                write!(
                    f,
                    "compile error: {}",
                    err
                )
            }
            BackendError::CoordNotFound(dim) => {
                write!(
                    f,
                    "coordinate array not found: {}",
                    dim
                )
            }
            BackendError::ArrayOpenFailed(
                msg,
            ) => {
                write!(
                    f,
                    "failed to open array: {}",
                    msg
                )
            }
            BackendError::ChunkReadFailed(
                msg,
            ) => {
                write!(
                    f,
                    "failed to read chunk: {}",
                    msg
                )
            }
            BackendError::MetadataNotLoaded => {
                write!(
                    f,
                    "metadata not yet loaded"
                )
            }
            BackendError::Other(msg) => {
                write!(f, "{}", msg)
            }
        }
    }
}

impl std::error::Error for BackendError {}

/// To Py Error
impl From<BackendError> for PyErr {
    fn from(error: BackendError) -> PyErr {
        match error {
            BackendError::CoordNotFound(msg) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "coordinate array not found: {}",
                    msg
                ))
            }
            _ => PyErr::new::<pyo3::exceptions::PyValueError, _>(
                error.to_string(),
            ),
        }
    }
}

/// Synchronous chunked data backend trait.
///
/// Provides access to chunked data
/// and dataset metadata.
#[delegatable_trait]
pub trait ChunkedDataBackendSync:
    Send + Sync
{
    /// Read a chunk (coordinate or variable)
    ///
    /// # Arguments
    /// * `var` - The variable name
    /// * `chunk_idx` - The chunk index to read
    ///
    /// # Returns
    /// The chunk data, freshly loaded.
    fn read_chunk_sync(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<ColumnData, BackendError>;
}

/// Asynchronous chunked data backend trait.
///
/// Provides async access to chunked data
/// and dataset metadata.
#[async_trait::async_trait]
pub trait ChunkedDataBackendAsync:
    Send + Sync
{
    /// Read a chunk (coordinate or variable) asynchronously
    ///
    /// # Arguments
    /// * `var` - The variable name
    /// * `chunk_idx` - The indices of the chunk to read
    ///
    /// # Returns
    /// The chunk data, freshly loaded.
    async fn read_chunk_async(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<ColumnData, BackendError>;
}

/// A backend that can retrieve metadata synchronously
#[delegatable_trait]
pub trait HasMetadataBackendSync<
    METADATA: Send + Sync,
>
{
    fn metadata(
        &self,
    ) -> Result<Arc<METADATA>, BackendError>;
}

/// A backend that can retrieve metadata asynchronously
#[async_trait::async_trait]
pub trait HasMetadataBackendAsync<
    METADATA: Send + Sync,
>: Send + Sync
{
    async fn metadata(
        &self,
    ) -> Result<Arc<METADATA>, BackendError>;
}

#[delegatable_trait]
pub trait HasStore {
    fn store(
        &self,
    ) -> &ReadableWritableListableStorage;
}

#[delegatable_trait]
pub trait HasAsyncStore {
    fn async_store(
        &self,
    ) -> &AsyncReadableWritableListableStorage;
}

// =============================================================================
// Cache wrappers with ambassador delegation for sync traits
// =============================================================================

/// Sync cache for chunked data - delegates metadata and store traits to backend
#[derive(Delegate)]
#[delegate(HasMetadataBackendSync<METADATA>, target = "backend", generics = "METADATA", where = "METADATA: Send + Sync, BACKEND: HasMetadataBackendSync<METADATA>")]
#[delegate(
    HasStore,
    target = "backend",
    where = "BACKEND: HasStore"
)]
pub struct ChunkedDataCacheSync<
    BACKEND: ChunkedDataBackendSync,
> {
    backend: BACKEND,
    chunk_cache: RwLock<
        BTreeMap<(IStr, Vec<u64>), ColumnData>,
    >,
}

/// Async cache for chunked data
pub struct ChunkedDataCacheAsync<
    BACKEND: ChunkedDataBackendAsync,
> {
    backend: BACKEND,
    chunk_cache: RwLock<
        BTreeMap<(IStr, Vec<u64>), ColumnData>,
    >,
}

/// Sync cache for metadata - delegates chunked data and store traits to backend
#[derive(Delegate)]
#[delegate(
    ChunkedDataBackendSync,
    target = "backend",
    where = "BACKEND: ChunkedDataBackendSync"
)]
#[delegate(
    HasStore,
    target = "backend",
    where = "BACKEND: HasStore"
)]
pub struct HasMetadataBackendCacheSync<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendSync<METADATA>,
> {
    backend: BACKEND,
    metadata: Option<Arc<METADATA>>,
}

/// Async cache for metadata
pub struct HasMetadataBackendCacheAsync<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendAsync<METADATA>,
> {
    backend: BACKEND,
    metadata: RwLock<Option<Arc<METADATA>>>,
}

// =============================================================================
// Constructors
// =============================================================================

impl<BACKEND: ChunkedDataBackendSync>
    ChunkedDataCacheSync<BACKEND>
{
    pub fn new(backend: BACKEND) -> Self {
        Self {
            backend,
            chunk_cache: RwLock::new(
                BTreeMap::new(),
            ),
        }
    }
}

impl<BACKEND: ChunkedDataBackendAsync>
    ChunkedDataCacheAsync<BACKEND>
{
    pub fn new(backend: BACKEND) -> Self {
        Self {
            backend,
            chunk_cache: RwLock::new(
                BTreeMap::new(),
            ),
        }
    }
}

impl<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendSync<METADATA>,
> HasMetadataBackendCacheSync<METADATA, BACKEND>
{
    pub fn new(backend: BACKEND) -> Self {
        Self {
            backend,
            metadata: None,
        }
    }
}

impl<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendAsync<METADATA>,
>
    HasMetadataBackendCacheAsync<
        METADATA,
        BACKEND,
    >
{
    pub fn new(backend: BACKEND) -> Self {
        Self {
            backend,
            metadata: RwLock::new(None),
        }
    }
}

// =============================================================================
// Caching implementations (these add logic, not just delegation)
// =============================================================================

impl<BACKEND: ChunkedDataBackendSync>
    ChunkedDataBackendSync
    for ChunkedDataCacheSync<BACKEND>
{
    fn read_chunk_sync(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<ColumnData, BackendError> {
        let key =
            (var.clone(), chunk_idx.to_vec());
        let cache =
            self.chunk_cache.blocking_read();
        if let Some(data) = cache.get(&key) {
            return Ok(data.clone());
        }
        drop(cache);
        let data = self
            .backend
            .read_chunk_sync(var, chunk_idx)?;
        self.chunk_cache
            .blocking_write()
            .insert(key, data.clone());
        Ok(data)
    }
}

#[async_trait::async_trait]
impl<BACKEND: ChunkedDataBackendAsync>
    ChunkedDataBackendAsync
    for ChunkedDataCacheAsync<BACKEND>
{
    async fn read_chunk_async(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<ColumnData, BackendError> {
        let key =
            (var.clone(), chunk_idx.to_vec());
        let cache = self.chunk_cache.read().await;
        if let Some(data) = cache.get(&key) {
            return Ok(data.clone());
        }
        drop(cache);
        let data = self
            .backend
            .read_chunk_async(var, chunk_idx)
            .await?;
        self.chunk_cache
            .write()
            .await
            .insert(key, data.clone());
        Ok(data)
    }
}

impl<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendSync<METADATA>,
> HasMetadataBackendSync<METADATA>
    for HasMetadataBackendCacheSync<
        METADATA,
        BACKEND,
    >
{
    fn metadata(
        &self,
    ) -> Result<Arc<METADATA>, BackendError> {
        if let Some(metadata) =
            self.metadata.clone()
        {
            return Ok(metadata.clone());
        }
        let metadata = self.backend.metadata()?;
        Ok(metadata)
    }
}

#[async_trait::async_trait]
impl<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendAsync<METADATA>,
> HasMetadataBackendAsync<METADATA>
    for HasMetadataBackendCacheAsync<
        METADATA,
        BACKEND,
    >
{
    async fn metadata(
        &self,
    ) -> Result<Arc<METADATA>, BackendError> {
        if let Some(metadata) =
            &*self.metadata.read().await
        {
            return Ok(metadata.clone());
        }
        let metadata =
            self.backend.metadata().await?;
        *self.metadata.write().await =
            Some(metadata.clone());
        Ok(metadata)
    }
}

// =============================================================================
// Manual async trait delegations (ambassador doesn't work well with async_trait)
// =============================================================================

#[async_trait::async_trait]
impl<
    METADATA: Send + Sync,
    BACKEND: ChunkedDataBackendAsync
        + HasMetadataBackendAsync<METADATA>,
> ChunkedDataBackendAsync
    for HasMetadataBackendCacheAsync<
        METADATA,
        BACKEND,
    >
{
    async fn read_chunk_async(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<ColumnData, BackendError> {
        self.backend
            .read_chunk_async(var, chunk_idx)
            .await
    }
}

#[async_trait::async_trait]
impl<
    METADATA: Send + Sync,
    BACKEND: ChunkedDataBackendAsync
        + HasMetadataBackendAsync<METADATA>,
> HasMetadataBackendAsync<METADATA>
    for ChunkedDataCacheAsync<BACKEND>
{
    async fn metadata(
        &self,
    ) -> Result<Arc<METADATA>, BackendError> {
        self.backend.metadata().await
    }
}

impl<
    BACKEND: ChunkedDataBackendAsync + HasAsyncStore,
> HasAsyncStore
    for ChunkedDataCacheAsync<BACKEND>
{
    fn async_store(
        &self,
    ) -> &AsyncReadableWritableListableStorage
    {
        self.backend.async_store()
    }
}

impl<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendAsync<METADATA>
        + HasAsyncStore,
> HasAsyncStore
    for HasMetadataBackendCacheAsync<
        METADATA,
        BACKEND,
    >
{
    fn async_store(
        &self,
    ) -> &AsyncReadableWritableListableStorage
    {
        self.backend.async_store()
    }
}

// =============================================================================
// Display implementations (ambassador doesn't delegate std traits)
// =============================================================================

impl<BACKEND: ChunkedDataBackendSync + Display>
    Display for ChunkedDataCacheSync<BACKEND>
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "ChunkedDataCacheSync({})",
            self.backend
        )
    }
}

impl<BACKEND: ChunkedDataBackendAsync + Display>
    Display for ChunkedDataCacheAsync<BACKEND>
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "ChunkedDataCacheAsync({})",
            self.backend
        )
    }
}

impl<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendSync<METADATA> + Display,
> Display
    for HasMetadataBackendCacheSync<
        METADATA,
        BACKEND,
    >
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "HasMetadataBackendCacheSync({})",
            self.backend
        )
    }
}

impl<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendAsync<METADATA> + Display,
> Display
    for HasMetadataBackendCacheAsync<
        METADATA,
        BACKEND,
    >
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "HasMetadataBackendCacheAsync({})",
            self.backend
        )
    }
}

// =============================================================================
// Blanket impls for Arc<T>
// =============================================================================

impl<T: HasStore> HasStore for Arc<T> {
    fn store(
        &self,
    ) -> &ReadableWritableListableStorage {
        (**self).store()
    }
}

impl<T: HasAsyncStore> HasAsyncStore for Arc<T> {
    fn async_store(
        &self,
    ) -> &AsyncReadableWritableListableStorage
    {
        (**self).async_store()
    }
}

impl<T: ChunkedDataBackendSync>
    ChunkedDataBackendSync for Arc<T>
{
    fn read_chunk_sync(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<ColumnData, BackendError> {
        (**self).read_chunk_sync(var, chunk_idx)
    }
}

#[async_trait::async_trait]
impl<T: ChunkedDataBackendAsync>
    ChunkedDataBackendAsync for Arc<T>
{
    async fn read_chunk_async(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<ColumnData, BackendError> {
        (**self)
            .read_chunk_async(var, chunk_idx)
            .await
    }
}

impl<
    METADATA: Send + Sync,
    T: HasMetadataBackendSync<METADATA>,
> HasMetadataBackendSync<METADATA> for Arc<T>
{
    fn metadata(
        &self,
    ) -> Result<Arc<METADATA>, BackendError> {
        (**self).metadata()
    }
}

#[async_trait::async_trait]
impl<
    METADATA: Send + Sync,
    T: HasMetadataBackendAsync<METADATA>,
> HasMetadataBackendAsync<METADATA> for Arc<T>
{
    async fn metadata(
        &self,
    ) -> Result<Arc<METADATA>, BackendError> {
        (**self).metadata().await
    }
}

pub struct CacheStats {
    pub chunk_entries: usize,
}

#[delegatable_trait]
pub trait EvictableChunkCacheSync {
    fn cache_stats(&self) -> CacheStats;
    fn clear(&self);
}

#[async_trait::async_trait]
pub trait EvictableChunkCacheAsync {
    async fn cache_stats(&self) -> CacheStats;
    async fn clear(&self);
}

impl<BACKEND: ChunkedDataBackendSync>
    EvictableChunkCacheSync
    for ChunkedDataCacheSync<BACKEND>
{
    fn cache_stats(&self) -> CacheStats {
        CacheStats {
            chunk_entries: self
                .chunk_cache
                .blocking_read()
                .len(),
        }
    }
    fn clear(&self) {
        self.chunk_cache.blocking_write().clear();
    }
}

#[async_trait::async_trait]
impl<BACKEND: ChunkedDataBackendAsync>
    EvictableChunkCacheAsync
    for ChunkedDataCacheAsync<BACKEND>
{
    async fn cache_stats(&self) -> CacheStats {
        CacheStats {
            chunk_entries: self
                .chunk_cache
                .read()
                .await
                .len(),
        }
    }
    async fn clear(&self) {
        self.chunk_cache.write().await.clear();
    }
}

impl<
    BACKEND: ChunkedDataBackendSync
        + EvictableChunkCacheSync,
> EvictableChunkCacheSync for Arc<BACKEND>
{
    fn cache_stats(&self) -> CacheStats {
        (**self).cache_stats()
    }
    fn clear(&self) {
        (**self).clear()
    }
}

#[async_trait::async_trait]
impl<
    BACKEND: ChunkedDataBackendAsync
        + EvictableChunkCacheAsync,
> EvictableChunkCacheAsync for Arc<BACKEND>
{
    async fn cache_stats(&self) -> CacheStats {
        (**self).cache_stats().await
    }
    async fn clear(&self) {
        (**self).clear().await
    }
}

impl<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendSync<METADATA>
        + EvictableChunkCacheSync,
> EvictableChunkCacheSync
    for HasMetadataBackendCacheSync<
        METADATA,
        BACKEND,
    >
{
    fn cache_stats(&self) -> CacheStats {
        self.backend.cache_stats()
    }
    fn clear(&self) {
        self.backend.clear()
    }
}

#[async_trait::async_trait]
impl<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendAsync<METADATA>
        + EvictableChunkCacheAsync,
> EvictableChunkCacheAsync
    for HasMetadataBackendCacheAsync<
        METADATA,
        BACKEND,
    >
{
    async fn cache_stats(&self) -> CacheStats {
        self.backend.cache_stats().await
    }
    async fn clear(&self) {
        self.backend.clear().await
    }
}

// =============================================================================
// Combined ChunkDataSource traits for generic chunk processing
// =============================================================================

use crate::meta::{ZarrDatasetMeta, ZarrMeta};

/// Synchronous chunk data source - provides all capabilities needed for chunk processing.
///
/// This trait combines chunk reading and metadata access into a single interface,
/// enabling generic chunk processing code that can work with any backend implementation.
pub trait ChunkDataSourceSync:
    ChunkedDataBackendSync
    + HasMetadataBackendSync<ZarrMeta>
    + Send
    + Sync
{
    /// Get the planning metadata (ZarrDatasetMeta) for schema and dimension info.
    fn planning_meta(
        &self,
    ) -> Result<ZarrDatasetMeta, BackendError>
    {
        Ok(self.metadata()?.planning_meta())
    }
}

/// Asynchronous chunk data source - provides all capabilities needed for chunk processing.
///
/// This trait combines chunk reading and metadata access into a single interface,
/// enabling generic chunk processing code that can work with any backend implementation.
#[async_trait::async_trait]
pub trait ChunkDataSourceAsync:
    ChunkedDataBackendAsync
    + HasMetadataBackendAsync<ZarrMeta>
    + Send
    + Sync
{
    /// Get the planning metadata (ZarrDatasetMeta) for schema and dimension info.
    async fn planning_meta(
        &self,
    ) -> Result<ZarrDatasetMeta, BackendError>
    {
        Ok(self.metadata().await?.planning_meta())
    }
}

// Blanket implementations for types that implement both traits
impl<T> ChunkDataSourceSync for T where
    T: ChunkedDataBackendSync
        + HasMetadataBackendSync<ZarrMeta>
        + Send
        + Sync
{
}

impl<T> ChunkDataSourceAsync for T where
    T: ChunkedDataBackendAsync
        + HasMetadataBackendAsync<ZarrMeta>
        + Send
        + Sync
{
}

// =============================================================================
// Type aliases
// =============================================================================

/// A type-erased async backend that can be shared across threads.
pub type DynChunkedDataAsyncBackend =
    Arc<dyn ChunkedDataBackendAsync>;

/// A type-erased sync backend that can be shared across threads.
pub type DynChunkedDataSyncBackend =
    Arc<dyn ChunkedDataBackendSync>;
