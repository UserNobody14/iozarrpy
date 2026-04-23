//! Backend traits for Zarr data access with caching support.
//!
//! These traits provide an abstraction layer over Zarr storage, enabling:
//! - Persistent caching of coordinate array chunks across scans
//! - Thread-safe access to cached metadata
//! - Extensibility for alternative backends (icechunk, gribberish, etc.)
// Use the synchronous cache.
use moka::future::Cache as MokaFutureCache;
use moka::sync::Cache as MokaCache;
use std::collections::HashSet;
use std::fmt::Display;
use std::sync::{Arc, RwLock as StdRwLock};

use ambassador::{Delegate, delegatable_trait};
use tokio::sync::RwLock;

use crate::errors::BackendError;
use crate::meta::ZarrMeta;
use crate::reader::ColumnData;
use crate::shared::options::BackendOptions;
use crate::shared::{IStr, IntoIStr};

/// Build a moka sync cache with `max_entries` capacity (`0` = unbounded).
fn build_sync_cache<K, V>(
    max_entries: u64,
) -> MokaCache<K, V>
where
    K: std::hash::Hash
        + Eq
        + Send
        + Sync
        + 'static,
    V: Clone + Send + Sync + 'static,
{
    if max_entries == 0 {
        MokaCache::builder().build()
    } else {
        MokaCache::new(max_entries)
    }
}

/// Build a moka future cache with `max_entries` capacity (`0` = unbounded).
fn build_future_cache<K, V>(
    max_entries: u64,
) -> MokaFutureCache<K, V>
where
    K: std::hash::Hash
        + Eq
        + Send
        + Sync
        + 'static,
    V: Clone + Send + Sync + 'static,
{
    if max_entries == 0 {
        MokaFutureCache::builder().build()
    } else {
        MokaFutureCache::new(max_entries)
    }
}

/// Normalize a path-like `IStr` to the canonical (no-slash) form used by
/// [`ZarrMeta::all_coord_paths`]. Cheap thanks to string interning.
fn canonical_path_istr(s: &IStr) -> IStr {
    let raw: &str = s.as_ref();
    let trimmed = raw
        .trim_start_matches('/')
        .trim_end_matches('/');
    if trimmed == raw {
        *s
    } else {
        trimmed.istr()
    }
}

/// Build a `HashSet` of canonical coordinate paths from `meta`.
fn coord_set_from_meta(
    meta: &ZarrMeta,
) -> Arc<HashSet<IStr>> {
    Arc::new(
        meta.all_coord_paths()
            .into_iter()
            .collect(),
    )
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
    ) -> Result<Arc<ColumnData>, BackendError>;
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
    ) -> Result<Arc<ColumnData>, BackendError>;
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

// =============================================================================
// Cache wrappers with ambassador delegation for sync traits
// =============================================================================

/// Sync cache for chunked data - delegates metadata and store traits to backend
#[derive(Delegate)]
#[allow(clippy::duplicated_attributes)]
// ambassador: separate `#[delegate]` per trait, same target field
#[delegate(HasMetadataBackendSync<METADATA>, target = "backend", generics = "METADATA", where = "METADATA: Send + Sync, BACKEND: HasMetadataBackendSync<METADATA>")]
/// Sync in-memory chunk cache with the same **in-flight coalescing** behavior as
/// [`ChunkedDataCacheAsync`] (via [`moka::sync::Cache::try_get_with`]).
///
/// Coordinate chunks (latitude, longitude, time, lead_time, ...) and
/// data-variable chunks live in **separate** moka caches so that a long
/// scan over many large variables cannot evict the small set of coordinate
/// chunks that the next call will almost certainly request again.
pub struct ChunkedDataCacheSync<
    BACKEND: ChunkedDataBackendSync,
> {
    backend: BACKEND,
    coord_cache: MokaCache<
        (IStr, Vec<u64>),
        Arc<ColumnData>,
    >,
    var_cache: MokaCache<
        (IStr, Vec<u64>),
        Arc<ColumnData>,
    >,
    /// Memoized canonical coord paths, populated lazily on first read.
    coord_paths: parking_lot::RwLock<
        Option<Arc<HashSet<IStr>>>,
    >,
}

/// Async cache for chunked data.
///
/// Chunk loads use Moka's [`moka::future::Cache::try_get_with`], which **coalesces**
/// concurrent requests for the same `(zarr path, chunk index)`: only one underlying
/// read runs; other waiters await the same result. This avoids duplicate I/O when many
/// tasks request the same coordinate chunk or variable chunk before the first load finishes.
///
/// Coordinate chunks and data-variable chunks live in **separate** moka caches
/// (see [`ChunkedDataCacheSync`] for rationale).
pub struct ChunkedDataCacheAsync<
    BACKEND: ChunkedDataBackendAsync,
> {
    backend: BACKEND,
    coord_cache: MokaFutureCache<
        (IStr, Vec<u64>),
        Arc<ColumnData>,
    >,
    var_cache: MokaFutureCache<
        (IStr, Vec<u64>),
        Arc<ColumnData>,
    >,
    /// Memoized canonical coord paths, populated lazily on first read.
    coord_paths:
        RwLock<Option<Arc<HashSet<IStr>>>>,
}

/// Sync cache for metadata - delegates chunked data and store traits to backend
#[derive(Delegate)]
#[allow(clippy::duplicated_attributes)]
// ambassador: separate `#[delegate]` per trait, same target field
#[delegate(
    ChunkedDataBackendSync,
    target = "backend",
    where = "BACKEND: ChunkedDataBackendSync"
)]
pub struct HasMetadataBackendCacheSync<
    METADATA: Send + Sync,
    BACKEND: HasMetadataBackendSync<METADATA>,
> {
    backend: BACKEND,
    metadata: StdRwLock<Option<Arc<METADATA>>>,
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
    /// Build the cache from [`BackendOptions`].
    ///
    /// Each of `coord_cache_max_entries` / `var_cache_max_entries` is the
    /// upper bound on cached `(zarr path, chunk index)` entries in the
    /// respective sub-cache; `0` means **unbounded**.
    pub fn new(
        backend: BACKEND,
        options: BackendOptions,
    ) -> Self {
        Self {
            backend,
            coord_cache: build_sync_cache(
                options.coord_cache_max_entries,
            ),
            var_cache: build_sync_cache(
                options.var_cache_max_entries,
            ),
            coord_paths: parking_lot::RwLock::new(
                None,
            ),
        }
    }
}

impl<BACKEND: ChunkedDataBackendAsync>
    ChunkedDataCacheAsync<BACKEND>
{
    /// Build the cache from [`BackendOptions`].
    ///
    /// Each of `coord_cache_max_entries` / `var_cache_max_entries` is the
    /// upper bound on cached `(zarr path, chunk index)` entries in the
    /// respective sub-cache; `0` means **unbounded**.
    pub fn new(
        backend: BACKEND,
        options: BackendOptions,
    ) -> Self {
        Self {
            backend,
            coord_cache: build_future_cache(
                options.coord_cache_max_entries,
            ),
            var_cache: build_future_cache(
                options.var_cache_max_entries,
            ),
            coord_paths: RwLock::new(None),
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
            metadata: StdRwLock::new(None),
        }
    }

    /// Whether metadata has already been loaded and cached.
    pub fn has_metadata_cached(&self) -> bool {
        self.metadata
            .read()
            .ok()
            .is_some_and(|g| g.is_some())
    }

    /// Clear both metadata and coordinate chunk caches (where supported).
    pub fn clear_all_caches(&self)
    where
        BACKEND: EvictableChunkCacheSync,
    {
        self.backend.clear();
        if let Ok(mut g) = self.metadata.write() {
            *g = None;
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

    /// Whether metadata has already been loaded and cached.
    pub async fn has_metadata_cached(
        &self,
    ) -> bool {
        self.metadata.read().await.is_some()
    }

    /// Clear both metadata and coordinate chunk caches (where supported).
    pub async fn clear_all_caches(&self)
    where
        BACKEND: EvictableChunkCacheAsync,
    {
        self.backend.clear().await;
        *self.metadata.write().await = None;
    }
}

// =============================================================================
// Caching implementations (these add logic, not just delegation)
// =============================================================================

impl<
    BACKEND: ChunkedDataBackendSync
        + HasMetadataBackendSync<ZarrMeta>,
> ChunkedDataCacheSync<BACKEND>
{
    /// Returns `true` if `var` resolves to a coordinate array.
    ///
    /// Lazily populates the memoized coord-path set on first call;
    /// falls back to `false` if metadata cannot be loaded (treating
    /// the read as a data variable, which is the safer default since
    /// the variable cache is the smaller one).
    fn is_coord_path(&self, var: &IStr) -> bool {
        let canon = canonical_path_istr(var);
        if let Some(set) =
            self.coord_paths.read().as_ref()
        {
            return set.contains(&canon);
        }
        let meta = match self.backend.metadata() {
            Ok(m) => m,
            Err(_) => return false,
        };
        let set = coord_set_from_meta(&meta);
        let contains = set.contains(&canon);
        *self.coord_paths.write() = Some(set);
        contains
    }
}

impl<
    BACKEND: ChunkedDataBackendAsync
        + HasMetadataBackendAsync<ZarrMeta>,
> ChunkedDataCacheAsync<BACKEND>
{
    /// Async counterpart of [`ChunkedDataCacheSync::is_coord_path`].
    async fn is_coord_path(
        &self,
        var: &IStr,
    ) -> bool {
        let canon = canonical_path_istr(var);
        {
            let g = self.coord_paths.read().await;
            if let Some(set) = g.as_ref() {
                return set.contains(&canon);
            }
        }
        let meta =
            match self.backend.metadata().await {
                Ok(m) => m,
                Err(_) => return false,
            };
        let set = coord_set_from_meta(&meta);
        let contains = set.contains(&canon);
        *self.coord_paths.write().await =
            Some(set);
        contains
    }
}

impl<
    BACKEND: ChunkedDataBackendSync
        + HasMetadataBackendSync<ZarrMeta>,
> ChunkedDataBackendSync
    for ChunkedDataCacheSync<BACKEND>
{
    fn read_chunk_sync(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<Arc<ColumnData>, BackendError>
    {
        let key = (*var, chunk_idx.to_vec());
        let cache = if self.is_coord_path(var) {
            &self.coord_cache
        } else {
            &self.var_cache
        };
        cache
            .try_get_with(key, || {
                self.backend.read_chunk_sync(
                    var, chunk_idx,
                )
            })
            .map_err(|arc| {
                BackendError::other(
                    arc.to_string(),
                )
            })
    }
}

#[async_trait::async_trait]
impl<
    BACKEND: ChunkedDataBackendAsync
        + HasMetadataBackendAsync<ZarrMeta>,
> ChunkedDataBackendAsync
    for ChunkedDataCacheAsync<BACKEND>
{
    async fn read_chunk_async(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<Arc<ColumnData>, BackendError>
    {
        let key = (*var, chunk_idx.to_vec());
        let cache =
            if self.is_coord_path(var).await {
                &self.coord_cache
            } else {
                &self.var_cache
            };
        cache
            .try_get_with(key, async {
                self.backend
                    .read_chunk_async(
                        var, chunk_idx,
                    )
                    .await
            })
            .await
            .map_err(|arc| {
                BackendError::other(
                    arc.to_string(),
                )
            })
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
        if let Ok(g) = self.metadata.read()
            && let Some(m) = g.as_ref()
        {
            return Ok(m.clone());
        }

        let metadata = self.backend.metadata()?;
        if let Ok(mut g) = self.metadata.write() {
            *g = Some(metadata.clone());
        }
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
    ) -> Result<Arc<ColumnData>, BackendError>
    {
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

impl<T: ChunkedDataBackendSync>
    ChunkedDataBackendSync for Arc<T>
{
    fn read_chunk_sync(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<Arc<ColumnData>, BackendError>
    {
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
    ) -> Result<Arc<ColumnData>, BackendError>
    {
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

/// Per-cache entry counts for a chunk cache.
///
/// `coord_entries` and `var_entries` are tracked independently because
/// the two sub-caches have independent capacities.
pub struct CacheStats {
    pub coord_entries: usize,
    pub var_entries: usize,
}

impl CacheStats {
    /// Total entries across both sub-caches.
    #[allow(dead_code)]
    pub fn total_entries(&self) -> usize {
        self.coord_entries + self.var_entries
    }
}

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
        // moka batches inserts into a write buffer; flush before reading
        // entry counts so callers see the current state.
        self.coord_cache.run_pending_tasks();
        self.var_cache.run_pending_tasks();
        CacheStats {
            coord_entries: self
                .coord_cache
                .entry_count()
                as usize,
            var_entries: self
                .var_cache
                .entry_count()
                as usize,
        }
    }
    fn clear(&self) {
        self.coord_cache.invalidate_all();
        self.var_cache.invalidate_all();
    }
}

#[async_trait::async_trait]
impl<BACKEND: ChunkedDataBackendAsync>
    EvictableChunkCacheAsync
    for ChunkedDataCacheAsync<BACKEND>
{
    async fn cache_stats(&self) -> CacheStats {
        // moka batches inserts into a write buffer; flush before reading
        // entry counts so callers see the current state.
        self.coord_cache
            .run_pending_tasks()
            .await;
        self.var_cache.run_pending_tasks().await;
        CacheStats {
            coord_entries: self
                .coord_cache
                .entry_count()
                as usize,
            var_entries: self
                .var_cache
                .entry_count()
                as usize,
        }
    }
    async fn clear(&self) {
        self.coord_cache.invalidate_all();
        self.var_cache.invalidate_all();
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
