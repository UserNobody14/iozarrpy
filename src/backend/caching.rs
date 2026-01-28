//! Caching backend implementation.
//!
//! Provides persistent caching of coordinate array chunks and metadata
//! across multiple scan operations.

use std::collections::BTreeMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use zarrs::array::Array;
use zarrs::storage::AsyncReadableWritableListableStorage;

use super::traits::{
    BackendError, CoordChunkData,
    ZarrBackendAsync,
};
use crate::IStr;
use crate::meta::ZarrDatasetMeta;

/// Cache key for coordinate chunks: (dimension_name, chunk_index)
type CoordCacheKey = (IStr, u64);

/// Async Zarr backend with persistent caching.
///
/// Caches:
/// - Dataset metadata (loaded once, reused across scans)
/// - Coordinate array chunks (chunk-level caching for efficient binary search)
///
/// Thread-safe via `tokio::sync::RwLock`.
pub struct CachingAsyncBackend {
    store: AsyncReadableWritableListableStorage,
    root: String,
    /// Cached metadata, loaded lazily on first access.
    metadata_cache:
        RwLock<Option<Arc<ZarrDatasetMeta>>>,
    /// Cached coordinate chunks.
    coord_cache: RwLock<
        BTreeMap<CoordCacheKey, CoordChunkData>,
    >,
    /// Maximum number of coord cache entries (0 = unlimited).
    max_cache_entries: usize,
}

impl CachingAsyncBackend {
    /// Create a new caching backend.
    ///
    /// # Arguments
    /// * `store` - The underlying async store
    /// * `root` - Root path within the store (e.g., "/" or "/dataset.zarr")
    /// * `max_cache_entries` - Maximum cached coord chunks (0 = unlimited)
    pub fn new(
        store: AsyncReadableWritableListableStorage,
        root: String,
        max_cache_entries: usize,
    ) -> Self {
        Self {
            store,
            root,
            metadata_cache: RwLock::new(None),
            coord_cache: RwLock::new(
                BTreeMap::new(),
            ),
            max_cache_entries,
        }
    }

    /// Create a caching backend with default settings.
    pub fn with_defaults(
        store: AsyncReadableWritableListableStorage,
        root: String,
    ) -> Self {
        Self::new(store, root, 0) // unlimited cache by default
    }

    /// Clear the coordinate cache.
    pub async fn clear_coord_cache(&self) {
        let mut cache =
            self.coord_cache.write().await;
        cache.clear();
    }

    /// Clear all caches (metadata and coordinates).
    pub async fn clear_all_caches(&self) {
        {
            let mut meta =
                self.metadata_cache.write().await;
            *meta = None;
        }
        self.clear_coord_cache().await;
    }

    /// Get cache statistics.
    pub async fn cache_stats(
        &self,
    ) -> CacheStats {
        let coord_entries =
            self.coord_cache.read().await.len();
        let has_metadata = self
            .metadata_cache
            .read()
            .await
            .is_some();
        CacheStats {
            coord_entries,
            has_metadata,
        }
    }

    /// Internal: Load metadata from store.
    async fn load_metadata_impl(
        &self,
    ) -> Result<Arc<ZarrDatasetMeta>, BackendError>
    {
        use crate::meta::load_zarr_meta_from_opened_async;
        use crate::store::AsyncOpenedStore;

        let opened = AsyncOpenedStore {
            store: self.store.clone(),
            root: self.root.clone(),
        };

        let zarr_meta =
            load_zarr_meta_from_opened_async(
                &opened,
            )
            .await
            .map_err(|e| {
                BackendError::Other(e)
            })?;

        // Convert to ZarrDatasetMeta - preserves hierarchical paths from path_to_array
        let meta =
            ZarrDatasetMeta::from(&zarr_meta);

        Ok(Arc::new(meta))
    }

    /// Internal: Load a coordinate chunk from store.
    async fn load_coord_chunk_impl(
        &self,
        dim: &IStr,
        chunk_idx: u64,
        meta: &ZarrDatasetMeta,
    ) -> Result<CoordChunkData, BackendError>
    {
        let array_meta = meta
            .arrays
            .get(dim)
            .ok_or_else(|| {
                BackendError::CoordNotFound(
                    dim.to_string(),
                )
            })?;

        let arr: Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits> =
            Array::async_open(self.store.clone(), &array_meta.path)
                .await
                .map_err(|e| BackendError::ArrayOpenFailed(e.to_string()))?;

        let chunk_indices = vec![chunk_idx];
        let dtype_id =
            arr.data_type().identifier();

        let data = match dtype_id {
            "float64" => {
                let values = arr
                    .async_retrieve_chunk::<Vec<f64>>(
                        &chunk_indices,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::ChunkReadFailed(
                            e.to_string(),
                        )
                    })?;
                CoordChunkData::F64(values)
            }
            "float32" => {
                let values: Vec<f64> = arr
                    .async_retrieve_chunk::<Vec<f32>>(
                        &chunk_indices,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::ChunkReadFailed(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                CoordChunkData::F64(values)
            }
            "int64" => {
                let values = arr
                    .async_retrieve_chunk::<Vec<i64>>(
                        &chunk_indices,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::ChunkReadFailed(
                            e.to_string(),
                        )
                    })?;
                CoordChunkData::I64(values)
            }
            "int32" => {
                let values: Vec<i64> = arr
                    .async_retrieve_chunk::<Vec<i32>>(
                        &chunk_indices,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::ChunkReadFailed(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                CoordChunkData::I64(values)
            }
            "int16" => {
                let values: Vec<i64> = arr
                    .async_retrieve_chunk::<Vec<i16>>(
                        &chunk_indices,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::ChunkReadFailed(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                CoordChunkData::I64(values)
            }
            "int8" => {
                let values: Vec<i64> = arr
                    .async_retrieve_chunk::<Vec<i8>>(
                        &chunk_indices,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::ChunkReadFailed(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                CoordChunkData::I64(values)
            }
            "uint64" => {
                let values = arr
                    .async_retrieve_chunk::<Vec<u64>>(
                        &chunk_indices,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::ChunkReadFailed(
                            e.to_string(),
                        )
                    })?;
                CoordChunkData::U64(values)
            }
            "uint32" => {
                let values: Vec<u64> = arr
                    .async_retrieve_chunk::<Vec<u32>>(
                        &chunk_indices,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::ChunkReadFailed(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as u64)
                    .collect();
                CoordChunkData::U64(values)
            }
            "uint16" => {
                let values: Vec<u64> = arr
                    .async_retrieve_chunk::<Vec<u16>>(
                        &chunk_indices,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::ChunkReadFailed(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as u64)
                    .collect();
                CoordChunkData::U64(values)
            }
            "uint8" => {
                let values: Vec<u64> = arr
                    .async_retrieve_chunk::<Vec<u8>>(
                        &chunk_indices,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::ChunkReadFailed(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as u64)
                    .collect();
                CoordChunkData::U64(values)
            }
            other => {
                return Err(BackendError::Other(
                    format!(
                        "unsupported coordinate dtype: {}",
                        other
                    ),
                ));
            }
        };

        Ok(data)
    }

    /// Evict oldest entries if cache is over limit.
    async fn maybe_evict(&self) {
        if self.max_cache_entries == 0 {
            return; // unlimited
        }

        let mut cache =
            self.coord_cache.write().await;
        while cache.len() > self.max_cache_entries
        {
            // BTreeMap doesn't have pop_first on all Rust versions, use remove on first key
            if let Some(key) =
                cache.keys().next().cloned()
            {
                cache.remove(&key);
            } else {
                break;
            }
        }
    }
}

#[async_trait::async_trait]
impl ZarrBackendAsync for CachingAsyncBackend {
    fn metadata(
        &self,
    ) -> Option<Arc<ZarrDatasetMeta>> {
        // Try to get without blocking - this is best-effort
        // For guaranteed access, use load_metadata()
        match self.metadata_cache.try_read() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        }
    }

    async fn load_metadata(
        &self,
    ) -> Result<Arc<ZarrDatasetMeta>, BackendError>
    {
        // Check if already cached
        {
            let cache =
                self.metadata_cache.read().await;
            if let Some(ref meta) = *cache {
                return Ok(meta.clone());
            }
        }

        // Load and cache
        let meta =
            self.load_metadata_impl().await?;
        {
            let mut cache =
                self.metadata_cache.write().await;
            *cache = Some(meta.clone());
        }

        Ok(meta)
    }

    async fn read_coord_chunk(
        &self,
        dim: &IStr,
        chunk_idx: u64,
    ) -> Result<CoordChunkData, BackendError>
    {
        let cache_key = (dim.clone(), chunk_idx);

        // Check cache first
        {
            let cache =
                self.coord_cache.read().await;
            if let Some(data) =
                cache.get(&cache_key)
            {
                return Ok(data.clone());
            }
        }

        // Need to load - first ensure metadata is available
        let meta = self.load_metadata().await?;

        // Load chunk
        let data = self
            .load_coord_chunk_impl(
                dim, chunk_idx, &meta,
            )
            .await?;

        // Cache it
        {
            let mut cache =
                self.coord_cache.write().await;
            cache.insert(cache_key, data.clone());
        }

        // Maybe evict old entries
        self.maybe_evict().await;

        Ok(data)
    }

    fn async_store(
        &self,
    ) -> AsyncReadableWritableListableStorage
    {
        self.store.clone()
    }

    fn root(&self) -> &str {
        &self.root
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub coord_entries: usize,
    pub has_metadata: bool,
}
