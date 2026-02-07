use crate::shared::traits::HasStats;
use crate::{IStr, PlannerStats};
use ambassador::delegatable_trait;
use std::collections::BTreeMap;
use std::sync::RwLock;
use zarrs::array::Array;
use zarrs::storage::{
    AsyncReadableWritableListableStorage,
    AsyncReadableWritableListableStorageTraits,
    ReadableWritableListableStorage,
    ReadableWritableListableStorageTraits,
};

/// Backend handler for (non-icechunk) zarr datasets
///
use super::traits::{
    BackendError, ChunkedDataBackendAsync,
    ChunkedDataBackendSync,
    ChunkedDataCacheAsync, ChunkedDataCacheSync,
    HasAsyncStore, HasMetadataBackendAsync,
    HasMetadataBackendCacheAsync,
    HasMetadataBackendCacheSync,
    HasMetadataBackendSync, HasStore,
};
use crate::meta::{
    ZarrMeta, load_zarr_meta_from_opened,
    load_zarr_meta_from_opened_async,
};
use crate::reader::{
    ColumnData, ShardedCacheAsync,
    ShardedCacheSync, retrieve_chunk,
    retrieve_chunk_async,
};
use crate::store::{
    AsyncOpenedStore, OpenedStore, StoreInput,
};
use std::fmt::Display;
use std::sync::Arc;

/// An opened array with its sharded cache for sync access.
struct OpenedArraySync {
    array: Arc<Array<dyn ReadableWritableListableStorageTraits>>,
    cache: ShardedCacheSync,
}

/// An opened array with its sharded cache for async access.
struct OpenedArrayAsync {
    array: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>,
    cache: Arc<ShardedCacheAsync>,
}

pub struct ZarrBackendSync {
    store: Arc<OpenedStore>,
    /// Opened arrays with their shard caches:
    opened_arrays:
        RwLock<BTreeMap<IStr, OpenedArraySync>>,
    stats: RwLock<PlannerStats>,
}

// Normalize path: if it doesn't start with '/', add it
pub(crate) fn normalize_path(
    path: &IStr,
) -> IStr {
    if path.starts_with('/') {
        path.clone()
    } else {
        format!("/{}", path).into()
    }
}

impl ZarrBackendSync {
    pub fn new(
        store: StoreInput,
    ) -> Result<Self, BackendError> {
        let opened =
            store.open_sync().map_err(|e| {
                BackendError::Other(e.to_string())
            })?;
        Ok(Self {
            store: Arc::new(opened),
            opened_arrays: RwLock::new(
                BTreeMap::new(),
            ),
            stats: RwLock::new(
                PlannerStats::default(),
            ),
        })
    }
}

impl HasStore for ZarrBackendSync {
    fn store(
        &self,
    ) -> &ReadableWritableListableStorage {
        &self.store.as_ref().store
    }
}

impl HasMetadataBackendSync<ZarrMeta>
    for ZarrBackendSync
{
    fn metadata(
        &self,
    ) -> Result<Arc<ZarrMeta>, BackendError> {
        // Called underneath the cache, so we don't need to check if the store is loaded
        // Instead we just load the metadata, caching is handled above
        // by the cache wrapper
        let meta = load_zarr_meta_from_opened(
            &self.store,
        )
        .map_err(|e| {
            BackendError::Other(e.to_string())
        })?;
        Ok(Arc::new(meta))
    }
}

impl ChunkedDataBackendSync for ZarrBackendSync {
    fn read_chunk_sync(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<ColumnData, BackendError> {
        let strtraits =
            self.store.as_ref().store.clone();

        // Try to get existing array and cache
        if let Some(opened) = self
            .opened_arrays
            .read()
            .unwrap()
            .get(var)
        {
            let chunk = retrieve_chunk(
                &opened.array,
                &opened.cache,
                chunk_idx,
            )
            .map_err(|e| {
                BackendError::ChunkReadFailed(
                    e.to_string(),
                )
            })?;
            return Ok(chunk);
        }

        // Open array and create cache
        let array = Array::open(
            strtraits,
            &normalize_path(var),
        )
        .map_err(|e| {
            BackendError::ArrayOpenFailed(
                e.to_string(),
            )
        })?;
        let array_arc = Arc::new(array);
        let cache = ShardedCacheSync::new(
            array_arc.as_ref(),
        );

        let chunk = retrieve_chunk(
            array_arc.as_ref(),
            &cache,
            chunk_idx,
        )
        .map_err(|e| {
            BackendError::ChunkReadFailed(
                e.to_string(),
            )
        })?;

        self.opened_arrays
            .write()
            .unwrap()
            .insert(
                var.clone(),
                OpenedArraySync {
                    array: array_arc,
                    cache,
                },
            );

        Ok(chunk)
    }
}

impl Display for ZarrBackendSync {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "ZarrBackendSync(root='{}')",
            self.store.as_ref().root.to_string()
        )
    }
}

pub struct ZarrBackendAsync {
    store: Arc<AsyncOpenedStore>,
    opened_arrays:
        RwLock<BTreeMap<IStr, OpenedArrayAsync>>,
    stats: RwLock<PlannerStats>,
}

impl HasAsyncStore for ZarrBackendAsync {
    fn async_store(
        &self,
    ) -> &AsyncReadableWritableListableStorage
    {
        &self.store.as_ref().store
    }
}

impl ZarrBackendAsync {
    pub fn new(
        store: StoreInput,
    ) -> Result<Self, BackendError> {
        let opened =
            store.open_async().map_err(|e| {
                BackendError::Other(e.to_string())
            })?;
        Ok(Self {
            store: Arc::new(opened),
            opened_arrays: RwLock::new(
                BTreeMap::new(),
            ),
            stats: RwLock::new(
                PlannerStats::default(),
            ),
        })
    }
}

#[async_trait::async_trait]
impl HasMetadataBackendAsync<ZarrMeta>
    for ZarrBackendAsync
{
    async fn metadata(
        &self,
    ) -> Result<Arc<ZarrMeta>, BackendError> {
        let meta =
            load_zarr_meta_from_opened_async(
                &self.store,
            )
            .await
            .map_err(|e| {
                BackendError::Other(e.to_string())
            })?;
        Ok(Arc::new(meta))
    }
}

#[async_trait::async_trait]
impl ChunkedDataBackendAsync
    for ZarrBackendAsync
{
    async fn read_chunk_async(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<ColumnData, BackendError> {
        let strtraits =
            self.store.as_ref().store.clone();

        // Clone the Arc values and drop the guard before await to keep the future Send
        let existing = self
            .opened_arrays
            .read()
            .unwrap()
            .get(&var)
            .map(|opened| {
                (
                    opened.array.clone(),
                    opened.cache.clone(),
                )
            });

        if let Some((array, cache)) = existing {
            let chunk = retrieve_chunk_async(
                &array, &cache, chunk_idx,
            )
            .await
            .map_err(|e| {
                BackendError::ChunkReadFailed(
                    e.to_string(),
                )
            })?;
            return Ok(chunk);
        }

        // Open array and create cache
        let array = Array::async_open(
            strtraits,
            &normalize_path(var),
        )
        .await
        .map_err(|e| {
            BackendError::ArrayOpenFailed(
                e.to_string(),
            )
        })?;
        let array_arc = Arc::new(array);
        let cache =
            Arc::new(ShardedCacheAsync::new(
                array_arc.as_ref(),
            ));

        let chunk = retrieve_chunk_async(
            array_arc.as_ref(),
            &cache,
            chunk_idx,
        )
        .await
        .map_err(|e| {
            BackendError::ChunkReadFailed(
                e.to_string(),
            )
        })?;

        self.opened_arrays
            .write()
            .unwrap()
            .insert(
                var.clone(),
                OpenedArrayAsync {
                    array: array_arc,
                    cache,
                },
            );

        Ok(chunk)
    }
}

impl Display for ZarrBackendAsync {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "ZarrBackendAsync(root='{}')",
            self.store.as_ref().root.to_string()
        )
    }
}

pub type FullyCachedZarrBackendSync =
    HasMetadataBackendCacheSync<
        ZarrMeta,
        ChunkedDataCacheSync<ZarrBackendSync>,
    >;
pub type FullyCachedZarrBackendAsync =
    HasMetadataBackendCacheAsync<
        ZarrMeta,
        ChunkedDataCacheAsync<ZarrBackendAsync>,
    >;

pub fn to_fully_cached_sync(
    backend: ZarrBackendSync,
) -> Result<
    FullyCachedZarrBackendSync,
    BackendError,
> {
    Ok(HasMetadataBackendCacheSync::new(
        ChunkedDataCacheSync::new(backend),
    ))
}

pub fn to_fully_cached_async(
    backend: ZarrBackendAsync,
) -> Result<
    FullyCachedZarrBackendAsync,
    BackendError,
> {
    Ok(HasMetadataBackendCacheAsync::new(
        ChunkedDataCacheAsync::new(backend),
    ))
}
