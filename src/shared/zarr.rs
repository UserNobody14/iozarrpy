use crate::shared::traits::HasStats;
use crate::{IStr, PlannerStats};
use ambassador::delegatable_trait;
use std::collections::BTreeMap;
use tokio::sync::RwLock;
use zarrs::array::Array;
use zarrs::storage::{
    AsyncReadableWritableListableStorage,
    ReadableWritableListableStorage,
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
use crate::store::{
    OpenedArrayAsync, OpenedArraySync,
};
use std::fmt::Display;
use std::sync::Arc;

pub struct ZarrBackendSync {
    store: Arc<OpenedStore>,
    /// Opened arrays with their shard caches:
    opened_arrays: RwLock<
        BTreeMap<IStr, Arc<OpenedArraySync>>,
    >,
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
        // Try to get existing array and cache
        let array_opt = self
            .opened_arrays
            .blocking_read()
            .get(var)
            .cloned();

        // Open array and create cache
        let opened = match array_opt {
            Some(opened) => opened,
            None => {
                let opened_inner = Arc::new(
                    self.store
                        .open_array_and_cache(
                            var,
                        )?,
                );
                self.opened_arrays
                    .blocking_write()
                    .insert(
                        var.clone(),
                        opened_inner.clone(),
                    );
                opened_inner
            }
        };

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
    opened_arrays: RwLock<
        BTreeMap<IStr, Arc<OpenedArrayAsync>>,
    >,
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
        // Clone the Arc values and drop the guard before await to keep the future Send
        let existing = self
            .opened_arrays
            .read()
            .await
            .get(var)
            .cloned();

        let opened: Arc<OpenedArrayAsync> =
            match existing {
                Some(opened) => opened.clone(),
                None => {
                    let opened_inner = Arc::new(
                        self.store
                            .open_array_and_cache(
                                var,
                            )
                            .await?,
                    );
                    self.opened_arrays
                        .write()
                        .await
                        .insert(
                            var.clone(),
                            opened_inner.clone(),
                        );
                    opened_inner
                }
            };
        let chunk = retrieve_chunk_async(
            opened.array.as_ref(),
            opened.cache.as_ref(),
            chunk_idx,
        )
        .await
        .map_err(|e| {
            BackendError::ChunkReadFailed(
                e.to_string(),
            )
        })?;

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
    max_entries: u64,
) -> Result<
    FullyCachedZarrBackendSync,
    BackendError,
> {
    Ok(HasMetadataBackendCacheSync::new(
        ChunkedDataCacheSync::new(
            backend,
            max_entries,
        ),
    ))
}

pub fn to_fully_cached_async(
    backend: ZarrBackendAsync,
    max_entries: u64,
) -> Result<
    FullyCachedZarrBackendAsync,
    BackendError,
> {
    Ok(HasMetadataBackendCacheAsync::new(
        ChunkedDataCacheAsync::new(
            backend,
            max_entries,
        ),
    ))
}
