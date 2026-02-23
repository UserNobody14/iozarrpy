use crate::IStr;
use std::collections::BTreeMap;
use tokio::sync::RwLock;
use zarrs::storage::{
    AsyncReadableWritableListableStorage,
    ReadableWritableListableStorage,
};

/// Backend handler for (non-icechunk) zarr datasets
///
use super::traits::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync,
    ChunkedDataCacheAsync, ChunkedDataCacheSync,
    HasAsyncStore, HasMetadataBackendAsync,
    HasMetadataBackendCacheAsync,
    HasMetadataBackendCacheSync,
    HasMetadataBackendSync, HasStore,
};
use crate::errors::{
    BackendError, BackendResult,
};
use crate::meta::{
    ZarrMeta, load_zarr_meta_from_opened,
    load_zarr_meta_from_opened_async,
};
use crate::reader::{
    ColumnData, retrieve_chunk,
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
    /// Opened arrays with their shard caches.
    /// Uses parking_lot (not tokio) to avoid deadlocks when
    /// the underlying store wraps an async runtime via TokioBlockOn.
    opened_arrays: parking_lot::RwLock<
        BTreeMap<IStr, Arc<OpenedArraySync>>,
    >,
    /// Cached ZarrMeta (once loaded)
    cached_meta: parking_lot::RwLock<
        Option<Arc<ZarrMeta>>,
    >,
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
    ) -> BackendResult<Self> {
        let opened = store.open_sync()?;
        Ok(Self {
            store: Arc::new(opened),
            opened_arrays:
                parking_lot::RwLock::new(
                    BTreeMap::new(),
                ),
            cached_meta: parking_lot::RwLock::new(
                None,
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
        // Check if already cached
        {
            let cached = self.cached_meta.read();
            if let Some(meta) = cached.as_ref() {
                return Ok(meta.clone());
            }
        }

        // Load and cache metadata
        let meta = load_zarr_meta_from_opened(
            &self.store,
        )?;
        let meta_arc = Arc::new(meta);

        *self.cached_meta.write() =
            Some(meta_arc.clone());

        Ok(meta_arc)
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
            .read()
            .get(var)
            .cloned();

        // Open array and create cache
        let opened = match array_opt {
            Some(opened) => opened,
            None => {
                // Get array metadata from cache if available
                let array_metadata = {
                    let cached =
                        self.cached_meta.read();
                    cached.as_ref().and_then(|meta| {
                        meta.array_by_path(var.clone()).and_then(|arr_meta| {
                            arr_meta.array_metadata.clone()
                        })
                    })
                };

                let opened_inner = Arc::new(
                    self.store
                        .open_array_and_cache(
                            var,
                            array_metadata
                                .as_ref()
                                .map(|a| {
                                    a.as_ref()
                                }),
                        )?,
                );
                self.opened_arrays
                    .write()
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
        )?;

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
    /// Cached ZarrMeta (once loaded)
    cached_meta: RwLock<Option<Arc<ZarrMeta>>>,
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
    ) -> BackendResult<Self> {
        let opened = store.open_async()?;
        Ok(Self {
            store: Arc::new(opened),
            opened_arrays: RwLock::new(
                BTreeMap::new(),
            ),
            cached_meta: RwLock::new(None),
        })
    }
}

#[async_trait::async_trait]
impl HasMetadataBackendAsync<ZarrMeta>
    for ZarrBackendAsync
{
    async fn metadata(
        &self,
    ) -> BackendResult<Arc<ZarrMeta>> {
        // Check if already cached
        {
            let cached =
                self.cached_meta.read().await;
            if let Some(meta) = cached.as_ref() {
                return Ok(meta.clone());
            }
        }

        // Load and cache metadata
        let meta =
            load_zarr_meta_from_opened_async(
                &self.store,
            )
            .await?;
        let meta_arc = Arc::new(meta);

        *self.cached_meta.write().await =
            Some(meta_arc.clone());

        Ok(meta_arc)
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
                    // Get array metadata from cache if available
                    let array_metadata = {
                        let cached = self
                            .cached_meta
                            .read()
                            .await;
                        cached.as_ref().and_then(
                            |meta| {
                                meta.array_by_path(var.clone())
                                    .and_then(
                                        |arr_meta| {
                                            arr_meta
                                            .array_metadata
                                            .clone()
                                        },
                                    )
                            },
                        )
                    };

                    let opened_inner = Arc::new(
                        self.store
                            .open_array_and_cache(
                                var,
                                array_metadata
                                    .as_ref()
                                    .map(|a| {
                                        a.as_ref()
                                    }),
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
        .await?;

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
