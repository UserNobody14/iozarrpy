//! Backend handler for Icechunk-backed zarr datasets (async-only).
//!
//! Icechunk provides version-controlled, transactional access to Zarr stores.
//! This module wraps `AsyncIcechunkStore` from `zarrs_icechunk` to provide
//! the same interface as `ZarrBackendAsync`.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::sync::Arc;
use tokio::sync::RwLock;

use icechunk::session::Session;
use zarrs::array::Array;
use zarrs::storage::{
    AsyncReadableWritableListableStorage,
    AsyncReadableWritableListableStorageTraits,
};
use zarrs_icechunk::AsyncIcechunkStore;

use crate::IStr;
use crate::errors::BackendError;
use crate::meta::{
    ZarrMeta, load_zarr_meta_from_store_async,
};
use crate::reader::{
    ColumnData, ShardedCacheAsync,
    retrieve_chunk_async,
};
use crate::shared::{
    ChunkedDataBackendAsync,
    ChunkedDataCacheAsync, HasAsyncStore,
    HasMetadataBackendAsync,
    HasMetadataBackendCacheAsync,
};

/// An opened array with its sharded cache for async access.
struct OpenedArrayAsync {
    array: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>,
    cache: Arc<ShardedCacheAsync>,
}

/// Async backend for Icechunk-backed zarr datasets.
///
/// Wraps an `AsyncIcechunkStore` and provides the same interface as
/// `ZarrBackendAsync`, enabling predicate pushdown and efficient scanning.
pub struct IcechunkBackendAsync {
    store: AsyncReadableWritableListableStorage,
    root: String,
    opened_arrays:
        RwLock<BTreeMap<IStr, OpenedArrayAsync>>,
}

impl IcechunkBackendAsync {
    /// Create a new Icechunk backend from a Session.
    ///
    /// # Arguments
    /// * `session` - An Icechunk Session (obtained from a Repository)
    /// * `root` - Root path within the store (typically "/")
    pub fn from_session(
        session: Session,
        root: Option<String>,
    ) -> Self {
        let icechunk_store =
            AsyncIcechunkStore::new(session);
        let store: AsyncReadableWritableListableStorage =
            Arc::new(icechunk_store);
        let root = root
            .unwrap_or_else(|| "/".to_string());
        Self {
            store,
            root,
            opened_arrays: RwLock::new(
                BTreeMap::new(),
            ),
        }
    }
}

impl HasAsyncStore for IcechunkBackendAsync {
    fn async_store(
        &self,
    ) -> &AsyncReadableWritableListableStorage
    {
        &self.store
    }
}

#[async_trait::async_trait]
impl HasMetadataBackendAsync<ZarrMeta>
    for IcechunkBackendAsync
{
    async fn metadata(
        &self,
    ) -> Result<Arc<ZarrMeta>, BackendError> {
        let meta =
            load_zarr_meta_from_store_async(
                &self.store,
                &self.root,
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
    for IcechunkBackendAsync
{
    async fn read_chunk_async(
        &self,
        var: &IStr,
        chunk_idx: &[u64],
    ) -> Result<ColumnData, BackendError> {
        let store = self.store.clone();

        // Clone the Arc values and drop the guard before await to keep the future Send
        let existing = self
            .opened_arrays
            .read()
            .await
            .get(var)
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
                BackendError::Other(e.to_string())
            })?;
            return Ok(chunk);
        }

        // Open array and create cache
        let array = Array::async_open(store, var)
            .await
            .map_err(|e| {
                BackendError::Other(e.to_string())
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
            BackendError::Other(e.to_string())
        })?;

        self.opened_arrays.write().await.insert(
            var.clone(),
            OpenedArrayAsync {
                array: array_arc,
                cache,
            },
        );

        Ok(chunk)
    }
}

impl Display for IcechunkBackendAsync {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "IcechunkBackendAsync(root='{}')",
            self.root
        )
    }
}

/// Fully cached Icechunk backend type alias.
///
/// Wraps `IcechunkBackendAsync` with metadata and chunk caching layers.
pub type FullyCachedIcechunkBackendAsync =
    HasMetadataBackendCacheAsync<
        ZarrMeta,
        ChunkedDataCacheAsync<
            IcechunkBackendAsync,
        >,
    >;

/// Convert an `IcechunkBackendAsync` to a fully cached version.
pub fn to_fully_cached_icechunk_async(
    backend: IcechunkBackendAsync,
    max_entries: u64,
) -> Result<
    FullyCachedIcechunkBackendAsync,
    BackendError,
> {
    Ok(HasMetadataBackendCacheAsync::new(
        ChunkedDataCacheAsync::new(
            backend,
            max_entries,
        ),
    ))
}
