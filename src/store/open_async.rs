use std::path::{Path, PathBuf};
use std::sync::Arc;

use url::Url;
use zarrs::storage::storage_adapter::sync_to_async::SyncToAsyncStorageAdapter;
use zarrs::storage::{
    AsyncReadableWritableListableStorage,
    ReadableWritableListableStorage,
};
use zarrs_object_store::AsyncObjectStore;
use zarrs_object_store::object_store;
use zarrs_object_store::object_store::ObjectStore;

use crate::IStr;
use crate::errors::BackendError;
use crate::reader::ShardedCacheAsync;
use crate::shared::normalize_path;
use crate::store::adapters::TokioSpawnBlocking;
use crate::store::array::OpenedArrayAsync;
use zarrs::array::Array;

pub struct AsyncOpenedStore {
    pub store:
        AsyncReadableWritableListableStorage,
    /// Root node path to pass to zarrs open functions (always starts with `/`).
    pub root: String,
}

impl AsyncOpenedStore {
    pub async fn open_array_and_cache(
        &self,
        var: &IStr,
        array_metadata: Option<
            &zarrs::array::ArrayMetadata,
        >,
    ) -> Result<OpenedArrayAsync, BackendError>
    {
        let strtraits = self.store.clone();
        let norm = normalize_path(var);

        let array = if let Some(metadata) =
            array_metadata
        {
            Array::new_with_metadata(
                strtraits,
                &norm,
                metadata.clone(),
            )
        } else {
            Array::async_open(strtraits, &norm)
                .await
        }
        .map_err(|e| {
            BackendError::ArrayOpenFailed(
                e.to_string(),
            )
        })?;
        let array_arc = Arc::new(array);
        let cache = ShardedCacheAsync::new(
            array_arc.as_ref(),
        );
        Ok(OpenedArrayAsync::new(
            array_arc,
            Arc::new(cache),
        ))
    }
}

/// Create an async store from an already-constructed `Arc<dyn ObjectStore>`.
///
/// This is used when the user passes an obstore instance directly instead of a URL.
pub fn open_store_from_object_store_async(
    store: Arc<dyn ObjectStore>,
    prefix: Option<String>,
) -> AsyncOpenedStore {
    if let Some(ref p) = prefix {
        let path = Path::new(p);
        if path.is_absolute() && path.exists() {
            if let Ok(opened) =
                open_filesystem_store_async(p)
            {
                return opened;
            }
        }
        if !path.is_absolute() {
            let abs = format!("/{p}");
            if Path::new(&abs).exists() {
                if let Ok(opened) =
                    open_filesystem_store_async(
                        &abs,
                    )
                {
                    return opened;
                }
            }
        }
    }
    let async_store: AsyncReadableWritableListableStorage =
        Arc::new(AsyncObjectStore::new(store));
    let root = prefix
        .map(|p| {
            if p.starts_with('/') {
                p
            } else {
                format!("/{p}")
            }
        })
        .unwrap_or_else(|| "/".to_string());
    AsyncOpenedStore {
        store: async_store,
        root,
    }
}

/// Async-first store opener.
///
/// For remote/object-store URLs this is fully async.
/// For local filesystem paths, this wraps the sync filesystem store with a spawn_blocking adapter.
pub fn open_store_async(
    zarr_url: &str,
) -> Result<AsyncOpenedStore, String> {
    if !zarr_url.contains("://") {
        return open_filesystem_store_async(
            zarr_url,
        );
    }

    let url = Url::parse(zarr_url)
        .map_err(|e| e.to_string())?;

    if url.scheme() == "file" {
        let path =
            url.to_file_path().map_err(|_| {
                "invalid file:// URL".to_string()
            })?;
        let path =
            path.to_string_lossy().to_string();
        return open_filesystem_store_async(
            &path,
        );
    }

    let (store, prefix) =
        object_store::parse_url(&url)
            .map_err(|e| e.to_string())?;
    let async_store: AsyncReadableWritableListableStorage =
        Arc::new(AsyncObjectStore::new(store));

    let root = if prefix.as_ref().is_empty() {
        "/".to_string()
    } else {
        format!("/{}", prefix.as_ref())
    };

    Ok(AsyncOpenedStore {
        store: async_store,
        root,
    })
}

fn open_filesystem_store_async(
    path: &str,
) -> Result<AsyncOpenedStore, String> {
    let store_path: PathBuf = path.into();
    let store =
        zarrs::filesystem::FilesystemStore::new(
            &store_path,
        )
        .map_err(|e| e.to_string())?;
    let async_store: AsyncReadableWritableListableStorage =
        Arc::new(SyncToAsyncStorageAdapter::new(
            Arc::new(store)
                as ReadableWritableListableStorage,
            TokioSpawnBlocking,
        ));
    Ok(AsyncOpenedStore {
        store: async_store,
        root: "/".to_string(),
    })
}
