use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::runtime::Runtime;
use url::Url;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncStorageAdapter;
use zarrs::storage::{
    AsyncReadableWritableListableStorage,
    ReadableWritableListableStorage,
};
use zarrs::array::Array;
use zarrs_object_store::AsyncObjectStore;
use zarrs_object_store::object_store;
use zarrs_object_store::object_store::ObjectStore;

use crate::IStr;
use crate::errors::BackendError;
use crate::reader::ShardedCacheSync;
use crate::shared::normalize_path;

use crate::errors::BackendResult;
use crate::store::adapters::TokioBlockOn;
use crate::store::array::OpenedArraySync;

pub struct OpenedStore {
    pub store: ReadableWritableListableStorage,
    /// Root node path to pass to zarrs open functions (always starts with `/`).
    pub root: String,
}

impl OpenedStore {
    pub fn open_array_and_cache(
        &self,
        var: &IStr,
        array_metadata: Option<
            &zarrs::array::ArrayMetadata,
        >,
    ) -> Result<OpenedArraySync, BackendError>
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
            Array::open(strtraits, &norm)
        }
        .map_err(|e| {
            BackendError::ArrayOpenFailed(
                e.to_string(),
            )
        })?;
        let array_arc = Arc::new(array);
        let cache = ShardedCacheSync::new(
            array_arc.as_ref(),
        );
        Ok(OpenedArraySync::new(array_arc, cache))
    }
}

/// Create a sync store from an already-constructed `Arc<dyn ObjectStore>`.
///
/// This is used when the user passes an obstore instance directly instead of a URL.
pub fn open_store_from_object_store(
    store: Arc<dyn ObjectStore>,
    prefix: Option<String>,
) -> BackendResult<OpenedStore> {
    if let Some(ref p) = prefix {
        let path = Path::new(p);
        if path.is_absolute() && path.exists() {
            return open_filesystem_store(p);
        }
        if !path.is_absolute() {
            let abs = format!("/{p}");
            if Path::new(&abs).exists() {
                return open_filesystem_store(
                    &abs,
                );
            }
        }
    }
    let async_store: AsyncReadableWritableListableStorage =
        Arc::new(AsyncObjectStore::new(store));
    let rt = Arc::new(Runtime::new().map_err(|e| {
        BackendError::Other(format!("failed to create tokio runtime: {e}"))
    })?);
    let sync_store =
        AsyncToSyncStorageAdapter::new(
            async_store,
            TokioBlockOn(rt),
        );

    let root = prefix
        .map(|p| {
            if p.starts_with('/') {
                p
            } else {
                format!("/{p}")
            }
        })
        .unwrap_or_else(|| "/".to_string());

    Ok(OpenedStore {
        store: Arc::new(sync_store),
        root,
    })
}

/// Open a Zarr store from a URL or filesystem path.
///
/// Supported URL schemes are those implemented by `object_store::parse_url`, including:
/// `file://`, `s3://`, `gs://`, `az://`, `abfs://`, `http://`, and `https://`.
///
/// If `zarr_url` is not a valid URL, it is treated as a local filesystem path.
pub fn open_store(
    zarr_url: &str,
) -> BackendResult<OpenedStore> {
    if !zarr_url.contains("://") {
        return open_filesystem_store(zarr_url);
    }

    let url = Url::parse(zarr_url)?;

    if url.scheme() == "file" {
        let path =
            url.to_file_path().map_err(|_| {
                BackendError::Other(
                    "invalid file:// URL"
                        .to_string(),
                )
            })?;
        let path =
            path.to_string_lossy().to_string();
        return open_filesystem_store(&path);
    }

    let (store, prefix) =
        object_store::parse_url(&url)?;

    let async_store: AsyncReadableWritableListableStorage =
        Arc::new(AsyncObjectStore::new(store));
    let rt = Arc::new(Runtime::new().map_err(|e| {
        BackendError::Other(format!("failed to create tokio runtime: {e}"))
    })?);
    let sync_store =
        AsyncToSyncStorageAdapter::new(
            async_store,
            TokioBlockOn(rt),
        );

    let root = if prefix.as_ref().is_empty() {
        "/".to_string()
    } else {
        format!("/{}", prefix.as_ref())
    };

    Ok(OpenedStore {
        store: Arc::new(sync_store),
        root,
    })
}

fn open_filesystem_store(
    path: &str,
) -> BackendResult<OpenedStore> {
    let store_path: PathBuf = path.into();
    let store =
        zarrs::filesystem::FilesystemStore::new(
            &store_path,
        )?;
    Ok(OpenedStore {
        store: Arc::new(store),
        root: "/".to_string(),
    })
}
