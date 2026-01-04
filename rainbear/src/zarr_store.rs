use std::path::PathBuf;
use std::sync::Arc;

use tokio::runtime::Runtime;
use url::Url;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncStorageAdapter;
use zarrs::storage::{AsyncReadableWritableListableStorage, ReadableWritableListableStorage};
use zarrs_object_store::object_store;
use zarrs_object_store::AsyncObjectStore;

struct TokioBlockOn(Arc<Runtime>);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

pub struct OpenedStore {
    pub store: ReadableWritableListableStorage,
    /// Root node path to pass to zarrs open functions (always starts with `/`).
    pub root: String,
}

/// Open a Zarr store from a URL or filesystem path.
///
/// Supported URL schemes are those implemented by `object_store::parse_url`, including:
/// `file://`, `s3://`, `gs://`, `az://`, `abfs://`, `http://`, and `https://`.
///
/// If `zarr_url` is not a valid URL, it is treated as a local filesystem path.
pub fn open_store(zarr_url: &str) -> Result<OpenedStore, String> {
    // Fast path: treat as local path if it doesn't look like a URL.
    if !zarr_url.contains("://") {
        return open_filesystem_store(zarr_url);
    }

    let url = Url::parse(zarr_url).map_err(|e| e.to_string())?;

    // Special-case `file://` so that plain paths and file URLs behave the same.
    if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| "invalid file:// URL".to_string())?;
        let path = path.to_string_lossy().to_string();
        return open_filesystem_store(&path);
    }

    // Use object_store for all other schemes (s3/gs/az/http/https/...).
    // `parse_url` returns the store and a path prefix within that store.
    let (store, prefix) = object_store::parse_url(&url).map_err(|e| e.to_string())?;

    // Wrap object_store with zarrs' async storage adapter, then adapt to sync for Polars' IO plugin.
    let async_store: AsyncReadableWritableListableStorage = Arc::new(AsyncObjectStore::new(store));
    let rt = Arc::new(
        Runtime::new().map_err(|e| format!("failed to create tokio runtime: {e}"))?,
    );
    let sync_store = AsyncToSyncStorageAdapter::new(async_store, TokioBlockOn(rt));

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

fn open_filesystem_store(path: &str) -> Result<OpenedStore, String> {
    let store_path: PathBuf = path.into();
    let store = zarrs::filesystem::FilesystemStore::new(&store_path).map_err(|e| e.to_string())?;
    Ok(OpenedStore {
        store: Arc::new(store),
        root: "/".to_string(),
    })
}

