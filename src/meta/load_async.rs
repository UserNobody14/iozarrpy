use zarrs::storage::AsyncReadableWritableListableStorage;

use crate::errors::BackendResult;
use crate::meta::shared::load_zarr_meta_inner;
use crate::meta::types::ZarrMeta;
use crate::store::AsyncOpenedStore;

// =============================================================================
// Unified Hierarchical Metadata Loading
// =============================================================================

/// Load unified metadata that supports both flat and hierarchical zarr stores.
pub async fn load_zarr_meta_from_opened_async(
    opened: &AsyncOpenedStore,
) -> BackendResult<ZarrMeta> {
    let store = opened.store.clone();
    let root_path = opened.root.clone();
    let root_path_str: &str = root_path.as_ref();

    let group = zarrs::group::Group::async_open(
        store.clone(),
        &root_path,
    )
    .await?;
    let nodes = group.async_traverse().await?;
    load_zarr_meta_inner(
        &store,
        &nodes,
        &root_path_str,
    )
}

/// Load unified metadata from a raw async store and root path.
///
/// This is useful for backends like Icechunk that don't use `AsyncOpenedStore`.
pub async fn load_zarr_meta_from_store_async(
    store: &AsyncReadableWritableListableStorage,
    root_path: &str,
) -> BackendResult<ZarrMeta> {
    let store = store.clone();

    let group = zarrs::group::Group::async_open(
        store.clone(),
        root_path,
    )
    .await?;
    let nodes = group.async_traverse().await?;

    load_zarr_meta_inner(
        &store, &nodes, root_path,
    )
}
