use std::sync::Arc;

use crate::errors::BackendResult;
use crate::meta::shared::load_zarr_meta_inner;

use crate::meta::types::ZarrMeta;
use crate::store::OpenedStore;

// =============================================================================
// Unified Hierarchical Metadata Loading (Sync)
// =============================================================================

/// Load unified metadata that supports both flat and hierarchical zarr stores (sync).
pub fn load_zarr_meta_from_opened(
    opened: &OpenedStore,
) -> BackendResult<ZarrMeta> {
    crate::codec_compat::ensure_zarr_compat_registered();

    let store = Arc::clone(&opened.store);
    let root_path = opened.root.clone();
    let root_path_str: &str = root_path.as_ref();

    let group = zarrs::group::Group::open(
        Arc::clone(&store),
        &root_path,
    )?;
    let nodes = group.traverse()?;

    load_zarr_meta_inner(
        &store,
        &nodes,
        root_path_str,
    )
}
