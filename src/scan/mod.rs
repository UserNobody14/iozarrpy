// Split from the historical `src/zarr_scan_async.rs` mega-file.

pub(crate) mod async_scan;
pub(crate) mod shared;
pub(crate) mod sync_scan;

pub(crate) use async_scan::chunk_to_df_from_grid_with_backend as chunk_to_df_from_grid_with_backend_async;
pub(crate) use sync_scan::chunk_to_df_from_grid_with_backend as chunk_to_df_from_grid_with_backend_sync;
