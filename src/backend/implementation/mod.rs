/// Implementation details for the backend.
mod icechunk;
mod lazy;

pub(crate) use lazy::scan_zarr_with_backend_sync;

pub(crate) use icechunk::{
    FullyCachedIcechunkBackendAsync,
    IcechunkBackendAsync,
    to_fully_cached_icechunk_async,
};
