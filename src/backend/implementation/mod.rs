/// Implementation details for the backend.
mod icechunk;
mod icechunk_iterating;
mod iterating;
mod lazy;
mod zarr_async;

pub(crate) use icechunk::{
    FullyCachedIcechunkBackendAsync,
    IcechunkBackendAsync,
    to_fully_cached_icechunk_async,
};
pub(crate) use icechunk_iterating::IcechunkIterator;
pub(crate) use iterating::ZarrIterator;
pub(crate) use lazy::scan_zarr_with_backend_sync;
pub(crate) use zarr_async::scan_zarr_with_backend_async;
