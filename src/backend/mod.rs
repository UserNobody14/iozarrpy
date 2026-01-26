//! Backend abstraction for Zarr data access with caching.
//!
//! This module provides:
//! - [`ZarrBackendSync`] and [`ZarrBackendAsync`] traits for backend abstraction
//! - [`CachingAsyncBackend`] - default caching implementation
//! - [`PyZarrBackend`] - Python-exposed backend class with scan methods

mod traits;
mod caching;
mod source;
mod py;

pub use traits::{
    BackendError, CoordChunkData, CoordScalarRaw, DynAsyncBackend, DynSyncBackend,
    ZarrBackendAsync, ZarrBackendSync,
};

pub use caching::CachingAsyncBackend;

pub(crate) use py::PyZarrBackend;
