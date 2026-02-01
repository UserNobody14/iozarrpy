//! Backend abstraction for Zarr data access with caching.
//!
//! This module provides:
//! - [`ZarrBackendSync`] and [`ZarrBackendAsync`] traits for backend abstraction
//! - [`CachingAsyncBackend`] - default caching implementation
//! - [`PyZarrBackend`] - Python-exposed backend class with scan methods

mod compile;
mod lazy;
mod sync;
mod traits;
mod zarr;
// mod source;  // TODO: This module was referenced but doesn't exist
mod py;

pub(crate) use py::PyZarrBackend;
