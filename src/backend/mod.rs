//! Backend abstraction for Zarr data access with caching.
//!
//! This module provides:
//! - [`ZarrBackendSync`] and [`ZarrBackendAsync`] traits for backend abstraction
//! - [`CachingAsyncBackend`] - default caching implementation
//! - [`PyZarrBackend`] - Python-exposed backend class with scan methods
//! - [`PyIcechunkBackend`] - Python-exposed Icechunk backend (async-only)

mod implementation;
mod py;

pub(crate) use py::PyIcechunkBackend;
pub(crate) use py::PyZarrBackend;
pub(crate) use py::PyZarrBackendSync;
