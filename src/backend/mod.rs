//! Backend abstraction for Zarr data access with caching.
//!
//! This module provides:
//! - [`ZarrBackendSync`] and [`ZarrBackendAsync`] traits for backend abstraction
//! - [`CachingAsyncBackend`] - default caching implementation
//! - [`PyZarrBackend`] - Python-exposed backend class with scan methods
//! - [`PyIcechunkBackend`] - Python-exposed Icechunk backend (async-only)

mod compile;
mod icechunk;
mod icechunk_py;
mod lazy;
mod py;
mod sync;
pub(crate) mod traits;
mod zarr;

pub(crate) use icechunk_py::{
    PyIcechunkBackend, PySession,
};
pub(crate) use py::PyZarrBackend;
pub(crate) use sync::PyZarrBackendSync;

// Re-export commonly used traits
pub(crate) use traits::{
    BackendError, ChunkDataSourceAsync,
    ChunkDataSourceSync, ChunkedDataBackendAsync,
    ChunkedDataBackendSync,
    HasMetadataBackendAsync,
    HasMetadataBackendSync,
};

// Re-export compile traits
pub(crate) use compile::{
    ChunkedExpressionCompilerAsync,
    ChunkedExpressionCompilerSync,
    ChunkedExpressionCompilerWithBackendAsync,
};
