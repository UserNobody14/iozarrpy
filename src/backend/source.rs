//! BackendSource for sync lazy scans via Polars' register_io_source.
//!
//! This module provides an iterator-based source that wraps a backend
//! for use with Polars' lazy IO plugin system.
//!
//! Note: The sync lazy scan is implemented in Python using register_io_source,
//! with the backend passed as state. This module provides the necessary
//! infrastructure for that.

use std::sync::Arc;

use super::CachingAsyncBackend;

/// Wrapper that holds a backend for use in Python's sync lazy scan.
///
/// This is used by the Python `scan_zarr()` method which uses `register_io_source`.
pub struct BackendHolder {
    pub backend: Arc<CachingAsyncBackend>,
}

impl BackendHolder {
    pub fn new(backend: Arc<CachingAsyncBackend>) -> Self {
        Self { backend }
    }
}
