//! Traits for coordinate-to-index resolution.
//!
//! These traits enable both synchronous and asynchronous batch resolution of
//! value ranges to index ranges, allowing efficient parallel fetching of
//! coordinate data.

use std::collections::HashMap;

use super::types::ValueRange;
use crate::{IStr, IntoIStr};

/// A request to resolve a value range to an index range for a specific dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ResolutionRequest {
    /// The dimension name (interned for efficiency).
    pub(crate) dim: IStr,
    /// The value range to resolve.
    pub(crate) value_range: ValueRange,
}

impl ResolutionRequest {
    /// Create a new resolution request.
    pub(crate) fn new(
        dim: &str,
        value_range: ValueRange,
    ) -> Self {
        Self {
            dim: dim.istr(),
            value_range,
        }
    }
}

/// Cache of resolved requests.
///
/// This is returned by resolvers after batch resolution and is used during
/// materialization to look up resolved index ranges.
pub(crate) trait ResolutionCache:
    std::fmt::Debug
{
    /// Get the resolved index range for a request.
    ///
    /// Returns:
    /// - `Some(Some(range))` if the request was resolved to a valid range
    /// - `Some(None)` if resolution was attempted but no range could be determined
    ///   (e.g., non-monotonic coordinate array)
    /// - `None` if the request was not found in the cache
    fn get(
        &self,
        request: &ResolutionRequest,
    ) -> Option<Option<std::ops::Range<u64>>>;
}

/// A simple HashMap-based resolution cache.
#[derive(Debug, Default)]
pub(crate) struct HashMapCache {
    cache: HashMap<
        ResolutionRequest,
        Option<std::ops::Range<u64>>,
    >,
}

impl HashMapCache {
    /// Create a new empty cache.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Insert a resolved result into the cache.
    pub(crate) fn insert(
        &mut self,
        request: ResolutionRequest,
        result: Option<std::ops::Range<u64>>,
    ) {
        self.cache.insert(request, result);
    }

    /// Get the number of cached entries.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.cache.len()
    }
}

impl ResolutionCache for HashMapCache {
    fn get(
        &self,
        request: &ResolutionRequest,
    ) -> Option<Option<std::ops::Range<u64>>>
    {
        self.cache.get(request).cloned()
    }
}

/// Synchronous batch coordinator resolver.
///
/// Implementors resolve a batch of value ranges to index ranges in a single
/// synchronous call, enabling efficient I/O batching.
pub(crate) trait SyncCoordResolver {
    /// Resolve a batch of requests.
    ///
    /// Returns a cache containing the results. Requests that couldn't be resolved
    /// (e.g., non-monotonic coordinates) should have `None` as their result.
    fn resolve_batch(
        &self,
        requests: Vec<ResolutionRequest>,
    ) -> Box<dyn ResolutionCache + Send + Sync>;
}

/// Asynchronous batch coordinator resolver.
///
/// Implementors resolve a batch of value ranges to index ranges asynchronously,
/// enabling concurrent I/O for multiple dimensions.
#[async_trait::async_trait]
pub(crate) trait AsyncCoordResolver:
    Send + Sync
{
    /// Resolve a batch of requests asynchronously.
    ///
    /// Returns a cache containing the results. Requests that couldn't be resolved
    /// (e.g., non-monotonic coordinates) should have `None` as their result.
    async fn resolve_batch(
        &self,
        requests: Vec<ResolutionRequest>,
    ) -> Box<dyn ResolutionCache + Send + Sync>;
}

/// Error type for resolution operations.
#[derive(Debug, Clone)]
pub(crate) enum ResolutionError {
    /// The request was not found in the cache (should not happen if collection is correct).
    NotFound(ResolutionRequest),
    /// Resolution was attempted but failed (e.g., non-monotonic array).
    Unresolvable(String),
}

impl std::fmt::Display for ResolutionError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            ResolutionError::NotFound(req) => {
                write!(
                    f,
                    "resolution request not found in cache: {:?}",
                    req
                )
            }
            ResolutionError::Unresolvable(
                msg,
            ) => {
                write!(f, "unresolvable: {}", msg)
            }
        }
    }
}

impl std::error::Error for ResolutionError {}
