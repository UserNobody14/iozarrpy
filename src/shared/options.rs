//! Backend construction options.
//!
//! Bundles tunable knobs for backend caches so callers can grow this
//! struct over time without churning every constructor signature.

use std::collections::HashSet;

use crate::shared::IStr;

/// Options that control how a fully-cached zarr/icechunk backend is built.
///
/// Coordinates (latitude, longitude, time, lead_time, ...) are typically few,
/// small, and re-requested across many scans. Data variables are the opposite:
/// many, large, and short-lived. Sizing the two caches independently avoids
/// thrashing when a multi-variable scan would otherwise evict every coordinate
/// chunk. A value of `0` means **unbounded** (no entry-count eviction).
#[derive(Debug, Clone)]
pub struct BackendOptions {
    /// Maximum cached coordinate chunks. `0` = unbounded.
    pub coord_cache_max_entries: u64,
    /// Maximum cached data-variable chunks. `0` = unbounded.
    pub var_cache_max_entries: u64,
    /// Dimensions whose coordinate arrays are guaranteed sorted by the caller.
    /// The planner skips the monotonicity probe for these dims and only
    /// samples the endpoints to recover the sort direction.
    pub assume_sorted: AssumeSortedDims,
}

/// Caller assertion that some (or all) dim coordinates are monotonically
/// sorted. Skips the sample-based monotonicity check during coord resolution.
#[derive(Debug, Clone, Default)]
pub enum AssumeSortedDims {
    /// Verify monotonicity at planning time (default).
    #[default]
    None,
    /// Every dim's coordinate array is monotonic.
    All,
    /// Only these dims are known monotonic; verify the rest.
    Only(HashSet<IStr>),
}

impl AssumeSortedDims {
    pub fn contains(&self, dim: &IStr) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Only(s) => s.contains(dim),
        }
    }
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            coord_cache_max_entries: 256,
            // Sized so multivar workloads (10-50+ vars × multiple chunks
            // per query) don't thrash on hot scans. The previous default
            // of 30 caused chunks to be re-decoded on every repeat call
            // even for modest variable counts.
            var_cache_max_entries: 4096,
            assume_sorted:
                AssumeSortedDims::default(),
        }
    }
}
