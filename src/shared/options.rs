//! Backend construction options.
//!
//! Bundles tunable knobs for backend caches so callers can grow this
//! struct over time without churning every constructor signature.

/// Options that control how a fully-cached zarr/icechunk backend is built.
///
/// Coordinates (latitude, longitude, time, lead_time, ...) are typically few,
/// small, and re-requested across many scans. Data variables are the opposite:
/// many, large, and short-lived. Sizing the two caches independently avoids
/// thrashing when a multi-variable scan would otherwise evict every coordinate
/// chunk. A value of `0` means **unbounded** (no entry-count eviction).
#[derive(Debug, Clone, Copy)]
pub struct BackendOptions {
    /// Maximum cached coordinate chunks. `0` = unbounded.
    pub coord_cache_max_entries: u64,
    /// Maximum cached data-variable chunks. `0` = unbounded.
    pub var_cache_max_entries: u64,
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
        }
    }
}
