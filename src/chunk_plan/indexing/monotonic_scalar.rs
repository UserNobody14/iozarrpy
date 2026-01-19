//! Synchronous monotonic coordinate resolver.
//!
//! Resolves value ranges to index ranges using binary search on coordinate arrays,
//! assuming the coordinate is monotonically increasing or decreasing.

use crate::chunk_plan::exprs::errors::{CoordIndexResolver, ResolveError};
use crate::chunk_plan::exprs::literals;
use crate::chunk_plan::prelude::*;
use super::types::{BoundKind, CoordScalar, IndexRange, ValueRange};
use super::resolver_traits::{HashMapCache, ResolutionCache, ResolutionRequest, SyncCoordResolver};

/// Synchronous coordinate resolver that uses binary search on monotonic coordinates.
pub(crate) struct MonotonicCoordResolver<'a> {
    meta: &'a ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
    coord_arrays:
        BTreeMap<String, Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>>,
    read_count: Arc<AtomicU64>,
    monotonic_cache: BTreeMap<String, Option<MonotonicDirection>>,
}

#[derive(Debug, Clone, Copy)]
enum MonotonicDirection {
    Increasing,
    Decreasing,
}

impl<'a> MonotonicCoordResolver<'a> {
    pub(crate) fn new(
        meta: &'a ZarrDatasetMeta,
        store: zarrs::storage::ReadableWritableListableStorage,
    ) -> Self {
        Self {
            meta,
            store,
            coord_arrays: BTreeMap::new(),
            read_count: Arc::new(AtomicU64::new(0)),
            monotonic_cache: BTreeMap::new(),
        }
    }

    pub(crate) fn coord_read_count(&self) -> u64 {
        self.read_count.load(Ordering::Relaxed)
    }

    fn coord_array(
        &mut self,
        dim: &str,
    ) -> Result<&Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>, ResolveError>
    {
        use std::collections::btree_map::Entry;

        let Some(m) = self.meta.arrays.get(dim) else {
            return Err(ResolveError::MissingCoord(dim.to_string()));
        };

        match self.coord_arrays.entry(dim.to_string()) {
            Entry::Occupied(o) => Ok(&*o.into_mut()),
            Entry::Vacant(v) => {
                let arr = Array::open(self.store.clone(), &m.path)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?;
                Ok(&*v.insert(arr))
            }
        }
    }

    fn scalar_at(&mut self, dim: &str, idx: u64) -> Result<CoordScalar, ResolveError> {
        let meta = self
            .meta
            .arrays
            .get(dim)
            .ok_or_else(|| ResolveError::MissingCoord(dim.to_string()))?;
        if meta.shape.len() != 1 {
            return Ok(CoordScalar::F64(f64::NAN)); // unsupported for now, will fail comparisons
        }
        if idx >= meta.shape[0] {
            return Err(ResolveError::OutOfBounds);
        }

        self.read_count.fetch_add(1, Ordering::Relaxed);
        let arr = self.coord_array(dim)?;
        let subset = ArraySubset::new_with_ranges(&[idx..(idx + 1)]);
        let id = arr.data_type().identifier();

        let te = meta.time_encoding.as_ref();
        match id {
            "float64" => {
                let v = arr
                    .retrieve_array_subset::<Vec<f64>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0];
                Ok(CoordScalar::F64(v))
            }
            "float32" => {
                let v = arr
                    .retrieve_array_subset::<Vec<f32>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as f64;
                Ok(CoordScalar::F64(v))
            }
            "int64" => {
                let raw = arr
                    .retrieve_array_subset::<Vec<i64>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0];
                Ok(literals::apply_time_encoding(raw, te))
            }
            "int32" => {
                let raw = arr
                    .retrieve_array_subset::<Vec<i32>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as i64;
                Ok(literals::apply_time_encoding(raw, te))
            }
            "int16" => {
                let raw = arr
                    .retrieve_array_subset::<Vec<i16>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as i64;
                Ok(literals::apply_time_encoding(raw, te))
            }
            "int8" => {
                let raw = arr
                    .retrieve_array_subset::<Vec<i8>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as i64;
                Ok(literals::apply_time_encoding(raw, te))
            }
            "uint64" => {
                let v = arr
                    .retrieve_array_subset::<Vec<u64>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0];
                Ok(CoordScalar::U64(v))
            }
            "uint32" => {
                let v = arr
                    .retrieve_array_subset::<Vec<u32>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as u64;
                Ok(CoordScalar::U64(v))
            }
            "uint16" => {
                let v = arr
                    .retrieve_array_subset::<Vec<u16>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as u64;
                Ok(CoordScalar::U64(v))
            }
            "uint8" => {
                let v = arr
                    .retrieve_array_subset::<Vec<u8>>(&subset)
                    .map_err(|e| ResolveError::Zarr(e.to_string()))?[0]
                    as u64;
                Ok(CoordScalar::U64(v))
            }
            other => Err(ResolveError::UnsupportedCoordDtype(other.to_string())),
        }
    }

    fn ensure_monotonic(&mut self, dim: &str) -> Result<Option<MonotonicDirection>, ResolveError> {
        if let Some(cached) = self.monotonic_cache.get(dim) {
            return Ok(*cached);
        }
        let Some(meta) = self.meta.arrays.get(dim) else {
            self.monotonic_cache.insert(dim.to_string(), None);
            return Ok(None);
        };
        if meta.shape.len() != 1 {
            self.monotonic_cache.insert(dim.to_string(), None);
            return Ok(None);
        }
        let n = meta.shape[0];
        if n < 2 {
            self.monotonic_cache
                .insert(dim.to_string(), Some(MonotonicDirection::Increasing));
            return Ok(Some(MonotonicDirection::Increasing));
        }

        // Cheap monotonic heuristic: sample a few points, including around a chunk boundary.
        // This is conservative: if we can't confirm, we return None (no binary-search pruning).
        let first = self.scalar_at(dim, 0)?;
        let last = self.scalar_at(dim, n - 1)?;
        let dir = match first.partial_cmp(&last) {
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal) => {
                MonotonicDirection::Increasing
            }
            Some(std::cmp::Ordering::Greater) => MonotonicDirection::Decreasing,
            None => {
                self.monotonic_cache.insert(dim.to_string(), None);
                return Ok(None);
            }
        };

        let arr = self.coord_array(dim)?;
        let reg_chunk = arr
            .chunk_shape(&[0])
            .map_err(|e| ResolveError::Zarr(e.to_string()))?
            .get(0)
            .map(|nz| nz.get())
            .unwrap_or(1);
        let mut sample_idxs = [
            0u64,
            (reg_chunk.saturating_sub(1)).min(n - 1),
            (reg_chunk).min(n - 1),
            (n / 2).min(n - 1),
            n - 1,
        ];
        sample_idxs.sort_unstable();

        let mut prev: Option<CoordScalar> = None;
        for &i in &sample_idxs {
            let v = self.scalar_at(dim, i)?;
            if let Some(p) = &prev {
                let ord = p.partial_cmp(&v);
                let ok = match (dir, ord) {
                    (
                        MonotonicDirection::Increasing,
                        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal),
                    ) => true,
                    (
                        MonotonicDirection::Decreasing,
                        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
                    ) => true,
                    _ => false,
                };
                if !ok {
                    self.monotonic_cache.insert(dim.to_string(), None);
                    return Ok(None);
                }
            }
            prev = Some(v);
        }

        self.monotonic_cache.insert(dim.to_string(), Some(dir));
        Ok(Some(dir))
    }

    // ========================================================================
    // Binary search bounds
    // ========================================================================

    fn lower_bound(
        &mut self,
        dim: &str,
        target: &CoordScalar,
        strict: bool,
        dir: MonotonicDirection,
        n: u64,
    ) -> Result<u64, ResolveError> {
        // For increasing: first idx with value > target (strict) or >= target (!strict).
        // For decreasing: first idx with value < target (strict) or <= target (!strict).
        let mut lo = 0u64;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let v = self.scalar_at(dim, mid)?;
            let cmp = v.partial_cmp(target);
            let go_left = match (dir, strict, cmp) {
                (
                    MonotonicDirection::Increasing,
                    false,
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
                ) => true,
                (MonotonicDirection::Increasing, true, Some(std::cmp::Ordering::Greater)) => true,
                (
                    MonotonicDirection::Decreasing,
                    false,
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal),
                ) => true,
                (MonotonicDirection::Decreasing, true, Some(std::cmp::Ordering::Less)) => true,
                _ => false,
            };
            if go_left {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }

    fn upper_bound(
        &mut self,
        dim: &str,
        target: &CoordScalar,
        strict: bool,
        dir: MonotonicDirection,
        n: u64,
    ) -> Result<u64, ResolveError> {
        // Return end_exclusive for max bound:
        // We want end_exclusive such that values satisfy value < max (Exclusive) or <= max (Inclusive).
        // So compute first idx that violates that.
        let mut lo = 0u64;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let v = self.scalar_at(dim, mid)?;
            let cmp = v.partial_cmp(target);
            let go_left = match (dir, strict, cmp) {
                (
                    MonotonicDirection::Increasing,
                    true,
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
                ) => true, // >= max
                (MonotonicDirection::Increasing, false, Some(std::cmp::Ordering::Greater)) => true, // > max
                (
                    MonotonicDirection::Decreasing,
                    true,
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal),
                ) => true,
                (MonotonicDirection::Decreasing, false, Some(std::cmp::Ordering::Less)) => true,
                _ => false,
            };
            if go_left {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }
}

// ============================================================================
// CoordIndexResolver trait implementation
// ============================================================================

impl CoordIndexResolver for MonotonicCoordResolver<'_> {
    fn index_range_for_value_range(
        &mut self,
        dim: &str,
        range: &ValueRange,
    ) -> Result<Option<IndexRange>, ResolveError> {
        let Some(meta) = self.meta.arrays.get(dim) else {
            return Ok(None);
        };
        if meta.shape.len() != 1 {
            return Ok(None);
        }
        let n = meta.shape[0];
        if n == 0 {
            return Ok(Some(IndexRange {
                start: 0,
                end_exclusive: 0,
            }));
        }

        let Some(dir) = self.ensure_monotonic(dim)? else {
            return Ok(None);
        };

        // Equality is treated as a tiny closed range in index space using two bounds.
        if let Some(eq) = &range.eq {
            let start = self.lower_bound(dim, eq, false, dir, n)?;
            let end_excl = self.upper_bound(dim, eq, false, dir, n)?;
            let out = IndexRange {
                start,
                end_exclusive: end_excl,
            };
            return Ok(Some(out));
        }

        let start = if let Some((v, bk)) = &range.min {
            let strict = *bk == BoundKind::Exclusive;
            self.lower_bound(dim, v, strict, dir, n)?
        } else {
            0
        };
        let end_exclusive = if let Some((v, bk)) = &range.max {
            let strict = *bk == BoundKind::Exclusive;
            self.upper_bound(dim, v, strict, dir, n)?
        } else {
            n
        };

        Ok(Some(IndexRange {
            start,
            end_exclusive,
        }))
    }
}

// ============================================================================
// SyncCoordResolver trait implementation for batched resolution
// ============================================================================

impl SyncCoordResolver for MonotonicCoordResolver<'_> {
    fn resolve_batch<'a>(
        &'a mut self,
        requests: &[ResolutionRequest],
        _meta: &ZarrDatasetMeta,
    ) -> Box<dyn ResolutionCache + 'a> {
        let mut cache = HashMapCache::new();

        for request in requests {
            let dim = request.dim.as_ref();
            let vr = &request.value_range;

            // Use the CoordIndexResolver implementation
            let result = self.index_range_for_value_range(dim, vr);

            match result {
                Ok(idx_range) => {
                    cache.insert(request.clone(), idx_range);
                }
                Err(_) => {
                    // Resolution failed - store None
                    cache.insert(request.clone(), None);
                }
            }
        }

        Box::new(cache)
    }
}
