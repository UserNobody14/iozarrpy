//! Synchronous monotonic coordinate resolver.
//!
//! Resolves value ranges to index ranges using binary search on coordinate arrays,
//! assuming the coordinate is monotonically increasing or decreasing.
//!
//! Uses chunk-level caching for efficient I/O: entire chunks are fetched and cached
//! rather than individual scalar values.

use super::resolver_traits::{
    HashMapCache, ResolutionCache,
    ResolutionRequest, SyncCoordResolver,
};
use super::types::{
    BoundKind, CoordScalar, IndexRange,
    ValueRange,
};
use crate::chunk_plan::exprs::errors::{
    CoordIndexResolver, ResolveError,
};
use crate::chunk_plan::exprs::literals;
use crate::chunk_plan::prelude::*;
use crate::{IStr, IntoIStr};

/// Cached chunk data for a coordinate array.
///
/// Stores decoded scalar values for efficient lookup.
#[derive(Debug)]
enum ChunkData {
    F64(Vec<f64>),
    I64(Vec<i64>),
    U64(Vec<u64>),
}

impl ChunkData {
    fn get(
        &self,
        offset: usize,
        te: Option<&crate::meta::TimeEncoding>,
    ) -> Option<CoordScalar> {
        match self {
            ChunkData::F64(v) => v
                .get(offset)
                .copied()
                .map(CoordScalar::F64),
            ChunkData::I64(v) => v
                .get(offset)
                .copied()
                .map(|raw| {
                    literals::apply_time_encoding(
                        raw, te,
                    )
                }),
            ChunkData::U64(v) => v
                .get(offset)
                .copied()
                .map(CoordScalar::U64),
        }
    }
}

/// Synchronous coordinate resolver that uses binary search on monotonic coordinates.
///
/// Features chunk-level caching: when a value is requested, the entire containing
/// chunk is fetched and cached, making subsequent reads from the same chunk instant.
pub(crate) struct MonotonicCoordResolver<'a> {
    meta: &'a ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
    coord_arrays:
        BTreeMap<IStr, Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>>,
    /// Chunk cache: (dim, chunk_idx) -> decoded chunk data
    chunk_cache: BTreeMap<(IStr, u64), ChunkData>,
    /// Number of chunk reads (I/O operations)
    chunk_read_count: Arc<AtomicU64>,
    monotonic_cache: BTreeMap<IStr, Option<MonotonicDirection>>,
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
            chunk_cache: BTreeMap::new(),
            chunk_read_count: Arc::new(
                AtomicU64::new(0),
            ),
            monotonic_cache: BTreeMap::new(),
        }
    }

    pub(crate) fn coord_read_count(&self) -> u64 {
        self.chunk_read_count
            .load(Ordering::Relaxed)
    }

    fn coord_array(
        &mut self,
        dim: &str,
    ) -> Result<&Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>, ResolveError>
    {
        use std::collections::btree_map::Entry;

        let dim_key = dim.istr();
        let Some(m) =
            self.meta.arrays.get(&dim_key)
        else {
            return Err(
                ResolveError::MissingCoord(
                    dim.to_string(),
                ),
            );
        };

        match self.coord_arrays.entry(dim_key) {
            Entry::Occupied(o) => {
                Ok(&*o.into_mut())
            }
            Entry::Vacant(v) => {
                let arr = Array::open(
                    self.store.clone(),
                    m.path.as_ref(),
                )
                .map_err(|e| {
                    ResolveError::Zarr(
                        e.to_string(),
                    )
                })?;
                Ok(&*v.insert(arr))
            }
        }
    }

    /// Get the chunk size for a 1D coordinate array.
    fn chunk_size(
        &mut self,
        dim: &str,
    ) -> Result<u64, ResolveError> {
        let arr = self.coord_array(dim)?;
        let chunk_shape = arr
            .chunk_shape(&[0])
            .map_err(|e| {
            ResolveError::Zarr(e.to_string())
        })?;
        Ok(chunk_shape
            .first()
            .map(|nz| nz.get())
            .unwrap_or(1))
    }

    /// Load a chunk and cache it.
    fn load_chunk(
        &mut self,
        dim: &str,
        chunk_idx: u64,
    ) -> Result<(), ResolveError> {
        let dim_key = dim.istr();
        let cache_key =
            (dim_key.clone(), chunk_idx);
        if self
            .chunk_cache
            .contains_key(&cache_key)
        {
            return Ok(());
        }

        let _meta = self
            .meta
            .arrays
            .get(&dim_key)
            .ok_or_else(|| {
                ResolveError::MissingCoord(
                    dim.to_string(),
                )
            })?;

        // Increment counter before borrowing array
        self.chunk_read_count
            .fetch_add(1, Ordering::Relaxed);

        let arr = self.coord_array(dim)?;
        let id = arr.data_type().identifier();
        let chunk_indices = vec![chunk_idx];

        let data = match id {
            "float64" => {
                let values = arr
                    .retrieve_chunk::<Vec<f64>>(
                        &chunk_indices,
                    )
                    .map_err(|e| {
                        ResolveError::Zarr(
                            e.to_string(),
                        )
                    })?;
                ChunkData::F64(values)
            }
            "float32" => {
                let values: Vec<f64> = arr
                    .retrieve_chunk::<Vec<f32>>(
                        &chunk_indices,
                    )
                    .map_err(|e| {
                        ResolveError::Zarr(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                ChunkData::F64(values)
            }
            "int64" => {
                let values = arr
                    .retrieve_chunk::<Vec<i64>>(
                        &chunk_indices,
                    )
                    .map_err(|e| {
                        ResolveError::Zarr(
                            e.to_string(),
                        )
                    })?;
                ChunkData::I64(values)
            }
            "int32" => {
                let values: Vec<i64> = arr
                    .retrieve_chunk::<Vec<i32>>(
                        &chunk_indices,
                    )
                    .map_err(|e| {
                        ResolveError::Zarr(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                ChunkData::I64(values)
            }
            "int16" => {
                let values: Vec<i64> = arr
                    .retrieve_chunk::<Vec<i16>>(
                        &chunk_indices,
                    )
                    .map_err(|e| {
                        ResolveError::Zarr(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                ChunkData::I64(values)
            }
            "int8" => {
                let values: Vec<i64> = arr
                    .retrieve_chunk::<Vec<i8>>(
                        &chunk_indices,
                    )
                    .map_err(|e| {
                        ResolveError::Zarr(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                ChunkData::I64(values)
            }
            "uint64" => {
                let values = arr
                    .retrieve_chunk::<Vec<u64>>(
                        &chunk_indices,
                    )
                    .map_err(|e| {
                        ResolveError::Zarr(
                            e.to_string(),
                        )
                    })?;
                ChunkData::U64(values)
            }
            "uint32" => {
                let values: Vec<u64> = arr
                    .retrieve_chunk::<Vec<u32>>(
                        &chunk_indices,
                    )
                    .map_err(|e| {
                        ResolveError::Zarr(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as u64)
                    .collect();
                ChunkData::U64(values)
            }
            "uint16" => {
                let values: Vec<u64> = arr
                    .retrieve_chunk::<Vec<u16>>(
                        &chunk_indices,
                    )
                    .map_err(|e| {
                        ResolveError::Zarr(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as u64)
                    .collect();
                ChunkData::U64(values)
            }
            "uint8" => {
                let values: Vec<u64> = arr
                    .retrieve_chunk::<Vec<u8>>(
                        &chunk_indices,
                    )
                    .map_err(|e| {
                        ResolveError::Zarr(
                            e.to_string(),
                        )
                    })?
                    .into_iter()
                    .map(|v| v as u64)
                    .collect();
                ChunkData::U64(values)
            }
            other => {
                return Err(
                    ResolveError::UnsupportedCoordDtype(
                        other.to_string(),
                    ),
                );
            }
        };

        self.chunk_cache.insert(cache_key, data);
        Ok(())
    }

    fn scalar_at(
        &mut self,
        dim: &str,
        idx: u64,
    ) -> Result<CoordScalar, ResolveError> {
        let dim_key = dim.istr();
        let meta = self
            .meta
            .arrays
            .get(&dim_key)
            .ok_or_else(|| {
                ResolveError::MissingCoord(
                    dim.to_string(),
                )
            })?;
        if meta.shape.len() != 1 {
            return Ok(CoordScalar::F64(
                f64::NAN,
            )); // unsupported for now, will fail comparisons
        }
        if idx >= meta.shape[0] {
            return Err(
                ResolveError::OutOfBounds,
            );
        }

        let chunk_size = self.chunk_size(dim)?;
        let chunk_idx = idx / chunk_size;
        let offset_in_chunk =
            (idx % chunk_size) as usize;

        // Ensure chunk is loaded
        self.load_chunk(dim, chunk_idx)?;

        // Get value from cache
        let cache_key = (dim_key, chunk_idx);
        let te = meta.time_encoding.as_ref();

        self.chunk_cache
            .get(&cache_key)
            .and_then(|data| {
                data.get(offset_in_chunk, te)
            })
            .ok_or(ResolveError::OutOfBounds)
    }

    fn ensure_monotonic(
        &mut self,
        dim: &str,
    ) -> Result<
        Option<MonotonicDirection>,
        ResolveError,
    > {
        let dim_key = dim.istr();
        if let Some(cached) =
            self.monotonic_cache.get(&dim_key)
        {
            return Ok(*cached);
        }
        let Some(meta) =
            self.meta.arrays.get(&dim_key)
        else {
            self.monotonic_cache
                .insert(dim_key, None);
            return Ok(None);
        };
        if meta.shape.len() != 1 {
            self.monotonic_cache
                .insert(dim_key, None);
            return Ok(None);
        }
        let n = meta.shape[0];
        if n < 2 {
            self.monotonic_cache.insert(
                dim_key,
                Some(MonotonicDirection::Increasing),
            );
            return Ok(Some(
                MonotonicDirection::Increasing,
            ));
        }

        // Check monotonicity by sampling first, last, and a few points around chunk boundaries.
        let first = self.scalar_at(dim, 0)?;
        let last = self.scalar_at(dim, n - 1)?;
        let dir = match first.partial_cmp(&last) {
            Some(
                std::cmp::Ordering::Less
                | std::cmp::Ordering::Equal,
            ) => MonotonicDirection::Increasing,
            Some(std::cmp::Ordering::Greater) => {
                MonotonicDirection::Decreasing
            }
            None => {
                self.monotonic_cache
                    .insert(dim_key, None);
                return Ok(None);
            }
        };

        let chunk_size = self.chunk_size(dim)?;
        let mut sample_idxs = [
            0u64,
            (chunk_size.saturating_sub(1))
                .min(n - 1),
            chunk_size.min(n - 1),
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
                        Some(
                            std::cmp::Ordering::Less
                            | std::cmp::Ordering::Equal,
                        ),
                    ) => true,
                    (
                        MonotonicDirection::Decreasing,
                        Some(
                            std::cmp::Ordering::Greater
                            | std::cmp::Ordering::Equal,
                        ),
                    ) => true,
                    _ => false,
                };
                if !ok {
                    self.monotonic_cache.insert(
                        dim_key.clone(),
                        None,
                    );
                    return Ok(None);
                }
            }
            prev = Some(v);
        }

        self.monotonic_cache
            .insert(dim_key, Some(dir));
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
                    Some(
                        std::cmp::Ordering::Greater
                        | std::cmp::Ordering::Equal,
                    ),
                ) => true,
                (
                    MonotonicDirection::Increasing,
                    true,
                    Some(std::cmp::Ordering::Greater),
                ) => true,
                (
                    MonotonicDirection::Decreasing,
                    false,
                    Some(
                        std::cmp::Ordering::Less
                        | std::cmp::Ordering::Equal,
                    ),
                ) => true,
                (
                    MonotonicDirection::Decreasing,
                    true,
                    Some(std::cmp::Ordering::Less),
                ) => true,
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
                    Some(
                        std::cmp::Ordering::Greater
                        | std::cmp::Ordering::Equal,
                    ),
                ) => true,
                (
                    MonotonicDirection::Increasing,
                    false,
                    Some(std::cmp::Ordering::Greater),
                ) => true,
                (
                    MonotonicDirection::Decreasing,
                    true,
                    Some(
                        std::cmp::Ordering::Less
                        | std::cmp::Ordering::Equal,
                    ),
                ) => true,
                (
                    MonotonicDirection::Decreasing,
                    false,
                    Some(std::cmp::Ordering::Less),
                ) => true,
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

impl CoordIndexResolver
    for MonotonicCoordResolver<'_>
{
    fn index_range_for_value_range(
        &mut self,
        dim: &str,
        range: &ValueRange,
    ) -> Result<Option<IndexRange>, ResolveError>
    {
        let dim_key = dim.istr();
        let Some(meta) =
            self.meta.arrays.get(&dim_key)
        else {
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

        let Some(dir) =
            self.ensure_monotonic(dim)?
        else {
            return Ok(None);
        };

        // Equality is treated as a tiny closed range in index space using two bounds.
        if let Some(eq) = &range.eq {
            let start = self.lower_bound(
                dim, eq, false, dir, n,
            )?;
            let end_excl = self.upper_bound(
                dim, eq, false, dir, n,
            )?;
            let out = IndexRange {
                start,
                end_exclusive: end_excl,
            };
            return Ok(Some(out));
        }

        let start =
            if let Some((v, bk)) = &range.min {
                let strict =
                    *bk == BoundKind::Exclusive;
                self.lower_bound(
                    dim, v, strict, dir, n,
                )?
            } else {
                0
            };
        let end_exclusive =
            if let Some((v, bk)) = &range.max {
                let strict =
                    *bk == BoundKind::Exclusive;
                self.upper_bound(
                    dim, v, strict, dir, n,
                )?
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

impl SyncCoordResolver
    for MonotonicCoordResolver<'_>
{
    fn resolve_batch<'a>(
        &'a mut self,
        requests: &[ResolutionRequest],
        _meta: &ZarrDatasetMeta,
    ) -> Box<dyn ResolutionCache + 'a> {
        let mut cache = HashMapCache::new();

        for request in requests {
            let dim = request.dim.as_ref();
            let vr = &request.value_range;

            let result = self
                .index_range_for_value_range(
                    dim, vr,
                );

            match result {
                Ok(idx_range) => {
                    cache.insert(
                        request.clone(),
                        idx_range,
                    );
                }
                Err(_) => {
                    cache.insert(
                        request.clone(),
                        None,
                    );
                }
            }
        }

        Box::new(cache)
    }
}
