//! Async monotonic coordinate resolver.
//!
//! Provides async/concurrent resolution of value ranges to index ranges,
//! enabling parallel fetching of coordinate data across dimensions.
//!
//! Uses chunk-level caching for efficient I/O: entire chunks are fetched and cached
//! rather than individual scalar values.

use std::collections::BTreeMap;
use tokio::sync::RwLock;

use super::resolver_traits::{
    AsyncCoordResolver, HashMapCache,
    ResolutionCache, ResolutionRequest,
};
use super::types::{
    BoundKind, CoordScalar, IndexRange,
    ValueRange,
};
use crate::IStr;
use crate::chunk_plan::exprs;
use crate::meta::ZarrDatasetMeta;

use zarrs::array::Array;

/// Cached chunk data for a coordinate array.
#[derive(Debug, Clone)]
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
                    exprs::apply_time_encoding(
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

/// Async monotonic coordinate resolver with chunk-level caching.
///
/// Resolves value ranges to index ranges using binary search on coordinate arrays.
/// Entire chunks are fetched and cached, making subsequent reads from the same chunk instant.
pub(crate) struct AsyncMonotonicResolver {
    store: zarrs::storage::AsyncReadableWritableListableStorage,
}

impl AsyncMonotonicResolver {
    /// Create a new async resolver.
    pub(crate) fn new(
        store: zarrs::storage::AsyncReadableWritableListableStorage,
    ) -> Self {
        Self { store }
    }
}

#[async_trait::async_trait]
impl AsyncCoordResolver
    for AsyncMonotonicResolver
{
    async fn resolve_batch(
        &self,
        requests: Vec<ResolutionRequest>,
        meta: &ZarrDatasetMeta,
    ) -> Box<dyn ResolutionCache + Send + Sync>
    {
        let mut cache = HashMapCache::new();

        // Group requests by dimension to minimize array opens
        let mut by_dim: BTreeMap<
            IStr,
            Vec<(ResolutionRequest, ValueRange)>,
        > = BTreeMap::new();
        for req in requests {
            by_dim
                .entry(req.dim.clone())
                .or_default()
                .push((
                    req.clone(),
                    req.value_range.clone(),
                ));
        }

        // Resolve each dimension concurrently
        let futures: Vec<_> = by_dim
            .into_iter()
            .map(|(dim, reqs)| {
                let store = self.store.clone();
                let meta = meta.clone();
                async move {
                    resolve_dimension_async(
                        &dim, reqs, &meta, store,
                    )
                    .await
                }
            })
            .collect();

        let results =
            futures::future::join_all(futures)
                .await;

        // Merge all results into the cache
        for dim_cache in results {
            for (req, result) in dim_cache {
                cache.insert(req, result);
            }
        }

        Box::new(cache)
    }
}

/// Per-dimension resolver state with chunk cache.
struct DimResolver {
    arr: Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    array_meta: crate::meta::ZarrArrayMeta,
    chunk_size: u64,
    n: u64,
    chunk_cache: RwLock<BTreeMap<u64, ChunkData>>,
}

impl DimResolver {
    async fn new(
        store: zarrs::storage::AsyncReadableWritableListableStorage,
        array_meta: &crate::meta::ZarrArrayMeta,
    ) -> Option<Self> {
        if array_meta.shape.len() != 1 {
            return None;
        }
        let n = array_meta.shape[0];

        let arr = Array::async_open(
            store,
            &array_meta.path,
        )
        .await
        .ok()?;

        let chunk_size = arr
            .chunk_shape(&[0])
            .ok()?
            .first()
            .map(|nz| nz.get())
            .unwrap_or(1);

        Some(Self {
            arr,
            array_meta: array_meta.clone(),
            chunk_size,
            n,
            chunk_cache: RwLock::new(
                BTreeMap::new(),
            ),
        })
    }

    async fn load_chunk(
        &self,
        chunk_idx: u64,
    ) -> Option<()> {
        // Check if already cached
        {
            let cache =
                self.chunk_cache.read().await;
            if cache.contains_key(&chunk_idx) {
                return Some(());
            }
        }

        // Fetch chunk
        let chunk_indices = vec![chunk_idx];
        let id =
            self.arr.data_type().identifier();

        let data = match id {
            "float64" => {
                let values = self
                    .arr
                    .async_retrieve_chunk::<Vec<f64>>(
                        &chunk_indices,
                    )
                    .await
                    .ok()?;
                ChunkData::F64(values)
            }
            "float32" => {
                let values: Vec<f64> = self
                    .arr
                    .async_retrieve_chunk::<Vec<f32>>(
                        &chunk_indices,
                    )
                    .await
                    .ok()?
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                ChunkData::F64(values)
            }
            "int64" => {
                let values = self
                    .arr
                    .async_retrieve_chunk::<Vec<i64>>(
                        &chunk_indices,
                    )
                    .await
                    .ok()?;
                ChunkData::I64(values)
            }
            "int32" => {
                let values: Vec<i64> = self
                    .arr
                    .async_retrieve_chunk::<Vec<i32>>(
                        &chunk_indices,
                    )
                    .await
                    .ok()?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                ChunkData::I64(values)
            }
            "int16" => {
                let values: Vec<i64> = self
                    .arr
                    .async_retrieve_chunk::<Vec<i16>>(
                        &chunk_indices,
                    )
                    .await
                    .ok()?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                ChunkData::I64(values)
            }
            "int8" => {
                let values: Vec<i64> = self
                    .arr
                    .async_retrieve_chunk::<Vec<i8>>(
                        &chunk_indices,
                    )
                    .await
                    .ok()?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                ChunkData::I64(values)
            }
            "uint64" => {
                let values = self
                    .arr
                    .async_retrieve_chunk::<Vec<u64>>(
                        &chunk_indices,
                    )
                    .await
                    .ok()?;
                ChunkData::U64(values)
            }
            "uint32" => {
                let values: Vec<u64> = self
                    .arr
                    .async_retrieve_chunk::<Vec<u32>>(
                        &chunk_indices,
                    )
                    .await
                    .ok()?
                    .into_iter()
                    .map(|v| v as u64)
                    .collect();
                ChunkData::U64(values)
            }
            "uint16" => {
                let values: Vec<u64> = self
                    .arr
                    .async_retrieve_chunk::<Vec<u16>>(
                        &chunk_indices,
                    )
                    .await
                    .ok()?
                    .into_iter()
                    .map(|v| v as u64)
                    .collect();
                ChunkData::U64(values)
            }
            "uint8" => {
                let values: Vec<u64> = self
                    .arr
                    .async_retrieve_chunk::<Vec<u8>>(
                        &chunk_indices,
                    )
                    .await
                    .ok()?
                    .into_iter()
                    .map(|v| v as u64)
                    .collect();
                ChunkData::U64(values)
            }
            _ => return None,
        };

        // Cache it
        {
            let mut cache =
                self.chunk_cache.write().await;
            cache.insert(chunk_idx, data);
        }
        Some(())
    }

    async fn scalar_at(
        &self,
        idx: u64,
    ) -> Option<CoordScalar> {
        if idx >= self.n {
            return None;
        }

        let chunk_idx = idx / self.chunk_size;
        let offset_in_chunk =
            (idx % self.chunk_size) as usize;

        // Load chunk if needed
        self.load_chunk(chunk_idx).await?;

        // Read from cache
        let cache = self.chunk_cache.read().await;
        let te = self
            .array_meta
            .time_encoding
            .as_ref();
        cache
            .get(&chunk_idx)?
            .get(offset_in_chunk, te)
    }

    async fn check_monotonic(
        &self,
    ) -> Option<MonotonicDir> {
        if self.n < 2 {
            return Some(
                MonotonicDir::Increasing,
            );
        }

        // Sample first and last values
        let first = self.scalar_at(0).await?;
        let last =
            self.scalar_at(self.n - 1).await?;

        let dir = match first.partial_cmp(&last) {
            Some(
                std::cmp::Ordering::Less
                | std::cmp::Ordering::Equal,
            ) => MonotonicDir::Increasing,
            Some(std::cmp::Ordering::Greater) => {
                MonotonicDir::Decreasing
            }
            None => return None,
        };

        // Sample a few more points to verify
        // Note: samples must be sorted in ascending order!
        let mut samples = [
            0u64,
            self.chunk_size
                .saturating_sub(1)
                .min(self.n - 1),
            self.chunk_size.min(self.n - 1),
            (self.n / 2).min(self.n - 1),
            self.n - 1,
        ];
        samples.sort();

        let mut prev: Option<CoordScalar> = None;
        for &i in &samples {
            let v = self.scalar_at(i).await?;
            if let Some(p) = &prev {
                let ord = p.partial_cmp(&v);
                let ok = match (dir, ord) {
                    (
                        MonotonicDir::Increasing,
                        Some(
                            std::cmp::Ordering::Less
                            | std::cmp::Ordering::Equal,
                        ),
                    ) => true,
                    (
                        MonotonicDir::Decreasing,
                        Some(
                            std::cmp::Ordering::Greater
                            | std::cmp::Ordering::Equal,
                        ),
                    ) => true,
                    _ => false,
                };
                if !ok {
                    return None;
                }
            }
            prev = Some(v);
        }

        Some(dir)
    }

    async fn lower_bound(
        &self,
        target: &CoordScalar,
        strict: bool,
        dir: MonotonicDir,
    ) -> Option<u64> {
        let mut lo = 0u64;
        let mut hi = self.n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let v = self.scalar_at(mid).await?;
            let cmp = v.partial_cmp(target);

            let go_left = match (dir, strict, cmp) {
                (
                    MonotonicDir::Increasing,
                    false,
                    Some(
                        std::cmp::Ordering::Greater
                        | std::cmp::Ordering::Equal,
                    ),
                ) => true,
                (
                    MonotonicDir::Increasing,
                    true,
                    Some(std::cmp::Ordering::Greater),
                ) => true,
                (
                    MonotonicDir::Decreasing,
                    false,
                    Some(
                        std::cmp::Ordering::Less
                        | std::cmp::Ordering::Equal,
                    ),
                ) => true,
                (
                    MonotonicDir::Decreasing,
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

        Some(lo)
    }

    async fn upper_bound(
        &self,
        target: &CoordScalar,
        strict: bool,
        dir: MonotonicDir,
    ) -> Option<u64> {
        let mut lo = 0u64;
        let mut hi = self.n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let v = self.scalar_at(mid).await?;
            let cmp = v.partial_cmp(target);

            let go_left = match (dir, strict, cmp) {
                (
                    MonotonicDir::Increasing,
                    true,
                    Some(
                        std::cmp::Ordering::Greater
                        | std::cmp::Ordering::Equal,
                    ),
                ) => true,
                (
                    MonotonicDir::Increasing,
                    false,
                    Some(std::cmp::Ordering::Greater),
                ) => true,
                (
                    MonotonicDir::Decreasing,
                    true,
                    Some(
                        std::cmp::Ordering::Less
                        | std::cmp::Ordering::Equal,
                    ),
                ) => true,
                (
                    MonotonicDir::Decreasing,
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

        Some(lo)
    }

    async fn resolve_range(
        &self,
        vr: &ValueRange,
        dir: MonotonicDir,
    ) -> Option<IndexRange> {
        if vr.empty {
            return Some(IndexRange {
                start: 0,
                end_exclusive: 0,
            });
        }

        // Equality case
        if let Some(eq) = &vr.eq {
            let start = self
                .lower_bound(eq, false, dir)
                .await?;
            let end = self
                .upper_bound(eq, false, dir)
                .await?;
            return Some(IndexRange {
                start,
                end_exclusive: end,
            });
        }

        let start = if let Some((v, bk)) = &vr.min
        {
            let strict =
                *bk == BoundKind::Exclusive;
            self.lower_bound(v, strict, dir)
                .await?
        } else {
            0
        };

        let end_exclusive =
            if let Some((v, bk)) = &vr.max {
                let strict =
                    *bk == BoundKind::Exclusive;
                self.upper_bound(v, strict, dir)
                    .await?
            } else {
                self.n
            };

        Some(IndexRange {
            start,
            end_exclusive,
        })
    }
}

/// Resolve all requests for a single dimension.
async fn resolve_dimension_async(
    dim: &IStr,
    requests: Vec<(
        ResolutionRequest,
        ValueRange,
    )>,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
) -> Vec<(ResolutionRequest, Option<IndexRange>)>
{
    let mut results =
        Vec::with_capacity(requests.len());

    // Get array metadata
    let Some(array_meta) = meta.arrays.get(dim)
    else {
        for (req, _) in requests {
            results.push((req, None));
        }
        return results;
    };

    // Create resolver for this dimension
    let Some(resolver) =
        DimResolver::new(store, array_meta).await
    else {
        for (req, _) in requests {
            results.push((req, None));
        }
        return results;
    };

    if resolver.n == 0 {
        for (req, _) in requests {
            results.push((
                req,
                Some(IndexRange {
                    start: 0,
                    end_exclusive: 0,
                }),
            ));
        }
        return results;
    }

    // Check monotonicity
    let Some(dir) =
        resolver.check_monotonic().await
    else {
        for (req, _) in requests {
            results.push((req, None));
        }
        return results;
    };

    // Resolve each request
    for (req, vr) in requests {
        let result = resolver
            .resolve_range(&vr, dir)
            .await;
        results.push((req, result));
    }

    results
}

#[derive(Debug, Clone, Copy)]
enum MonotonicDir {
    Increasing,
    Decreasing,
}
