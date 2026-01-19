//! Async monotonic coordinate resolver.
//!
//! Provides async/concurrent resolution of value ranges to index ranges,
//! enabling parallel fetching of coordinate data across dimensions.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::resolver_traits::{AsyncCoordResolver, HashMapCache, ResolutionCache, ResolutionRequest};
use super::types::{BoundKind, CoordScalar, IndexRange, ValueRange};
use crate::chunk_plan::exprs::literals;
use crate::meta::ZarrDatasetMeta;

use zarrs::array::Array;

/// Async monotonic coordinate resolver.
///
/// Unlike `MonotonicCoordResolver`, this resolver performs async I/O
/// and can resolve multiple requests concurrently.
pub(crate) struct AsyncMonotonicResolver {
    store: zarrs::storage::AsyncReadableWritableListableStorage,
}

impl AsyncMonotonicResolver {
    /// Create a new async resolver.
    pub(crate) fn new(store: zarrs::storage::AsyncReadableWritableListableStorage) -> Self {
        Self { store }
    }
}

#[async_trait::async_trait]
impl AsyncCoordResolver for AsyncMonotonicResolver {
    async fn resolve_batch(
        &self,
        requests: Vec<ResolutionRequest>,
        meta: &ZarrDatasetMeta,
    ) -> Box<dyn ResolutionCache + Send + Sync> {
        let mut cache = HashMapCache::new();

        // Group requests by dimension to minimize array opens
        let mut by_dim: BTreeMap<Arc<str>, Vec<(ResolutionRequest, ValueRange)>> = BTreeMap::new();
        for req in requests {
            by_dim
                .entry(req.dim.clone())
                .or_default()
                .push((req.clone(), req.value_range.clone()));
        }

        // Resolve each dimension concurrently
        let futures: Vec<_> = by_dim
            .into_iter()
            .map(|(dim, reqs)| {
                let store = self.store.clone();
                let meta = meta.clone();
                async move {
                    resolve_dimension_async(&dim, reqs, &meta, store).await
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        // Merge all results into the cache
        for dim_cache in results {
            for (req, result) in dim_cache {
                cache.insert(req, result);
            }
        }

        Box::new(cache)
    }
}

/// Resolve all requests for a single dimension.
async fn resolve_dimension_async(
    dim: &str,
    requests: Vec<(ResolutionRequest, ValueRange)>,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
) -> Vec<(ResolutionRequest, Option<IndexRange>)> {
    let mut results = Vec::with_capacity(requests.len());

    // Get array metadata
    let Some(array_meta) = meta.arrays.get(dim) else {
        // No coordinate array - can't resolve
        for (req, _) in requests {
            results.push((req, None));
        }
        return results;
    };

    if array_meta.shape.len() != 1 {
        // Not a 1D coordinate array
        for (req, _) in requests {
            results.push((req, None));
        }
        return results;
    }

    let n = array_meta.shape[0];
    if n == 0 {
        for (req, _) in requests {
            results.push((req, Some(IndexRange { start: 0, end_exclusive: 0 })));
        }
        return results;
    }

    // Open the array
    let arr = match Array::async_open(store.clone(), &array_meta.path).await {
        Ok(a) => a,
        Err(_) => {
            for (req, _) in requests {
                results.push((req, None));
            }
            return results;
        }
    };

    // Check monotonicity by sampling
    let dir = match check_monotonic_async(&arr, array_meta, n).await {
        Some(d) => d,
        None => {
            for (req, _) in requests {
                results.push((req, None));
            }
            return results;
        }
    };

    // Resolve each request
    for (req, vr) in requests {
        let result = resolve_single_range_async(&arr, array_meta, &vr, dir, n).await;
        results.push((req, result));
    }

    results
}

#[derive(Debug, Clone, Copy)]
enum MonotonicDir {
    Increasing,
    Decreasing,
}

async fn check_monotonic_async(
    arr: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    array_meta: &crate::meta::ZarrArrayMeta,
    n: u64,
) -> Option<MonotonicDir> {
    if n < 2 {
        return Some(MonotonicDir::Increasing);
    }

    // Sample first and last values
    let first = scalar_at_async(arr, array_meta, 0).await.ok()?;
    let last = scalar_at_async(arr, array_meta, n - 1).await.ok()?;

    let dir = match first.partial_cmp(&last) {
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal) => MonotonicDir::Increasing,
        Some(std::cmp::Ordering::Greater) => MonotonicDir::Decreasing,
        None => return None,
    };

    // Sample a few more points to verify
    let chunk_size = arr
        .chunk_shape(&[0])
        .ok()?
        .first()
        .map(|nz| nz.get())
        .unwrap_or(1);

    let samples = [
        0u64,
        chunk_size.saturating_sub(1).min(n - 1),
        chunk_size.min(n - 1),
        (n / 2).min(n - 1),
        n - 1,
    ];

    let mut prev: Option<CoordScalar> = None;
    for &i in &samples {
        let v = scalar_at_async(arr, array_meta, i).await.ok()?;
        if let Some(p) = &prev {
            let ord = p.partial_cmp(&v);
            let ok = match (dir, ord) {
                (MonotonicDir::Increasing, Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)) => true,
                (MonotonicDir::Decreasing, Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)) => true,
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

async fn scalar_at_async(
    arr: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    array_meta: &crate::meta::ZarrArrayMeta,
    idx: u64,
) -> Result<CoordScalar, ()> {
    use zarrs::array_subset::ArraySubset;

    let subset = ArraySubset::new_with_ranges(&[idx..(idx + 1)]);
    let id = arr.data_type().identifier();
    let te = array_meta.time_encoding.as_ref();

    match id {
        "float64" => {
            let v = arr
                .async_retrieve_array_subset::<Vec<f64>>(&subset)
                .await
                .map_err(|_| ())?[0];
            Ok(CoordScalar::F64(v))
        }
        "float32" => {
            let v = arr
                .async_retrieve_array_subset::<Vec<f32>>(&subset)
                .await
                .map_err(|_| ())?[0] as f64;
            Ok(CoordScalar::F64(v))
        }
        "int64" => {
            let raw = arr
                .async_retrieve_array_subset::<Vec<i64>>(&subset)
                .await
                .map_err(|_| ())?[0];
            Ok(literals::apply_time_encoding(raw, te))
        }
        "int32" => {
            let raw = arr
                .async_retrieve_array_subset::<Vec<i32>>(&subset)
                .await
                .map_err(|_| ())?[0] as i64;
            Ok(literals::apply_time_encoding(raw, te))
        }
        "int16" => {
            let raw = arr
                .async_retrieve_array_subset::<Vec<i16>>(&subset)
                .await
                .map_err(|_| ())?[0] as i64;
            Ok(literals::apply_time_encoding(raw, te))
        }
        "int8" => {
            let raw = arr
                .async_retrieve_array_subset::<Vec<i8>>(&subset)
                .await
                .map_err(|_| ())?[0] as i64;
            Ok(literals::apply_time_encoding(raw, te))
        }
        "uint64" => {
            let v = arr
                .async_retrieve_array_subset::<Vec<u64>>(&subset)
                .await
                .map_err(|_| ())?[0];
            Ok(CoordScalar::U64(v))
        }
        "uint32" => {
            let v = arr
                .async_retrieve_array_subset::<Vec<u32>>(&subset)
                .await
                .map_err(|_| ())?[0] as u64;
            Ok(CoordScalar::U64(v))
        }
        "uint16" => {
            let v = arr
                .async_retrieve_array_subset::<Vec<u16>>(&subset)
                .await
                .map_err(|_| ())?[0] as u64;
            Ok(CoordScalar::U64(v))
        }
        "uint8" => {
            let v = arr
                .async_retrieve_array_subset::<Vec<u8>>(&subset)
                .await
                .map_err(|_| ())?[0] as u64;
            Ok(CoordScalar::U64(v))
        }
        _ => Err(()),
    }
}

async fn resolve_single_range_async(
    arr: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    array_meta: &crate::meta::ZarrArrayMeta,
    vr: &ValueRange,
    dir: MonotonicDir,
    n: u64,
) -> Option<IndexRange> {
    if vr.empty {
        return Some(IndexRange { start: 0, end_exclusive: 0 });
    }

    // Equality case
    if let Some(eq) = &vr.eq {
        let start = lower_bound_async(arr, array_meta, eq, false, dir, n).await?;
        let end = upper_bound_async(arr, array_meta, eq, false, dir, n).await?;
        return Some(IndexRange { start, end_exclusive: end });
    }

    let start = if let Some((v, bk)) = &vr.min {
        let strict = *bk == BoundKind::Exclusive;
        lower_bound_async(arr, array_meta, v, strict, dir, n).await?
    } else {
        0
    };

    let end_exclusive = if let Some((v, bk)) = &vr.max {
        let strict = *bk == BoundKind::Exclusive;
        upper_bound_async(arr, array_meta, v, strict, dir, n).await?
    } else {
        n
    };

    Some(IndexRange { start, end_exclusive })
}

async fn lower_bound_async(
    arr: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    array_meta: &crate::meta::ZarrArrayMeta,
    target: &CoordScalar,
    strict: bool,
    dir: MonotonicDir,
    n: u64,
) -> Option<u64> {
    let mut lo = 0u64;
    let mut hi = n;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let v = scalar_at_async(arr, array_meta, mid).await.ok()?;
        let cmp = v.partial_cmp(target);

        let go_left = match (dir, strict, cmp) {
            (MonotonicDir::Increasing, false, Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)) => true,
            (MonotonicDir::Increasing, true, Some(std::cmp::Ordering::Greater)) => true,
            (MonotonicDir::Decreasing, false, Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)) => true,
            (MonotonicDir::Decreasing, true, Some(std::cmp::Ordering::Less)) => true,
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

async fn upper_bound_async(
    arr: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    array_meta: &crate::meta::ZarrArrayMeta,
    target: &CoordScalar,
    strict: bool,
    dir: MonotonicDir,
    n: u64,
) -> Option<u64> {
    let mut lo = 0u64;
    let mut hi = n;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let v = scalar_at_async(arr, array_meta, mid).await.ok()?;
        let cmp = v.partial_cmp(target);

        let go_left = match (dir, strict, cmp) {
            (MonotonicDir::Increasing, true, Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)) => true,
            (MonotonicDir::Increasing, false, Some(std::cmp::Ordering::Greater)) => true,
            (MonotonicDir::Decreasing, true, Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)) => true,
            (MonotonicDir::Decreasing, false, Some(std::cmp::Ordering::Less)) => true,
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
