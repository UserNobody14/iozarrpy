//! Backend-based async coordinate resolver.
//!
//! This resolver uses the generic `ChunkedDataBackendAsync` trait for coordinate
//! data access, enabling backends that don't expose a raw store to still support
//! predicate pushdown.

use std::collections::BTreeMap;
use std::sync::Arc;
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
use crate::backend::traits::ChunkedDataBackendAsync;
use crate::chunk_plan::exprs;
use crate::meta::{ZarrDatasetMeta, ZarrMeta};
use crate::reader::ColumnData;

/// Cached chunk data for a coordinate array (converted from ColumnData).
#[derive(Debug, Clone)]
enum ChunkData {
    F64(Vec<f64>),
    I64(Vec<i64>),
    U64(Vec<u64>),
}

impl ChunkData {
    fn from_column_data(
        cd: ColumnData,
    ) -> Option<Self> {
        match cd {
            ColumnData::F64(v) => {
                Some(ChunkData::F64(v))
            }
            ColumnData::F32(v) => {
                Some(ChunkData::F64(
                    v.into_iter()
                        .map(|x| x as f64)
                        .collect(),
                ))
            }
            ColumnData::I64(v) => {
                Some(ChunkData::I64(v))
            }
            ColumnData::I32(v) => {
                Some(ChunkData::I64(
                    v.into_iter()
                        .map(|x| x as i64)
                        .collect(),
                ))
            }
            ColumnData::I16(v) => {
                Some(ChunkData::I64(
                    v.into_iter()
                        .map(|x| x as i64)
                        .collect(),
                ))
            }
            ColumnData::I8(v) => {
                Some(ChunkData::I64(
                    v.into_iter()
                        .map(|x| x as i64)
                        .collect(),
                ))
            }
            ColumnData::U64(v) => {
                Some(ChunkData::U64(v))
            }
            ColumnData::U32(v) => {
                Some(ChunkData::U64(
                    v.into_iter()
                        .map(|x| x as u64)
                        .collect(),
                ))
            }
            ColumnData::U16(v) => {
                Some(ChunkData::U64(
                    v.into_iter()
                        .map(|x| x as u64)
                        .collect(),
                ))
            }
            ColumnData::U8(v) => {
                Some(ChunkData::U64(
                    v.into_iter()
                        .map(|x| x as u64)
                        .collect(),
                ))
            }
            _ => None,
        }
    }

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

/// Monotonicity direction.
#[derive(Debug, Clone, Copy)]
enum MonotonicDir {
    Increasing,
    Decreasing,
}

/// Backend-based async monotonic coordinate resolver.
///
/// Uses the generic `ChunkedDataBackendAsync` trait for data access,
/// enabling any backend implementation to support predicate pushdown.
pub(crate) struct BackendAsyncResolver<
    B: ChunkedDataBackendAsync,
> {
    backend: Arc<B>,
    meta: Arc<ZarrMeta>,
}

impl<B: ChunkedDataBackendAsync>
    BackendAsyncResolver<B>
{
    /// Create a new backend-based resolver.
    pub(crate) fn new(
        backend: Arc<B>,
        meta: Arc<ZarrMeta>,
    ) -> Self {
        Self { backend, meta }
    }
}

#[async_trait::async_trait]
impl<B: ChunkedDataBackendAsync + 'static>
    AsyncCoordResolver
    for BackendAsyncResolver<B>
{
    async fn resolve_batch(
        &self,
        requests: Vec<ResolutionRequest>,
        legacy_meta: &ZarrDatasetMeta,
    ) -> Box<dyn ResolutionCache + Send + Sync>
    {
        let mut cache = HashMapCache::new();

        // Group requests by dimension
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

        // Resolve each dimension
        for (dim, reqs) in by_dim {
            let results = self
                .resolve_dimension(
                    &dim,
                    reqs,
                    legacy_meta,
                )
                .await;
            for (req, result) in results {
                cache.insert(req, result);
            }
        }

        Box::new(cache)
    }
}

impl<B: ChunkedDataBackendAsync>
    BackendAsyncResolver<B>
{
    async fn resolve_dimension(
        &self,
        dim: &IStr,
        reqs: Vec<(
            ResolutionRequest,
            ValueRange,
        )>,
        legacy_meta: &ZarrDatasetMeta,
    ) -> Vec<(
        ResolutionRequest,
        Option<IndexRange>,
    )> {
        // Get coordinate array metadata
        let coord_meta = match legacy_meta
            .arrays
            .get(dim)
        {
            Some(m) => m,
            None => {
                return reqs
                    .into_iter()
                    .map(|(req, _)| (req, None))
                    .collect();
            }
        };

        if coord_meta.shape.len() != 1 {
            return reqs
                .into_iter()
                .map(|(req, _)| (req, None))
                .collect();
        }

        let n = coord_meta.shape[0];
        if n == 0 {
            return reqs
                .into_iter()
                .map(|(req, _)| {
                    (
                        req,
                        Some(IndexRange {
                            start: 0,
                            end_exclusive: 0,
                        }),
                    )
                })
                .collect();
        }

        let chunk_size = coord_meta
            .chunk_shape
            .first()
            .copied()
            .unwrap_or(n);
        let time_enc =
            coord_meta.time_encoding.clone();

        // Create dimension resolver
        let resolver = DimResolverBackend {
            backend: self.backend.clone(),
            coord_path: coord_meta.path.clone(),
            chunk_size,
            n,
            time_enc,
            chunk_cache: RwLock::new(
                BTreeMap::new(),
            ),
        };

        // Check monotonicity
        let Some(dir) =
            resolver.check_monotonic().await
        else {
            return reqs
                .into_iter()
                .map(|(req, _)| (req, None))
                .collect();
        };

        // Resolve all requests for this dimension
        let mut results =
            Vec::with_capacity(reqs.len());
        for (req, vr) in reqs {
            let result = resolver
                .resolve_range(&vr, dir)
                .await;
            results.push((req, result));
        }

        results
    }
}

/// Per-dimension resolver with backend-based chunk reading.
struct DimResolverBackend<
    B: ChunkedDataBackendAsync,
> {
    backend: Arc<B>,
    coord_path: IStr,
    chunk_size: u64,
    n: u64,
    time_enc: Option<crate::meta::TimeEncoding>,
    chunk_cache: RwLock<BTreeMap<u64, ChunkData>>,
}

impl<B: ChunkedDataBackendAsync>
    DimResolverBackend<B>
{
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

        // Fetch chunk via backend
        let chunk_data = self
            .backend
            .read_chunk_async(
                &self.coord_path,
                &[chunk_idx],
            )
            .await
            .ok()?;

        let data = ChunkData::from_column_data(
            chunk_data,
        )?;

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
        let offset =
            (idx % self.chunk_size) as usize;

        self.load_chunk(chunk_idx).await?;

        let cache = self.chunk_cache.read().await;
        let chunk = cache.get(&chunk_idx)?;
        chunk.get(offset, self.time_enc.as_ref())
    }

    async fn check_monotonic(
        &self,
    ) -> Option<MonotonicDir> {
        if self.n < 2 {
            return Some(
                MonotonicDir::Increasing,
            );
        }

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

        // Verify with sample points
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
                        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal),
                    ) => true,
                    (
                        MonotonicDir::Decreasing,
                        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
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
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
                ) => true,
                (
                    MonotonicDir::Increasing,
                    true,
                    Some(std::cmp::Ordering::Greater),
                ) => true,
                (
                    MonotonicDir::Decreasing,
                    false,
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal),
                ) => true,
                (MonotonicDir::Decreasing, true, Some(std::cmp::Ordering::Less)) => {
                    true
                }
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

            let go_right = match (dir, strict, cmp) {
                (
                    MonotonicDir::Increasing,
                    false,
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal),
                ) => true,
                (MonotonicDir::Increasing, true, Some(std::cmp::Ordering::Less)) => {
                    true
                }
                (
                    MonotonicDir::Decreasing,
                    false,
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
                ) => true,
                (
                    MonotonicDir::Decreasing,
                    true,
                    Some(std::cmp::Ordering::Greater),
                ) => true,
                _ => false,
            };

            if go_right {
                lo = mid + 1;
            } else {
                hi = mid;
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
