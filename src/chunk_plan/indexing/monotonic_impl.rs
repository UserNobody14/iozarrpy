use crate::chunk_plan::exprs::errors::{CoordIndexResolver, ResolveError};
use super::monotonic_scalar::MonotonicCoordResolver;
use crate::chunk_plan::prelude::*;
use super::types::{BoundKind, IndexRange, ValueRange};
use super::resolver_traits::{HashMapCache, ResolutionCache, ResolutionRequest, SyncCoordResolver};

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
            // strict means "< v" so violation at >= v; non-strict means "<= v" so violation at > v.
            self.upper_bound(dim, v, strict, dir, n)?
        } else {
            n
        };

        Ok(Some(IndexRange {
            start,
            end_exclusive,
        }))
    }

    fn coord_read_count(&self) -> u64 {
        self.read_count.load(Ordering::Relaxed)
    }
}


// ============================================================================
// SyncCoordResolver implementation for batched resolution
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

            // Use the existing CoordIndexResolver implementation
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
