use super::errors::{CoordIndexResolver, ResolveError};
use super::monotonic_scalar::MonotonicCoordResolver;
use super::prelude::*;
use super::types::{BoundKind, CoordScalar, IndexRange, ValueRange};

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

    fn index_for_value(
        &mut self,
        dim: &str,
        value: &CoordScalar,
    ) -> Result<Option<u64>, ResolveError> {
        let Some(meta) = self.meta.arrays.get(dim) else {
            return Ok(None);
        };
        if meta.shape.len() != 1 {
            return Ok(None);
        }
        let n = meta.shape[0];
        let Some(dir) = self.ensure_monotonic(dim)? else {
            return Ok(None);
        };
        let start = self.lower_bound(dim, value, false, dir, n)?;
        let end = self.upper_bound(dim, value, false, dir, n)?;
        if start < end {
            Ok(Some(start))
        } else {
            Ok(None)
        }
    }

    fn coord_read_count(&self) -> u64 {
        self.read_count.load(Ordering::Relaxed)
    }
}

