use crate::chunk_plan::exprs::errors::ResolveError;
use super::monotonic_scalar::{MonotonicCoordResolver, MonotonicDirection};
use super::types::CoordScalar;

impl<'a> MonotonicCoordResolver<'a> {
    pub(super) fn lower_bound(
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

    pub(super) fn upper_bound(
        &mut self,
        dim: &str,
        target: &CoordScalar,
        strict: bool,
        dir: MonotonicDirection,
        n: u64,
    ) -> Result<u64, ResolveError> {
        // Return end_exclusive for max bound:
        // For increasing: first idx with value >= target (strict) or > target (!strict)??? Wait:
        // We want end_exclusive such that values satisfy value < max (Exclusive) or <= max (Inclusive).
        // So compute first idx that violates that, i.e. value >= max (Exclusive) or value > max (Inclusive).
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
                ) => true, // <= max violates for decreasing? symmetric
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

