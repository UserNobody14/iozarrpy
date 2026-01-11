use super::types::{BoundKind, CoordScalar, IndexRange, ValueRange};

pub(super) fn index_range_for_index_dim(vr: &ValueRange, dim_len_est: u64) -> Option<IndexRange> {
    let to_i128 = |v: &CoordScalar| -> Option<i128> {
        match v {
            CoordScalar::I64(x) => Some(*x as i128),
            CoordScalar::U64(x) => Some(*x as i128),
            _ => None,
        }
    };

    let clamp_u64 = |x: i128| -> u64 {
        if x <= 0 {
            0
        } else if (x as u128) >= (u64::MAX as u128) {
            u64::MAX
        } else {
            x as u64
        }
    };

    // Equality: [idx, idx+1)
    if let Some(eq) = &vr.eq {
        let idx = to_i128(eq)?;
        if idx < 0 {
            return Some(IndexRange { start: 0, end_exclusive: 0 });
        }
        let start = clamp_u64(idx).min(dim_len_est);
        let end_exclusive = start.saturating_add(1).min(dim_len_est);
        return Some(IndexRange { start, end_exclusive });
    }

    let start = if let Some((v, bk)) = &vr.min {
        let idx = to_i128(v)?;
        let idx = match bk {
            BoundKind::Inclusive => idx,
            BoundKind::Exclusive => idx.saturating_add(1),
        };
        if idx < 0 {
            0
        } else {
            clamp_u64(idx).min(dim_len_est)
        }
    } else {
        0
    };

    let end_exclusive = if let Some((v, bk)) = &vr.max {
        let idx = to_i128(v)?;
        let end = match bk {
            BoundKind::Inclusive => idx.saturating_add(1),
            BoundKind::Exclusive => idx,
        };
        if end < 0 {
            0
        } else {
            clamp_u64(end).min(dim_len_est)
        }
    } else {
        dim_len_est
    };

    Some(IndexRange { start, end_exclusive })
}
