#[derive(Debug, Clone)]
pub(crate) struct ChunkId {
    pub(crate) indices: Vec<u64>,
    pub(crate) origin: Vec<u64>,
    pub(crate) shape: Vec<u64>,
}

use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BoundKind {
    Inclusive,
    Exclusive,
}

#[derive(Debug, Clone)]
pub(crate) enum CoordScalar {
    I64(i64),
    U64(u64),
    F64(f64),
    /// Nanoseconds since unix epoch.
    DatetimeNs(i64),
    /// Nanoseconds duration.
    DurationNs(i64),
    // Reserved for future: String/Binary/Categorical/etc.
}

impl PartialEq for CoordScalar {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CoordScalar::I64(a), CoordScalar::I64(b)) => a == b,
            (CoordScalar::U64(a), CoordScalar::U64(b)) => a == b,
            (CoordScalar::F64(a), CoordScalar::F64(b)) => a.to_bits() == b.to_bits(),
            (CoordScalar::DatetimeNs(a), CoordScalar::DatetimeNs(b)) => a == b,
            (CoordScalar::DurationNs(a), CoordScalar::DurationNs(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for CoordScalar {}

impl Hash for CoordScalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            CoordScalar::I64(v) => v.hash(state),
            CoordScalar::U64(v) => v.hash(state),
            CoordScalar::F64(v) => v.to_bits().hash(state),
            CoordScalar::DatetimeNs(v) => v.hash(state),
            CoordScalar::DurationNs(v) => v.hash(state),
        }
    }
}

impl CoordScalar {
    fn as_i128_orderable(&self) -> Option<i128> {
        match self {
            CoordScalar::I64(v) => Some(*v as i128),
            CoordScalar::U64(v) => Some(*v as i128),
            CoordScalar::DatetimeNs(v) => Some(*v as i128),
            CoordScalar::DurationNs(v) => Some(*v as i128),
            CoordScalar::F64(_) => None,
        }
    }

    pub(crate) fn partial_cmp(&self, other: &CoordScalar) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (CoordScalar::F64(a), CoordScalar::F64(b)) => a.partial_cmp(b),
            (CoordScalar::F64(a), b) => Some((*a).partial_cmp(&(b.as_i128_orderable()? as f64))?),
            (a, CoordScalar::F64(b)) => Some((a.as_i128_orderable()? as f64).partial_cmp(b)?),
            _ => Some(self.as_i128_orderable()?.cmp(&other.as_i128_orderable()?)),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub(crate) struct ValueRange {
    pub(crate) min: Option<(CoordScalar, BoundKind)>,
    pub(crate) max: Option<(CoordScalar, BoundKind)>,
    pub(crate) eq: Option<CoordScalar>,
    pub(crate) empty: bool,
}

impl ValueRange {
    pub(crate) fn intersect(&self, other: &ValueRange) -> ValueRange {
        if self.empty || other.empty {
            return ValueRange {
                empty: true,
                ..Default::default()
            };
        }
        let mut out = ValueRange::default();
        out.eq = match (&self.eq, &other.eq) {
            (Some(a), Some(b)) if a == b => Some(a.clone()),
            (Some(_), Some(_)) => {
                out.empty = true;
                return out;
            }
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
        };

        out.min = pick_tighter_min(self.min.clone(), other.min.clone());
        out.max = pick_tighter_max(self.max.clone(), other.max.clone());

        // If we have an equality constraint, ensure it's compatible with min/max.
        if let Some(eq) = &out.eq {
            if let Some((min_v, min_k)) = &out.min {
                let ord = eq.partial_cmp(min_v);
                let ok = match (ord, min_k) {
                    (Some(std::cmp::Ordering::Greater), _) => true,
                    (Some(std::cmp::Ordering::Equal), BoundKind::Inclusive) => true,
                    _ => false,
                };
                if !ok {
                    out.empty = true;
                    return out;
                }
            }
            if let Some((max_v, max_k)) = &out.max {
                let ord = eq.partial_cmp(max_v);
                let ok = match (ord, max_k) {
                    (Some(std::cmp::Ordering::Less), _) => true,
                    (Some(std::cmp::Ordering::Equal), BoundKind::Inclusive) => true,
                    _ => false,
                };
                if !ok {
                    out.empty = true;
                    return out;
                }
            }
        }

        out
    }
}

fn pick_tighter_min(
    a: Option<(CoordScalar, BoundKind)>,
    b: Option<(CoordScalar, BoundKind)>,
) -> Option<(CoordScalar, BoundKind)> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some((av, ak)), Some((bv, bk))) => match av.partial_cmp(&bv) {
            Some(std::cmp::Ordering::Less) => Some((bv, bk)),
            Some(std::cmp::Ordering::Greater) => Some((av, ak)),
            Some(std::cmp::Ordering::Equal) => Some((
                av,
                if ak == BoundKind::Exclusive || bk == BoundKind::Exclusive {
                    BoundKind::Exclusive
                } else {
                    BoundKind::Inclusive
                },
            )),
            None => None,
        },
    }
}

fn pick_tighter_max(
    a: Option<(CoordScalar, BoundKind)>,
    b: Option<(CoordScalar, BoundKind)>,
) -> Option<(CoordScalar, BoundKind)> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some((av, ak)), Some((bv, bk))) => match av.partial_cmp(&bv) {
            Some(std::cmp::Ordering::Less) => Some((av, ak)),
            Some(std::cmp::Ordering::Greater) => Some((bv, bk)),
            Some(std::cmp::Ordering::Equal) => Some((
                av,
                if ak == BoundKind::Exclusive || bk == BoundKind::Exclusive {
                    BoundKind::Exclusive
                } else {
                    BoundKind::Inclusive
                },
            )),
            None => None,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct IndexRange {
    pub(crate) start: u64,
    pub(crate) end_exclusive: u64,
}

impl IndexRange {
    pub(crate) fn is_empty(&self) -> bool {
        self.end_exclusive <= self.start
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DimChunkRange {
    pub(crate) start_chunk: u64,
    pub(crate) end_chunk_inclusive: u64,
}

impl DimChunkRange {
    pub(crate) fn intersect(&self, other: &DimChunkRange) -> Option<DimChunkRange> {
        let s = self.start_chunk.max(other.start_chunk);
        let e = self.end_chunk_inclusive.min(other.end_chunk_inclusive);
        if e < s {
            None
        } else {
            Some(DimChunkRange {
                start_chunk: s,
                end_chunk_inclusive: e,
            })
        }
    }
}

