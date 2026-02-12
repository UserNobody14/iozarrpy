use std::hash::{Hash, Hasher};

use smallvec::SmallVec;

use crate::IStr;
use polars::prelude::Operator;

/// Chunk grid signature - dimensions + chunk shape for grouping.
///
/// Variables with the same chunk grid signature can share chunk iteration.
/// This extends the old DimSignature by also including chunk shape,
/// so variables with same dims but different chunking are handled separately.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
pub struct ChunkGridSignature {
    /// Dimension names (ordered)
    dims: SmallVec<[IStr; 4]>,
    /// Chunk shape per dimension (determines grid layout)
    chunk_shape: SmallVec<[u64; 4]>,
}

impl ChunkGridSignature {
    /// Create a new chunk grid signature from dimension names and chunk shape.
    pub fn new(
        dims: impl Into<SmallVec<[IStr; 4]>>,
        chunk_shape: impl Into<SmallVec<[u64; 4]>>,
    ) -> Self {
        Self {
            dims: dims.into(),
            chunk_shape: chunk_shape.into(),
        }
    }

    /// Create a signature from dims only (with empty chunk shape).
    /// Used during lazy compilation when chunk info isn't known yet.
    pub fn from_dims_only(
        dims: impl Into<SmallVec<[IStr; 4]>>,
    ) -> Self {
        Self {
            dims: dims.into(),
            chunk_shape: SmallVec::new(),
        }
    }

    /// Get the dimension names.
    pub fn dims(&self) -> &[IStr] {
        &self.dims
    }

    /// Get the chunk shape.
    pub fn chunk_shape(&self) -> &[u64] {
        &self.chunk_shape
    }
}

impl From<&[IStr]> for ChunkGridSignature {
    fn from(dims: &[IStr]) -> Self {
        Self::from_dims_only(
            dims.iter()
                .cloned()
                .collect::<SmallVec<[IStr; 4]>>(),
        )
    }
}

/// Type alias for backwards compatibility during transition.
/// DimSignature is now ChunkGridSignature (with optional chunk_shape).
pub type DimSignature = ChunkGridSignature;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash,
)]
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
            (
                CoordScalar::I64(a),
                CoordScalar::I64(b),
            ) => a == b,
            (
                CoordScalar::U64(a),
                CoordScalar::U64(b),
            ) => a == b,
            (
                CoordScalar::F64(a),
                CoordScalar::F64(b),
            ) => a.to_bits() == b.to_bits(),
            (
                CoordScalar::DatetimeNs(a),
                CoordScalar::DatetimeNs(b),
            ) => a == b,
            (
                CoordScalar::DurationNs(a),
                CoordScalar::DurationNs(b),
            ) => a == b,
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
            CoordScalar::F64(v) => {
                v.to_bits().hash(state)
            }
            CoordScalar::DatetimeNs(v) => {
                v.hash(state)
            }
            CoordScalar::DurationNs(v) => {
                v.hash(state)
            }
        }
    }
}

impl CoordScalar {
    fn as_i128_orderable(&self) -> Option<i128> {
        match self {
            CoordScalar::I64(v) => {
                Some(*v as i128)
            }
            CoordScalar::U64(v) => {
                Some(*v as i128)
            }
            CoordScalar::DatetimeNs(v) => {
                Some(*v as i128)
            }
            CoordScalar::DurationNs(v) => {
                Some(*v as i128)
            }
            CoordScalar::F64(_) => None,
        }
    }

    pub(crate) fn partial_cmp(
        &self,
        other: &CoordScalar,
    ) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (
                CoordScalar::F64(a),
                CoordScalar::F64(b),
            ) => a.partial_cmp(b),
            (CoordScalar::F64(a), b) => {
                Some((*a).partial_cmp(
                    &(b.as_i128_orderable()?
                        as f64),
                )?)
            }
            (a, CoordScalar::F64(b)) => Some(
                (a.as_i128_orderable()? as f64)
                    .partial_cmp(b)?,
            ),
            _ => Some(
                self.as_i128_orderable()?.cmp(
                    &other.as_i128_orderable()?,
                ),
            ),
        }
    }
}

impl PartialOrd for CoordScalar {
    fn partial_cmp(
        &self,
        other: &CoordScalar,
    ) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (
                CoordScalar::F64(a),
                CoordScalar::F64(b),
            ) => a.partial_cmp(b),
            (CoordScalar::F64(a), b) => {
                Some((*a).partial_cmp(
                    &(b.as_i128_orderable()?
                        as f64),
                )?)
            }
            (a, CoordScalar::F64(b)) => Some(
                (a.as_i128_orderable()? as f64)
                    .partial_cmp(b)?,
            ),
            _ => Some(
                self.as_i128_orderable()?.cmp(
                    &other.as_i128_orderable()?,
                ),
            ),
        }
    }
}

#[derive(
    Debug, Clone, Default, PartialEq, Eq, Hash,
)]
pub(crate) struct ValueRangePresent {
    min: Option<(CoordScalar, BoundKind)>,
    max: Option<(CoordScalar, BoundKind)>,
}

impl ValueRangePresent {
    pub(crate) fn from_equal_case(
        eq: CoordScalar,
    ) -> Self {
        Self {
            min: Some((
                eq.clone(),
                BoundKind::Inclusive,
            )),
            max: Some((
                eq.clone(),
                BoundKind::Inclusive,
            )),
            ..Default::default()
        }
    }

    pub(crate) fn from_min_only(
        min: CoordScalar,
        bound_kind: BoundKind,
    ) -> Self {
        Self {
            min: Some((min, bound_kind)),
            ..Default::default()
        }
    }

    pub(crate) fn from_max_only(
        max: CoordScalar,
        bound_kind: BoundKind,
    ) -> Self {
        Self {
            max: Some((max, bound_kind)),
            ..Default::default()
        }
    }

    pub(crate) fn in_range(
        &self,
        value: CoordScalar,
    ) -> bool {
        let min = self.min.as_ref();
        let max = self.max.as_ref();
        if let (
            Some((min_v, min_k)),
            Some((max_v, max_k)),
        ) = (min, max)
        {
            match (min_k, max_k) {
                (
                    BoundKind::Inclusive,
                    BoundKind::Inclusive,
                ) => {
                    &value >= min_v
                        && value <= *max_v
                }
                (
                    BoundKind::Inclusive,
                    BoundKind::Exclusive,
                ) => {
                    &value >= min_v
                        && value < *max_v
                }
                (
                    BoundKind::Exclusive,
                    BoundKind::Inclusive,
                ) => {
                    &value > min_v
                        && value <= *max_v
                }
                (
                    BoundKind::Exclusive,
                    BoundKind::Exclusive,
                ) => {
                    &value > min_v
                        && value < *max_v
                }
            }
        } else {
            false
        }
    }

    pub(crate) fn from_polars_op(
        op: Operator,
        scalar: CoordScalar,
    ) -> Option<Self> {
        match op {
            Operator::Eq => Some(
                ValueRangePresent::from_equal_case(
                    scalar,
                ),
            ),
            Operator::Gt => Some(
                ValueRangePresent::from_min_only(
                    scalar,
                    BoundKind::Exclusive,
                ),
            ),
            Operator::GtEq => Some(
                ValueRangePresent::from_min_only(
                    scalar,
                    BoundKind::Inclusive,
                ),
            ),
            Operator::Lt => Some(
                ValueRangePresent::from_max_only(
                    scalar,
                    BoundKind::Exclusive,
                ),
            ),
            Operator::LtEq => Some(
                ValueRangePresent::from_max_only(
                    scalar,
                    BoundKind::Inclusive,
                ),
            ),
            _ => None,
        }
    }

    pub(crate) fn index_range_for_index_dim(
        &self,
        dim_len_est: u64,
    ) -> Option<std::ops::Range<u64>> {
        let to_i128 =
            |v: &CoordScalar| -> Option<i128> {
                match v {
                    CoordScalar::I64(x) => {
                        Some(*x as i128)
                    }
                    CoordScalar::U64(x) => {
                        Some(*x as i128)
                    }
                    CoordScalar::F64(_)
                    | CoordScalar::DatetimeNs(
                        _,
                    )
                    | CoordScalar::DurationNs(
                        _,
                    ) => None,
                }
            };

        let clamp_u64 = |x: i128| -> u64 {
            if x <= 0 {
                0
            } else if (x as u128)
                >= (u64::MAX as u128)
            {
                u64::MAX
            } else {
                x as u64
            }
        };

        // Equality: [idx, idx+1)
        if let Some(eq) = self.equal_case() {
            let idx = to_i128(&eq)?;
            if idx < 0 {
                return Some(0..0);
            }
            let start =
                clamp_u64(idx).min(dim_len_est);
            let end_exclusive = start
                .saturating_add(1)
                .min(dim_len_est);
            return Some(start..end_exclusive);
        }

        let start = if let Some((v, bk)) =
            &self.min
        {
            let idx = to_i128(v)?;
            let idx = match bk {
                BoundKind::Inclusive => idx,
                BoundKind::Exclusive => {
                    idx.saturating_add(1)
                }
            };
            if idx < 0 {
                0
            } else {
                clamp_u64(idx).min(dim_len_est)
            }
        } else {
            0
        };

        let end_exclusive = if let Some((v, bk)) =
            &self.max
        {
            let idx = to_i128(v)?;
            let end = match bk {
                BoundKind::Inclusive => {
                    idx.saturating_add(1)
                }
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

        Some(start..end_exclusive)
    }

    pub(crate) fn to_max_case(
        &self,
    ) -> Option<(CoordScalar, BoundKind)> {
        self.max
            .as_ref()
            .map(|(v, k)| (v.clone(), k.clone()))
    }

    pub(crate) fn to_min_case(
        &self,
    ) -> Option<(CoordScalar, BoundKind)> {
        self.min
            .as_ref()
            .map(|(v, k)| (v.clone(), k.clone()))
    }
}
pub(crate) type ValueRange =
    Option<ValueRangePresent>;

pub(crate) trait HasIntersect:
    Sized
{
    fn intersect(
        &self,
        other: Option<Self>,
    ) -> Option<Self>;
}

pub(crate) trait HasEqualCase {
    fn equal_case(&self) -> Option<CoordScalar>;
}

impl HasEqualCase for ValueRangePresent {
    fn equal_case(&self) -> Option<CoordScalar> {
        let min = self.min.as_ref();
        let max = self.max.as_ref();
        if let (
            Some((min_v, min_k)),
            Some((max_v, max_k)),
        ) = (min, max)
        {
            if min_v == max_v
                && matches!(
                    min_k,
                    BoundKind::Inclusive
                )
                && matches!(
                    max_k,
                    BoundKind::Inclusive
                )
            {
                Some(min_v.clone())
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl HasIntersect for ValueRangePresent {
    fn intersect(
        &self,
        other_range: Option<Self>,
    ) -> Option<Self> {
        if let Some(other) = other_range {
            let out;
            match (
                self.equal_case(),
                other.equal_case(),
            ) {
                (Some(eq), Some(other_eq)) => {
                    if eq != other_eq {
                        return None;
                    } else {
                        out = ValueRangePresent::from_equal_case(eq);
                    }
                }
                (Some(eq), None) => {
                    // If we have an equality constraint, ensure it's compatible with min/max.
                    if !other.in_range(eq.clone())
                    {
                        return None;
                    }
                    out = ValueRangePresent::from_equal_case(eq);
                }
                (None, Some(other_eq)) => {
                    if !self.in_range(
                        other_eq.clone(),
                    ) {
                        return None;
                    }
                    out = ValueRangePresent::from_equal_case(other_eq)
                }
                (None, None) => {
                    out = ValueRangePresent {
                        min: pick_tighter_min(
                            self.min.clone(),
                            other.min.clone(),
                        ),
                        max: pick_tighter_max(
                            self.max.clone(),
                            other.max.clone(),
                        ),
                    };
                }
            }
            Some(out)
        } else {
            None
        }
    }
}

impl HasIntersect for Option<ValueRangePresent> {
    fn intersect(
        &self,
        other: Option<Self>,
    ) -> Option<Self> {
        match (self, other) {
            (None, None) => None,
            (Some(a), None) => {
                Some(Some(a.clone()))
            }
            (None, Some(a)) => Some(a),
            (Some(a), Some(b)) => {
                Some(a.intersect(Some(b?)))
            }
        }
    }
}

impl HasIntersect
    for std::sync::Arc<Option<ValueRangePresent>>
{
    fn intersect(
        &self,
        other: Option<Self>,
    ) -> Option<Self> {
        self.as_ref()
            .intersect(other.map(|o| {
                o.as_ref()
                    .as_ref()
                    .map(|o| o.clone())
            }))
            .map(|o| std::sync::Arc::new(o))
    }
}

fn pick_tighter_min(
    a: Option<(CoordScalar, BoundKind)>,
    b: Option<(CoordScalar, BoundKind)>,
) -> Option<(CoordScalar, BoundKind)> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some((av, ak)), Some((bv, bk))) => {
            match av.partial_cmp(&bv) {
                Some(std::cmp::Ordering::Less) => {
                    Some((bv, bk))
                }
                Some(std::cmp::Ordering::Greater) => {
                    Some((av, ak))
                }
                Some(std::cmp::Ordering::Equal) => Some((
                    av,
                    if ak == BoundKind::Exclusive
                        || bk == BoundKind::Exclusive
                    {
                        BoundKind::Exclusive
                    } else {
                        BoundKind::Inclusive
                    },
                )),
                None => None,
            }
        }
    }
}

fn pick_tighter_max(
    a: Option<(CoordScalar, BoundKind)>,
    b: Option<(CoordScalar, BoundKind)>,
) -> Option<(CoordScalar, BoundKind)> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some((av, ak)), Some((bv, bk))) => {
            match av.partial_cmp(&bv) {
                Some(std::cmp::Ordering::Less) => {
                    Some((av, ak))
                }
                Some(std::cmp::Ordering::Greater) => {
                    Some((bv, bk))
                }
                Some(std::cmp::Ordering::Equal) => Some((
                    av,
                    if ak == BoundKind::Exclusive
                        || bk == BoundKind::Exclusive
                    {
                        BoundKind::Exclusive
                    } else {
                        BoundKind::Inclusive
                    },
                )),
                None => None,
            }
        }
    }
}
