use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::ops::{Bound, RangeBounds};

use smallvec::SmallVec;

use crate::IStr;
use crate::errors::{
    BackendError, BackendResult,
};
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

impl Display for ChunkGridSignature {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "ChunkGridSignature(")?;
        write!(f, "dims: {:?}", self.dims)?;
        write!(
            f,
            "chunk_shape: {:?}",
            self.chunk_shape
        )?;
        write!(f, ")")
    }
}

/// Type alias for backwards compatibility during transition.
/// DimSignature is now ChunkGridSignature (with optional chunk_shape).
pub type DimSignature = ChunkGridSignature;

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

pub(crate) type CoordBound = Bound<CoordScalar>;

/// A value-space range constraint, stored as a (start, end) pair of `Bound`s.
///
/// Implements `RangeBounds<CoordScalar>` and can be constructed from any
/// standard Rust range type (`Range`, `RangeFrom`, `RangeToInclusive`, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ValueRangePresent(
    pub(crate) CoordBound,
    pub(crate) CoordBound,
);

impl RangeBounds<CoordScalar>
    for ValueRangePresent
{
    fn start_bound(&self) -> Bound<&CoordScalar> {
        self.0.as_ref()
    }
    fn end_bound(&self) -> Bound<&CoordScalar> {
        self.1.as_ref()
    }
}

impl From<std::ops::Range<CoordScalar>>
    for ValueRangePresent
{
    fn from(
        r: std::ops::Range<CoordScalar>,
    ) -> Self {
        Self(
            Bound::Included(r.start),
            Bound::Excluded(r.end),
        )
    }
}

impl From<std::ops::RangeInclusive<CoordScalar>>
    for ValueRangePresent
{
    fn from(
        r: std::ops::RangeInclusive<CoordScalar>,
    ) -> Self {
        let (start, end) = r.into_inner();
        Self(
            Bound::Included(start),
            Bound::Included(end),
        )
    }
}

impl From<std::ops::RangeFrom<CoordScalar>>
    for ValueRangePresent
{
    fn from(
        r: std::ops::RangeFrom<CoordScalar>,
    ) -> Self {
        Self(
            Bound::Included(r.start),
            Bound::Unbounded,
        )
    }
}

impl From<std::ops::RangeTo<CoordScalar>>
    for ValueRangePresent
{
    fn from(
        r: std::ops::RangeTo<CoordScalar>,
    ) -> Self {
        Self(
            Bound::Unbounded,
            Bound::Excluded(r.end),
        )
    }
}

impl From<std::ops::RangeToInclusive<CoordScalar>>
    for ValueRangePresent
{
    fn from(
        r: std::ops::RangeToInclusive<
            CoordScalar,
        >,
    ) -> Self {
        Self(
            Bound::Unbounded,
            Bound::Included(r.end),
        )
    }
}

impl From<std::ops::RangeFull>
    for ValueRangePresent
{
    fn from(_: std::ops::RangeFull) -> Self {
        Self(Bound::Unbounded, Bound::Unbounded)
    }
}

impl From<(CoordBound, CoordBound)>
    for ValueRangePresent
{
    fn from(
        (start, end): (CoordBound, CoordBound),
    ) -> Self {
        Self(start, end)
    }
}

pub(crate) trait HasCoordBound {
    fn get_scalar(&self) -> Option<CoordScalar>;
    fn is_exclusive(&self) -> bool;
}

impl HasCoordBound for CoordBound {
    fn get_scalar(&self) -> Option<CoordScalar> {
        match self {
            CoordBound::Included(scalar) => {
                Some(scalar.clone())
            }
            CoordBound::Excluded(scalar) => {
                Some(scalar.clone())
            }
            CoordBound::Unbounded => None,
        }
    }

    fn is_exclusive(&self) -> bool {
        match self {
            CoordBound::Included(_) => false,
            CoordBound::Excluded(_) => true,
            CoordBound::Unbounded => false,
        }
    }
}

impl ValueRangePresent {
    /// Point equality: `value == eq`.
    pub(crate) fn from_equal_case(
        eq: CoordScalar,
    ) -> Self {
        Self(
            Bound::Included(eq.clone()),
            Bound::Included(eq),
        )
    }

    /// Build from optional start/end bounds.
    /// Returns `None` only when both bounds are `None`
    /// (i.e. no constraint could be determined).
    pub(crate) fn from_option_bounds(
        min: Option<CoordBound>,
        max: Option<CoordBound>,
    ) -> Option<Self> {
        match (min, max) {
            (Some(min), Some(max)) => {
                Some(Self(min, max))
            }
            (Some(min), None) => {
                Some(Self(min, Bound::Unbounded))
            }
            (None, Some(max)) => {
                Some(Self(Bound::Unbounded, max))
            }
            (None, None) => None,
        }
    }

    pub(crate) fn from_min_exclusive(
        min: CoordScalar,
    ) -> Self {
        Self(
            Bound::Excluded(min),
            Bound::Unbounded,
        )
    }

    pub(crate) fn from_max_exclusive(
        max: CoordScalar,
    ) -> Self {
        Self(
            Bound::Unbounded,
            Bound::Excluded(max),
        )
    }

    pub(crate) fn from_min_inclusive(
        min: CoordScalar,
    ) -> Self {
        Self(
            Bound::Included(min),
            Bound::Unbounded,
        )
    }

    pub(crate) fn from_max_inclusive(
        max: CoordScalar,
    ) -> Self {
        Self(
            Bound::Unbounded,
            Bound::Included(max),
        )
    }

    /// Build from a Polars comparison operator + scalar.
    pub(crate) fn from_polars_op(
        op: Operator,
        scalar: CoordScalar,
    ) -> BackendResult<Self> {
        match op {
            Operator::Eq => Ok(
                Self::from_equal_case(scalar),
            ),
            Operator::Gt => Ok(
                Self::from_min_exclusive(scalar),
            ),
            Operator::GtEq => Ok(
                Self::from_min_inclusive(scalar),
            ),
            Operator::Lt => Ok(
                Self::from_max_exclusive(scalar),
            ),
            Operator::LtEq => Ok(
                Self::from_max_inclusive(scalar),
            ),
            _ => Err(BackendError::UnsupportedOperator { op }),
        }
    }

    /// Convert to an index range for an integer-indexed dimension.
    pub(crate) fn index_range_for_index_dim(
        &self,
        dim_len_est: u64,
    ) -> Option<std::ops::Range<u64>> {
        let mut start = 0u64;
        let mut end_exclusive = dim_len_est;

        match self.start_bound() {
            Bound::Included(s)
            | Bound::Excluded(s) => {
                start =
                    s.as_i128_orderable()? as u64;
            }
            Bound::Unbounded => {}
        }
        match self.end_bound() {
            Bound::Included(s) => {
                let v =
                    s.as_i128_orderable()? as u64;
                end_exclusive = v
                    .saturating_add(1)
                    .min(dim_len_est);
            }
            Bound::Excluded(s) => {
                end_exclusive =
                    s.as_i128_orderable()? as u64;
            }
            Bound::Unbounded => {}
        }
        Some(start..end_exclusive)
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

impl HasIntersect for ValueRangePresent {
    fn intersect(
        &self,
        other_range: Option<Self>,
    ) -> Option<Self> {
        let Some(other) = other_range else {
            return Some(self.clone());
        };
        // Intersect = tighter start × tighter end.
        let new_start = pick_tighter_min_bound(
            self.0.clone(),
            other.0.clone(),
        );
        let new_end = pick_tighter_max_bound(
            self.1.clone(),
            other.1.clone(),
        );
        Self::from_option_bounds(
            new_start, new_end,
        )
    }
}

impl HasIntersect for Option<ValueRangePresent> {
    fn intersect(
        &self,
        other: Option<Self>,
    ) -> Option<Self> {
        match (self, other) {
            (Some(a), None) => {
                Some(Some(a.clone()))
            }
            (None, a) => a,
            (Some(a), Some(b)) => match b {
                Some(b) => Some(
                    a.intersect(Some(b.clone())),
                ),
                None => Some(Some(a.clone())),
            },
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

fn pick_tighter_min_bound(
    a: CoordBound,
    b: CoordBound,
) -> Option<CoordBound> {
    let (a_scalar, a_excl) = match &a {
        CoordBound::Included(s) => {
            (s.clone(), false)
        }
        CoordBound::Excluded(s) => {
            (s.clone(), true)
        }
        CoordBound::Unbounded => return Some(b),
    };
    let (b_scalar, b_excl) = match &b {
        CoordBound::Included(s) => {
            (s.clone(), false)
        }
        CoordBound::Excluded(s) => {
            (s.clone(), true)
        }
        CoordBound::Unbounded => return Some(a),
    };
    match a_scalar.partial_cmp(&b_scalar) {
        Some(std::cmp::Ordering::Less) => Some(b),
        Some(std::cmp::Ordering::Greater) => {
            Some(a)
        }
        Some(std::cmp::Ordering::Equal) => {
            let excl = a_excl || b_excl;
            Some(if excl {
                CoordBound::Excluded(a_scalar)
            } else {
                CoordBound::Included(a_scalar)
            })
        }
        None => None,
    }
}

fn pick_tighter_max_bound(
    a: CoordBound,
    b: CoordBound,
) -> Option<CoordBound> {
    let (a_scalar, a_excl) = match &a {
        CoordBound::Included(s) => {
            (s.clone(), false)
        }
        CoordBound::Excluded(s) => {
            (s.clone(), true)
        }
        CoordBound::Unbounded => return Some(b),
    };
    let (b_scalar, b_excl) = match &b {
        CoordBound::Included(s) => {
            (s.clone(), false)
        }
        CoordBound::Excluded(s) => {
            (s.clone(), true)
        }
        CoordBound::Unbounded => return Some(a),
    };
    match a_scalar.partial_cmp(&b_scalar) {
        Some(std::cmp::Ordering::Less) => Some(a),
        Some(std::cmp::Ordering::Greater) => {
            Some(b)
        }
        Some(std::cmp::Ordering::Equal) => {
            let excl = a_excl || b_excl;
            Some(if excl {
                CoordBound::Excluded(a_scalar)
            } else {
                CoordBound::Included(a_scalar)
            })
        }
        None => None,
    }
}

/// Compute (start, end) bounds from a ValueRangePresent using provided lower/upper bound functions.
pub(crate) fn compute_bounds_from_value_range<
    FL,
    FU,
>(
    vr: &ValueRangePresent,
    n: u64,
    lower: FL,
    upper: FU,
) -> Option<(u64, u64)>
where
    FL: Fn(&CoordBound) -> Option<u64>,
    FU: Fn(&CoordBound) -> Option<u64>,
{
    let mut start = 0u64;
    let mut end = n;
    if !matches!(vr.0, Bound::Unbounded) {
        start = lower(&vr.0)?;
    }
    if !matches!(vr.1, Bound::Unbounded) {
        end = upper(&vr.1)?;
    }
    Some((start, end))
}
