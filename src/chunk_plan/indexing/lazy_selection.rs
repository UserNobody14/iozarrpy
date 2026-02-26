//! Lazy selection types that defer value-to-index resolution.
//!
//! These types mirror the concrete selection types in `selection.rs` but store
//! `ValueRange` constraints instead of resolved `std::ops::Range<u64>`s, enabling batched
//! resolution.

use std::collections::BTreeMap;

use smallvec::SmallVec;

use super::grouped_selection::ArraySelectionType;
use super::selection::Emptyable;
use super::types::ValueRangePresent;
use crate::IStr;
use std::ops::Range;

/// A per-dimension constraint in value-space (deferred resolution).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LazyDimConstraint {
    /// No constraint on this dimension (select all indices).
    All,
    /// Proven empty (short-circuit optimization).
    Empty,
    /// Needs resolution from value-space to index-space.
    Unresolved(ValueRangePresent),
    /// Needs resolution with interpolation expansion (expand to chunk boundaries, or across 2 chunks if at boundary).
    /// Used for interpolation operations that need bracketing indices.
    InterpolationRange(ValueRangePresent),
}

impl LazyDimConstraint {
    /// Returns true if this constraint is proven empty.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        matches!(self, LazyDimConstraint::Empty)
    }
}

/// Conjunction (AND) of per-dimension lazy constraints.
///
/// `None` inner map = empty (selects nothing).
/// `Some(map)` with empty map = all (no constraints).
/// Missing dimension keys mean "all indices along that dimension".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LazyHyperRectangle {
    /// `None` => empty rectangle; `Some(map)` => active constraints.
    dims:
        Option<BTreeMap<IStr, LazyDimConstraint>>,
}

impl Default for LazyHyperRectangle {
    fn default() -> Self {
        Self {
            dims: Some(BTreeMap::new()),
        }
    }
}

impl LazyHyperRectangle {
    /// Create an empty rectangle (selects nothing).
    pub(crate) fn empty() -> Self {
        Self { dims: None }
    }

    /// Create a rectangle that selects all indices.
    pub(crate) fn all() -> Self {
        Self::default()
    }

    /// Returns true if this rectangle is proven empty.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.dims.is_none()
    }

    /// Returns true if this rectangle represents "select all" (no constraints).
    #[inline]
    pub(crate) fn is_all(&self) -> bool {
        matches!(&self.dims, Some(m) if m.is_empty())
    }

    /// Iterate over all dimension constraints.
    pub(crate) fn dims(
        &self,
    ) -> impl Iterator<
        Item = (&IStr, &LazyDimConstraint),
    > {
        self.dims.iter().flat_map(|m| m.iter())
    }

    /// Add a constraint to a dimension.
    pub(crate) fn with_dim(
        mut self,
        dim: IStr,
        constraint: LazyDimConstraint,
    ) -> Self {
        if constraint.is_empty() {
            return Self::empty();
        }
        if let Some(ref mut m) = self.dims {
            if !matches!(
                constraint,
                LazyDimConstraint::All
            ) {
                m.insert(dim, constraint);
            }
        }
        self
    }

    /// Create a rectangle from a map of constraints.
    pub(crate) fn with_dims(
        constraints: BTreeMap<
            IStr,
            LazyDimConstraint,
        >,
    ) -> Self {
        if constraints
            .values()
            .any(|c| c.is_empty())
        {
            Self::empty()
        } else {
            Self {
                dims: Some(constraints),
            }
        }
    }
}

/// Disjunction (OR) of lazy hyper-rectangles, or a deferred set operation.
///
/// Uses `SmallVec` to avoid heap allocation for common cases with few rectangles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LazyArraySelection {
    /// Simple case: a disjunction of rectangles.
    Rectangles(SmallVec<[LazyHyperRectangle; 4]>),
    /// Deferred difference operation (A \ B).
    /// Stored because computing difference may require resolution of both sides.
    Difference(
        Box<LazyArraySelection>,
        Box<LazyArraySelection>,
    ),
    /// Deferred union operation (A | B).
    /// Stored when we can't easily flatten the union (e.g., union of Differences).
    Union(
        Box<LazyArraySelection>,
        Box<LazyArraySelection>,
    ),
}

impl Default for LazyArraySelection {
    fn default() -> Self {
        Self::empty()
    }
}

impl LazyArraySelection {
    /// Create an empty selection (selects nothing).
    pub(crate) fn empty() -> Self {
        Self::Rectangles(SmallVec::new())
    }

    /// Create a selection that selects all indices.
    pub(crate) fn all() -> Self {
        let mut sv = SmallVec::new();
        sv.push(LazyHyperRectangle::all());
        Self::Rectangles(sv)
    }

    /// Create a selection from a single rectangle.
    pub(crate) fn from_rectangle(
        rect: LazyHyperRectangle,
    ) -> Self {
        if rect.is_empty() {
            Self::empty()
        } else {
            let mut sv = SmallVec::new();
            sv.push(rect);
            Self::Rectangles(sv)
        }
    }

    /// Returns true if this selection is proven empty.
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Rectangles(rects) => {
                rects.is_empty()
                    || rects
                        .iter()
                        .all(|r| r.is_empty())
            }
            // For difference/union, we can't determine emptiness without resolution
            Self::Difference(_, _) => false,
            Self::Union(a, b) => {
                a.is_empty() && b.is_empty()
            }
        }
    }

    /// Returns true if this selection represents "select all" (no constraints).
    pub(crate) fn is_all(&self) -> bool {
        match self {
            Self::Rectangles(rects) => {
                rects.len() == 1
                    && rects[0].is_all()
            }
            Self::Difference(_, _)
            | Self::Union(_, _) => false,
        }
    }
}

// ============================================================================
// SetOperations implementations for lazy types
// ============================================================================

use super::selection::SetOperations;

impl Emptyable for Range<u64> {
    fn empty() -> Self {
        0..0
    }
    fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}
impl SetOperations for Range<u64> {
    fn union(&self, other: &Self) -> Self {
        self.start.min(other.start)
            ..self.end.max(other.end)
    }
    fn intersect(&self, other: &Self) -> Self {
        self.start.max(other.start)
            ..self.end.min(other.end)
    }
    fn difference(&self, other: &Self) -> Self {
        self.start.max(other.start)
            ..self.end.min(other.end)
    }
    fn exclusive_or(&self, other: &Self) -> Self {
        self.start.max(other.start)
            ..self.end.min(other.end)
    }
}

impl Emptyable for LazyDimConstraint {
    fn empty() -> Self {
        LazyDimConstraint::Empty
    }
    fn is_empty(&self) -> bool {
        LazyDimConstraint::is_empty(self)
    }
}
impl SetOperations for LazyDimConstraint {
    fn union(&self, other: &Self) -> Self {
        // Union of constraints: the less restrictive one wins
        match (self, other) {
            (LazyDimConstraint::Empty, x)
            | (x, LazyDimConstraint::Empty) => {
                x.clone()
            }
            (LazyDimConstraint::All, _)
            | (_, LazyDimConstraint::All) => {
                LazyDimConstraint::All
            }
            // Can't merge unresolved constraints without resolution - return All conservatively
            _ => LazyDimConstraint::All,
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        match (self, other) {
            (LazyDimConstraint::Empty, _) | (_, LazyDimConstraint::Empty) => {
                LazyDimConstraint::Empty
            }
            (LazyDimConstraint::All, x) | (x, LazyDimConstraint::All) => x.clone(),

            (LazyDimConstraint::Unresolved(a), LazyDimConstraint::Unresolved(b)) => {
                match a.intersect(b) {
                    Some(vr) => LazyDimConstraint::Unresolved(vr),
                    None => LazyDimConstraint::Empty,
                }
            }
            (LazyDimConstraint::InterpolationRange(a), LazyDimConstraint::InterpolationRange(b))
            | (LazyDimConstraint::InterpolationRange(a), LazyDimConstraint::Unresolved(b))
            | (LazyDimConstraint::Unresolved(b), LazyDimConstraint::InterpolationRange(a)) => {
                match a.intersect(b) {
                    Some(vr) => LazyDimConstraint::InterpolationRange(vr),
                    None => LazyDimConstraint::Empty,
                }
            }
        }
    }

    fn difference(&self, other: &Self) -> Self {
        match (self, other) {
            (LazyDimConstraint::Empty, _) => {
                LazyDimConstraint::Empty
            }
            (x, LazyDimConstraint::Empty) => {
                x.clone()
            }
            (_, LazyDimConstraint::All) => {
                LazyDimConstraint::Empty
            }
            (LazyDimConstraint::All, _) => {
                LazyDimConstraint::All
            } // Can't compute A \ B without knowing B
            // Can't compute difference with unresolved - return self conservatively
            _ => self.clone(),
        }
    }

    fn exclusive_or(&self, other: &Self) -> Self {
        self.difference(other)
            .union(&other.difference(self))
    }
}

impl Emptyable for LazyArraySelection {
    fn empty() -> Self {
        LazyArraySelection::empty()
    }
    fn is_empty(&self) -> bool {
        LazyArraySelection::is_empty(self)
    }
}
impl SetOperations for LazyArraySelection {
    fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        match (self, other) {
            (
                LazyArraySelection::Rectangles(a),
                LazyArraySelection::Rectangles(b),
            ) => {
                let mut combined = a.clone();
                combined
                    .extend(b.iter().cloned());
                LazyArraySelection::Rectangles(
                    combined,
                )
            }
            // For complex cases with Difference/Union, store as deferred Union
            _ => LazyArraySelection::Union(
                Box::new(self.clone()),
                Box::new(other.clone()),
            ),
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return LazyArraySelection::empty();
        }

        match (self, other) {
            (
                LazyArraySelection::Rectangles(a),
                LazyArraySelection::Rectangles(b),
            ) => {
                let mut result = SmallVec::new();
                for rect_a in a.iter() {
                    for rect_b in b.iter() {
                        if rect_a.is_empty()
                            || rect_b.is_empty()
                        {
                            continue;
                        }
                        // Intersect the two rectangles
                        let intersected =
                            intersect_lazy_rectangles(
                                rect_a, rect_b,
                            );
                        if !intersected.is_empty()
                        {
                            result.push(
                                intersected,
                            );
                        }
                    }
                }
                if result.is_empty() {
                    LazyArraySelection::empty()
                } else {
                    LazyArraySelection::Rectangles(
                        result,
                    )
                }
            }
            // For Difference types, defer intersection
            _ => {
                // Conservative: can't easily intersect with Difference
                self.clone()
            }
        }
    }

    fn difference(&self, other: &Self) -> Self {
        if self.is_empty() {
            return LazyArraySelection::empty();
        }
        if other.is_empty() {
            return self.clone();
        }

        // Special case: all - all = empty
        // This is important for selector XOR operations where both sides select "all"
        if self.is_all() && other.is_all() {
            return LazyArraySelection::empty();
        }

        // Store as deferred difference for materialization
        LazyArraySelection::Difference(
            Box::new(self.clone()),
            Box::new(other.clone()),
        )
    }

    fn exclusive_or(&self, other: &Self) -> Self {
        self.difference(other)
            .union(&other.difference(self))
    }
}

impl ArraySelectionType for LazyArraySelection {}

/// Intersect two lazy hyper-rectangles.
fn intersect_lazy_rectangles(
    a: &LazyHyperRectangle,
    b: &LazyHyperRectangle,
) -> LazyHyperRectangle {
    let (Some(a_dims), Some(b_dims)) =
        (&a.dims, &b.dims)
    else {
        return LazyHyperRectangle::empty();
    };

    let all_dim_keys: std::collections::BTreeSet<
        &IStr,
    > = a_dims
        .keys()
        .chain(b_dims.keys())
        .collect();

    let mut merged = BTreeMap::new();
    for dim in all_dim_keys {
        let ca = a_dims
            .get(dim)
            .cloned()
            .unwrap_or(LazyDimConstraint::All);
        let cb = b_dims
            .get(dim)
            .cloned()
            .unwrap_or(LazyDimConstraint::All);
        let intersected = ca.intersect(&cb);

        if intersected.is_empty() {
            return LazyHyperRectangle::empty();
        }
        if !matches!(
            intersected,
            LazyDimConstraint::All
        ) {
            merged
                .insert(dim.clone(), intersected);
        }
    }

    LazyHyperRectangle { dims: Some(merged) }
}
