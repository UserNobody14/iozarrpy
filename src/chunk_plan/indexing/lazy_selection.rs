//! Lazy selection types that defer value-to-index resolution.
//!
//! These types mirror the concrete selection types in `selection.rs` but store
//! `ValueRange` constraints instead of resolved `IndexRange`s, enabling batched
//! resolution.

use std::collections::BTreeMap;
use std::sync::Arc;

use smallvec::SmallVec;

use super::selection::{ArraySubsetList, Emptyable};
use super::types::ValueRange;
use std::ops::Range;
use crate::{IStr, IntoIStr};

/// A per-dimension constraint in value-space (deferred resolution).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LazyDimConstraint {
    /// No constraint on this dimension (select all indices).
    All,
    /// Proven empty (short-circuit optimization).
    Empty,
    /// Needs resolution from value-space to index-space.
    Unresolved(ValueRange),
    /// Needs resolution with interpolation expansion (expand by 1 on each side).
    /// Used for interpolation operations that need bracketing indices.
    UnresolvedInterpolation(Arc<ValueRange>),
    /// Interpolation with multiple target points - each needs bracketing indices.
    /// The resolution will union the bracketing ranges for each point with clamping.
    UnresolvedInterpolationPoints(Arc<Vec<super::types::CoordScalar>>),
    /// Already resolved (optimization for pre-computed constraints).
    Resolved(Range<u64>),
}

impl LazyDimConstraint {
    /// Returns true if this constraint is proven empty.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        matches!(self, LazyDimConstraint::Empty)
            || matches!(self, LazyDimConstraint::Resolved(r) if r.is_empty())
            || matches!(self, LazyDimConstraint::Unresolved(vr) if vr.empty)
            || matches!(self, LazyDimConstraint::UnresolvedInterpolation(vr) if vr.empty)
            || matches!(self, LazyDimConstraint::UnresolvedInterpolationPoints(pts) if pts.is_empty())
    }

}

/// Conjunction (AND) of per-dimension lazy constraints.
///
/// Missing dimension keys mean "all indices along that dimension".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LazyHyperRectangle {
    dims: BTreeMap<IStr, LazyDimConstraint>,
    empty: bool,
}

impl LazyHyperRectangle {
    /// Create an empty rectangle (selects nothing).
    pub(crate) fn empty() -> Self {
        Self {
            dims: BTreeMap::new(),
            empty: true,
        }
    }

    /// Create a rectangle that selects all indices.
    pub(crate) fn all() -> Self {
        Self {
            dims: BTreeMap::new(),
            empty: false,
        }
    }

    /// Returns true if this rectangle is proven empty.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.empty || self.dims.values().any(|c| c.is_empty())
    }

    /// Get constraint for a specific dimension.
    pub(crate) fn get_dim(&self, dim: &str) -> Option<&LazyDimConstraint> {
        self.dims.get(&dim.istr())
    }

    /// Iterate over all dimension constraints.
    pub(crate) fn dims(&self) -> impl Iterator<Item = (&IStr, &LazyDimConstraint)> {
        self.dims.iter()
    }

    /// Add a constraint to a dimension.
    pub(crate) fn with_dim(mut self, dim: IStr, constraint: LazyDimConstraint) -> Self {
        if constraint.is_empty() {
            self.empty = true;
            self.dims.clear();
            return self;
        }
        if !self.empty && !matches!(constraint, LazyDimConstraint::All) {
            self.dims.insert(dim, constraint);
        }
        self
    }

    /// Add a constraint from a string dimension name.
    pub(crate) fn with_dim_str(self, dim: &str, constraint: LazyDimConstraint) -> Self {
        self.with_dim(dim.istr(), constraint)
    }

    /// Create a rectangle from a map of constraints.
    pub(crate) fn with_dims(constraints: BTreeMap<IStr, LazyDimConstraint>) -> Self {
        let empty = constraints.values().any(|c| c.is_empty());
        Self { dims: constraints, empty }
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
    Difference(Box<LazyArraySelection>, Box<LazyArraySelection>),
    /// Deferred union operation (A | B).
    /// Stored when we can't easily flatten the union (e.g., union of Differences).
    Union(Box<LazyArraySelection>, Box<LazyArraySelection>),
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
    pub(crate) fn from_rectangle(rect: LazyHyperRectangle) -> Self {
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
            Self::Rectangles(rects) => rects.is_empty() || rects.iter().all(|r| r.is_empty()),
            // For difference/union, we can't determine emptiness without resolution
            Self::Difference(_, _) => false,
            Self::Union(a, b) => a.is_empty() && b.is_empty(),
        }
    }

}

/// Dataset-level lazy selection: variable name -> selection on that variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LazyDatasetSelection {
    /// No selection was made (equivalent to "select all").
    NoSelectionMade,
    /// Everything has been excluded.
    Empty,
    /// Standard selection mapping variables to their selections.
    Selection(BTreeMap<IStr, LazyArraySelection>),
}

impl Default for LazyDatasetSelection {
    fn default() -> Self {
        Self::NoSelectionMade
    }
}

impl LazyDatasetSelection {
    /// Create an empty selection (selects nothing).
    pub(crate) fn empty() -> Self {
        Self::Empty
    }

    /// Returns true if this is an empty selection.
    pub(crate) fn is_empty_selection(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Iterate over variables and their selections.
    pub(crate) fn vars(&self) -> impl Iterator<Item = (&str, &LazyArraySelection)> {
        match self {
            Self::Selection(sel) => Box::new(sel.iter().map(|(k, v)| (k.as_ref(), v)))
                as Box<dyn Iterator<Item = _>>,
            Self::NoSelectionMade | Self::Empty => Box::new(std::iter::empty()),
        }
    }

}

/// Create a lazy dataset selection for the given variables with all indices selected.
pub(crate) fn lazy_dataset_all_for_vars(vars: impl IntoIterator<Item = IStr>) -> LazyDatasetSelection {
    let mut m = BTreeMap::new();
    for v in vars {
        m.insert(v, LazyArraySelection::all());
    }
    LazyDatasetSelection::Selection(m)
}

/// Create a lazy dataset selection for the given variables with the specified selection.
pub(crate) fn lazy_dataset_for_vars_with_selection(
    vars: impl IntoIterator<Item = IStr>,
    sel: LazyArraySelection,
) -> LazyDatasetSelection {
    let mut m = BTreeMap::new();
    for v in vars {
        m.insert(v, sel.clone());
    }
    LazyDatasetSelection::Selection(m)
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
        self.start.min(other.start)..self.end.max(other.end)
    }
    fn intersect(&self, other: &Self) -> Self {
        self.start.max(other.start)..self.end.min(other.end)
    }
    fn difference(&self, other: &Self) -> Self {
        self.start.max(other.start)..self.end.min(other.end)
    }
    fn exclusive_or(&self, other: &Self) -> Self {
        self.start.max(other.start)..self.end.min(other.end)
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
            (LazyDimConstraint::Empty, x) | (x, LazyDimConstraint::Empty) => x.clone(),
            (LazyDimConstraint::All, _) | (_, LazyDimConstraint::All) => LazyDimConstraint::All,
            (LazyDimConstraint::Resolved(a), LazyDimConstraint::Resolved(b)) => {
                LazyDimConstraint::Resolved(a.union(b))
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
            (LazyDimConstraint::Resolved(a), LazyDimConstraint::Resolved(b)) => {
                let result = a.intersect(b);
                if result.is_empty() {
                    LazyDimConstraint::Empty
                } else {
                    LazyDimConstraint::Resolved(result)
                }
            }
            (LazyDimConstraint::Unresolved(a), LazyDimConstraint::Unresolved(b)) => {
                let vr = a.intersect(b);
                if vr.empty {
                    LazyDimConstraint::Empty
                } else {
                    LazyDimConstraint::Unresolved(vr)
                }
            }
            // UnresolvedInterpolation combined with Unresolved: intersect their value ranges
            (
                LazyDimConstraint::UnresolvedInterpolation(a),
                LazyDimConstraint::UnresolvedInterpolation(b),
            ) => {
                let vr = a.intersect(b);
                if vr.empty {
                    LazyDimConstraint::Empty
                } else {
                    LazyDimConstraint::UnresolvedInterpolation(Arc::new(vr))
                }
            }
            (LazyDimConstraint::UnresolvedInterpolation(a), LazyDimConstraint::Unresolved(b))
            | (LazyDimConstraint::Unresolved(b), LazyDimConstraint::UnresolvedInterpolation(a)) => {
                let vr = a.intersect(b);
                if vr.empty {
                    LazyDimConstraint::Empty
                } else {
                    // Keep as interpolation to preserve the expansion behavior
                    LazyDimConstraint::UnresolvedInterpolation(Arc::new(vr))
                }
            }
            // Mixed resolved/unresolved - can't intersect without resolution
            // Keep the unresolved one (conservative: may over-select)
            (LazyDimConstraint::Unresolved(vr), _) | (_, LazyDimConstraint::Unresolved(vr)) => {
                LazyDimConstraint::Unresolved(vr.clone())
            }
            (LazyDimConstraint::UnresolvedInterpolation(vr), _)
            | (_, LazyDimConstraint::UnresolvedInterpolation(vr)) => {
                LazyDimConstraint::UnresolvedInterpolation(vr.clone())
            }
            // InterpolationPoints can't be easily intersected - keep the first one
            (LazyDimConstraint::UnresolvedInterpolationPoints(pts), _)
            | (_, LazyDimConstraint::UnresolvedInterpolationPoints(pts)) => {
                LazyDimConstraint::UnresolvedInterpolationPoints(pts.clone())
            }
        }
    }

    fn difference(&self, other: &Self) -> Self {
        match (self, other) {
            (LazyDimConstraint::Empty, _) => LazyDimConstraint::Empty,
            (x, LazyDimConstraint::Empty) => x.clone(),
            (_, LazyDimConstraint::All) => LazyDimConstraint::Empty,
            (LazyDimConstraint::All, _) => LazyDimConstraint::All, // Can't compute A \ B without knowing B
            (LazyDimConstraint::Resolved(a), LazyDimConstraint::Resolved(b)) => {
                let result = a.difference(b);
                if result.is_empty() {
                    LazyDimConstraint::Empty
                } else {
                    LazyDimConstraint::Resolved(result)
                }
            }
            // Can't compute difference with unresolved - return self conservatively
            _ => self.clone(),
        }
    }

    fn exclusive_or(&self, other: &Self) -> Self {
        self.difference(other).union(&other.difference(self))
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
            (LazyArraySelection::Rectangles(a), LazyArraySelection::Rectangles(b)) => {
                let mut combined = a.clone();
                combined.extend(b.iter().cloned());
                LazyArraySelection::Rectangles(combined)
            }
            // For complex cases with Difference/Union, store as deferred Union
            _ => LazyArraySelection::Union(Box::new(self.clone()), Box::new(other.clone())),
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return LazyArraySelection::empty();
        }

        match (self, other) {
            (LazyArraySelection::Rectangles(a), LazyArraySelection::Rectangles(b)) => {
                let mut result = SmallVec::new();
                for rect_a in a.iter() {
                    for rect_b in b.iter() {
                        if rect_a.is_empty() || rect_b.is_empty() {
                            continue;
                        }
                        // Intersect the two rectangles
                        let intersected = intersect_lazy_rectangles(rect_a, rect_b);
                        if !intersected.is_empty() {
                            result.push(intersected);
                        }
                    }
                }
                if result.is_empty() {
                    LazyArraySelection::empty()
                } else {
                    LazyArraySelection::Rectangles(result)
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

        // Store as deferred difference for materialization
        LazyArraySelection::Difference(Box::new(self.clone()), Box::new(other.clone()))
    }

    fn exclusive_or(&self, other: &Self) -> Self {
        self.difference(other).union(&other.difference(self))
    }

}

impl Emptyable for LazyDatasetSelection {
    fn empty() -> Self {
        LazyDatasetSelection::empty()
    }
    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}
impl SetOperations for LazyDatasetSelection {
    fn union(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::NoSelectionMade, _) | (_, Self::NoSelectionMade) => Self::NoSelectionMade,
            (Self::Empty, x) | (x, Self::Empty) => x.clone(),
            (Self::Selection(a), Self::Selection(b)) => {
                let mut out = a.clone();
                for (k, v) in b {
                    out.entry(k.clone())
                        .and_modify(|cur| *cur = cur.union(v))
                        .or_insert_with(|| v.clone());
                }
                Self::Selection(out)
            }
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::NoSelectionMade, x) | (x, Self::NoSelectionMade) => x.clone(),
            (Self::Empty, _) | (_, Self::Empty) => Self::Empty,
            (Self::Selection(a), Self::Selection(b)) => {
                let mut out = BTreeMap::new();
                for (k, sel_a) in a {
                    if let Some(sel_b) = b.get(k) {
                        let intersected = sel_a.intersect(sel_b);
                        if !intersected.is_empty() {
                            out.insert(k.clone(), intersected);
                        }
                    }
                }
                if out.is_empty() {
                    Self::Empty
                } else {
                    Self::Selection(out)
                }
            }
        }
    }

    fn difference(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::NoSelectionMade, _) => Self::NoSelectionMade,
            (_, Self::NoSelectionMade) => Self::Empty,
            (Self::Empty, _) => Self::Empty,
            (x, Self::Empty) => x.clone(),
            (Self::Selection(a), Self::Selection(b)) => {
                let mut out = BTreeMap::new();
                for (k, sel_a) in a {
                    if let Some(sel_b) = b.get(k) {
                        let diff = sel_a.difference(sel_b);
                        if !diff.is_empty() {
                            out.insert(k.clone(), diff);
                        }
                    } else {
                        out.insert(k.clone(), sel_a.clone());
                    }
                }
                if out.is_empty() {
                    Self::Empty
                } else {
                    Self::Selection(out)
                }
            }
        }
    }

    fn exclusive_or(&self, other: &Self) -> Self {
        self.difference(other).union(&other.difference(self))
    }

}

/// Intersect two lazy hyper-rectangles.
fn intersect_lazy_rectangles(a: &LazyHyperRectangle, b: &LazyHyperRectangle) -> LazyHyperRectangle {
    if a.is_empty() || b.is_empty() {
        return LazyHyperRectangle::empty();
    }

    let mut result = LazyHyperRectangle::all();

    // Collect all dimension names from both rectangles
    let all_dims: std::collections::BTreeSet<&IStr> =
        a.dims.keys().chain(b.dims.keys()).collect();

    for dim in all_dims {
        let constraint_a = a.dims.get(dim).cloned().unwrap_or(LazyDimConstraint::All);
        let constraint_b = b.dims.get(dim).cloned().unwrap_or(LazyDimConstraint::All);
        let intersected = constraint_a.intersect(&constraint_b);

        if intersected.is_empty() {
            return LazyHyperRectangle::empty();
        }

        if !matches!(intersected, LazyDimConstraint::All) {
            result.dims.insert(dim.clone(), intersected);
        }
    }

    result
}
