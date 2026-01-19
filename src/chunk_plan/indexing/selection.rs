use std::collections::{BTreeMap, BTreeSet};

use super::types::BoundKind;

/// Dataset-level selection: variable name -> selection on that variable.
///
/// This is intended to be a pure index-selection representation (no chunking info).
#[derive(Debug, Clone, PartialEq, Eq)]
// pub(crate) struct DatasetSelection(pub(crate) BTreeMap<String, DataArraySelection>);
pub (crate) enum DatasetSelection {
    /// If no selection was made:
    NoSelectionMade,
    /// If everything has been excluded
    Empty,
    /// Standard selection:
    Selection(BTreeMap<String, DataArraySelection>),
}

impl DatasetSelection {
    pub(crate) fn vars(&self) -> impl Iterator<Item = (&str, &DataArraySelection)> {
        match self {
            Self::Selection(selection) => Box::new(selection.iter().map(|(k, v)| (k.as_str(), v)).collect::<Vec<_>>().into_iter()),
            Self::NoSelectionMade | Self::Empty => Box::new(Vec::new().into_iter()),
        }
    }

    pub(crate) fn contains_dim(&self, dim: &str) -> bool {
        match self {
            Self::Selection(selection) => selection.contains_key(dim),
            Self::NoSelectionMade | Self::Empty => false,
        }
    }

    pub(crate) fn insert_dim(&mut self, dim: String, da: DataArraySelection) {
        match self {
            Self::Selection(selection) => {
                let _ = selection.insert(dim, da);
            }
            Self::NoSelectionMade | Self::Empty => {
                *self = Self::Selection(BTreeMap::new());
                self.insert_dim(dim, da);
            }
        }
    }

}

/// Selection for a single array, expressed as a disjunction (OR) of hyper-rectangles.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DataArraySelection(pub(crate) Vec<HyperRectangleSelection>);

impl DataArraySelection {
    pub(crate) fn empty() -> Self {
        Self(Vec::new())
    }

    /// Select all indices for this array.
    ///
    /// Represented as a single hyper-rectangle with no constrained dimensions.
    pub(crate) fn all() -> Self {
        Self(vec![HyperRectangleSelection::all()])
    }

}
/// Conjunction (AND) of per-dimension index constraints.
///
/// Missing dimension keys mean “all indices along that dimension”.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct HyperRectangleSelection {
    dims: BTreeMap<String, RangeList>,
    empty: bool,
}

impl HyperRectangleSelection {
    pub(crate) fn empty() -> Self {
        Self {
            dims: BTreeMap::new(),
            empty: true,
        }
    }

    pub(crate) fn all() -> Self {
        Self {
            dims: BTreeMap::new(),
            empty: false,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.empty || self.dims.values().any(|r| r.is_empty())
    }

    pub(crate) fn get_dim(&self, dim: &str) -> Option<&RangeList> {
        self.dims.get(dim)
    }

    pub(crate) fn dims(&self) -> impl Iterator<Item = (&str, &RangeList)> {
        self.dims.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub(crate) fn dim_names_union(&self, other: &HyperRectangleSelection) -> BTreeSet<String> {
        self.dims
            .keys()
            .chain(other.dims.keys())
            .cloned()
            .collect::<BTreeSet<_>>()
    }

    pub(crate) fn with_dim(mut self, dim: impl Into<String>, ranges: RangeList) -> Self {
        if ranges.is_empty() {
            self.empty = true;
            self.dims.clear();
            return self;
        }
        if !self.empty && ranges != RangeList::all() {
            self.dims.insert(dim.into(), ranges);
        }
        self
    }

}

/// A scalar range using inclusive/exclusive endpoints.
///
/// Internally we normalize these ranges into half-open `[start, end_exclusive)` ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScalarRange {
    pub(crate) start: Option<(u64, BoundKind)>,
    pub(crate) end: Option<(u64, BoundKind)>,
}


/// A list of half-open index ranges.
///
/// Invariants (after `normalize()`):
/// - sorted by `start`
/// - non-overlapping
/// - non-adjacent (coalesced)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RangeList {
    ranges: Vec<super::types::IndexRange>,
}

impl RangeList {
    pub(crate) fn empty() -> Self {
        Self { ranges: Vec::new() }
    }

    pub(crate) fn all() -> Self {
        // Represent “all indices” as a single unbounded-ish range. Planning later clamps to
        // the actual dimension length.
        Self {
            ranges: vec![super::types::IndexRange {
                start: 0,
                end_exclusive: u64::MAX,
            }],
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub(crate) fn ranges(&self) -> &[super::types::IndexRange] {
        &self.ranges
    }

    pub(crate) fn from_index_range(r: super::types::IndexRange) -> Self {
        let mut out = Self { ranges: vec![r] };
        out.normalize();
        out
    }

    pub(crate) fn normalize(&mut self) {
        self.ranges.retain(|r| !r.is_empty());
        self.ranges.sort_by_key(|r| r.start);

        let mut merged: Vec<super::types::IndexRange> = Vec::with_capacity(self.ranges.len());
        for r in self.ranges.drain(..) {
            if let Some(last) = merged.last_mut() {
                if r.start <= last.end_exclusive {
                    last.end_exclusive = last.end_exclusive.max(r.end_exclusive);
                    continue;
                }
            }
            merged.push(r);
        }
        self.ranges = merged;
    }

    pub(crate) fn clamp_to_len(&self, len: u64) -> Self {
        let mut out = Self::empty();
        for r in &self.ranges {
            let s = r.start.min(len);
            let e = r.end_exclusive.min(len);
            if e > s {
                out.ranges.push(super::types::IndexRange {
                    start: s,
                    end_exclusive: e,
                });
            }
        }
        out.normalize();
        out
    }
}

/// Operations for sets of selections.
pub trait SetOperations {
    fn union(&self, other: &Self) -> Self;
    fn intersect(&self, other: &Self) -> Self;
    fn difference(&self, other: &Self) -> Self;
    fn exclusive_or(&self, other: &Self) -> Self;
    fn is_empty(&self) -> bool;
}

impl SetOperations for RangeList {
    fn union(&self, other: &RangeList) -> RangeList {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        let mut out = RangeList {
            ranges: self
                .ranges
                .iter()
                .chain(other.ranges.iter())
                .copied()
                .collect(),
        };
        out.normalize();
        out
    }

    fn intersect(&self, other: &RangeList) -> RangeList {
        let mut out = RangeList::empty();
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.ranges.len() && j < other.ranges.len() {
            let a = self.ranges[i];
            let b = other.ranges[j];
            let s = a.start.max(b.start);
            let e = a.end_exclusive.min(b.end_exclusive);
            if e > s {
                out.ranges.push(super::types::IndexRange {
                    start: s,
                    end_exclusive: e,
                });
            }
            if a.end_exclusive < b.end_exclusive {
                i += 1;
            } else {
                j += 1;
            }
        }
        out.normalize();
        out
    }

    fn difference(&self, other: &RangeList) -> RangeList {
        if self.is_empty() || other.is_empty() {
            return self.clone();
        }
        let mut out = RangeList::empty();
        let mut j = 0usize;
        for &a in &self.ranges {
            let mut cur_start = a.start;
            let cur_end = a.end_exclusive;

            while j < other.ranges.len() && other.ranges[j].end_exclusive <= cur_start {
                j += 1;
            }

            let mut k = j;
            while k < other.ranges.len() {
                let b = other.ranges[k];
                if b.start >= cur_end {
                    break;
                }
                // Overlap: emit left piece if any.
                if b.start > cur_start {
                    out.ranges.push(super::types::IndexRange {
                        start: cur_start,
                        end_exclusive: b.start.min(cur_end),
                    });
                }
                cur_start = cur_start.max(b.end_exclusive);
                if cur_start >= cur_end {
                    break;
                }
                k += 1;
            }
            if cur_start < cur_end {
                out.ranges.push(super::types::IndexRange {
                    start: cur_start,
                    end_exclusive: cur_end,
                });
            }
        }
        out.normalize();
        out
    }

    fn exclusive_or(&self, other: &RangeList) -> RangeList {
        self.difference(other).union(&other.difference(self))
    }

    fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

impl SetOperations for DatasetSelection {
    fn union(&self, other: &DatasetSelection) -> DatasetSelection {
        match (self, other) {
            (Self::Selection(s), Self::Selection(o)) => {
                if s.is_empty() {
                    return other.clone();
                }
                if o.is_empty() {
                    return self.clone();
                }
                let mut out = s.clone();
                for (k, v) in o {
                    out.entry(k.clone())
                        .and_modify(|cur| *cur = cur.union(v))
                        .or_insert_with(|| v.clone());
                }
                DatasetSelection::Selection(out)
            },
            // NoSelectionMade means "all chunks" - union with all is all
            (Self::NoSelectionMade, _) | (_, Self::NoSelectionMade) => Self::NoSelectionMade,
            // Empty means "no chunks" - union with empty is the other side
            (Self::Empty, _) => other.clone(),
            (_, Self::Empty) => self.clone(),
        }
    }

    fn intersect(&self, other: &DatasetSelection) -> DatasetSelection {
        match (self, other) {
            (Self::Selection(s), Self::Selection(o)) => {
                let mut out = BTreeMap::new();
                for (k, a) in s {
                    if let Some(b) = o.get(k) {
                        let sel = a.intersect(b);
                        if !sel.is_empty() {
                            out.insert(k.clone(), sel);
                        }
                    }
                }
                DatasetSelection::Selection(out)
            }
            (Self::NoSelectionMade, _) => other.clone(),
            (_, Self::NoSelectionMade) => self.clone(),
            (Self::Empty, _) => self.clone(),
            (_, Self::Empty) => other.clone(),
        }
    }

    fn difference(&self, other: &DatasetSelection) -> DatasetSelection {
        match (self, other) {
            (Self::Selection(s), Self::Selection(o)) => {
                let mut out = BTreeMap::new();
                for (k, a) in s {
                    if let Some(b) = o.get(k) {
                        let sel = a.difference(b);
                        if !sel.is_empty() {
                            out.insert(k.clone(), sel);
                        }
                    } else {
                        // Key only in self: keep it (we're subtracting nothing)
                        out.insert(k.clone(), a.clone());
                    }
                }
                DatasetSelection::Selection(out)
            }
            // NoSelectionMade - X = can't represent complement, stay conservative
            (Self::NoSelectionMade, _) => Self::NoSelectionMade,
            // X - NoSelectionMade = empty (subtracting everything)
            (_, Self::NoSelectionMade) => Self::Empty,
            // Empty - X = Empty
            (Self::Empty, _) => Self::Empty,
            // X - Empty = X
            (_, Self::Empty) => self.clone(),
        }
    }

    fn exclusive_or(&self, other: &DatasetSelection) -> DatasetSelection {
        match (self, other) {
            (Self::Selection(s), Self::Selection(o)) => {
                let mut out = BTreeMap::new();
                // Keys in self
                for (k, a) in s {
                    if let Some(b) = o.get(k) {
                        let sel = a.exclusive_or(b);
                        if !sel.is_empty() {
                            out.insert(k.clone(), sel);
                        }
                    } else {
                        // Key only in self: include it
                        out.insert(k.clone(), a.clone());
                    }
                }
                // Keys only in other
                for (k, b) in o {
                    if !s.contains_key(k) {
                        out.insert(k.clone(), b.clone());
                    }
                }
                DatasetSelection::Selection(out)
            }
            // NoSelectionMade XOR X = can't represent complement, stay conservative
            (Self::NoSelectionMade, _) | (_, Self::NoSelectionMade) => Self::NoSelectionMade,
            // Empty XOR X = X
            (Self::Empty, _) => other.clone(),
            (_, Self::Empty) => self.clone(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Selection(s) => s.is_empty(),
            Self::NoSelectionMade | Self::Empty => true,
        }
    }
}

impl SetOperations for DataArraySelection {

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn union(&self, other: &DataArraySelection) -> DataArraySelection {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        let mut out = self.0.clone();
        out.extend(other.0.iter().cloned());
        DataArraySelection(out)
    }

    fn intersect(&self, other: &DataArraySelection) -> DataArraySelection {
        if self.is_empty() || other.is_empty() {
            return DataArraySelection::empty();
        }
        let mut out: Vec<HyperRectangleSelection> = Vec::new();
        for a in &self.0 {
            for b in &other.0 {
                if a.is_empty() || b.is_empty() {
                    continue;
                }
        
                let mut rect_dims = BTreeMap::new();
                for dim in a.dim_names_union(b) {
                    let a = a.dims.get(&dim).cloned().unwrap_or_else(RangeList::all);
                    let b = b.dims.get(&dim).cloned().unwrap_or_else(RangeList::all);
                    let r = a.intersect(&b);
                    if r.is_empty() {
                        continue;
                    }
                    if r != RangeList::all() {
                        rect_dims.insert(dim, r);
                    }
                }
                out.push(HyperRectangleSelection {
                    dims: rect_dims,
                    empty: false,
                });
            }
        }
        DataArraySelection(out)
    }

    fn difference(&self, other: &DataArraySelection) -> DataArraySelection {
        if self.is_empty() {
            return DataArraySelection::empty();
        }
        if other.is_empty() {
            return self.clone();
        }

        let mut cur = self.clone();
        for b in &other.0 {
            let mut next: Vec<HyperRectangleSelection> = Vec::new();
            for a in &cur.0 {
                if a.is_empty() {
                    continue;
                }
                if b.is_empty() {
                    // b is empty, so a - b = a
                    next.push(a.clone());
                    continue;
                }
                if b.dims.is_empty() && !b.empty {
                    // subtracting the universal rectangle => empty
                    continue;
                }

                // Compute hyper-rectangle difference A \ B.
                // Result is a union of "slabs" where we escape B in each dimension.
                // For each dimension in B, we create a slab that is:
                // - Outside B in that dimension (A_dim \ B_dim)
                // - Intersected with B in all prior dimensions (to avoid double-counting)
                // - Keeps all of A's other constraints
                let b_dims: Vec<String> = b.dims.keys().cloned().collect();
                let mut prefix_constraint: BTreeMap<String, RangeList> = BTreeMap::new();

                for dim in b_dims.iter() {
                    let a_dim = a.dims.get(dim).cloned().unwrap_or_else(RangeList::all);
                    let b_dim = b.dims.get(dim).cloned().unwrap_or_else(RangeList::all);
                    let diff = a_dim.difference(&b_dim);

                    if !diff.is_empty() {
                        // Create a slab: start with A's constraints
                        let mut rect_dims = a.dims.clone();
                        // Add prefix constraints (intersection with B in prior dims)
                        for (prefix_dim, prefix_range) in &prefix_constraint {
                            let a_prefix = rect_dims.get(prefix_dim).cloned().unwrap_or_else(RangeList::all);
                            let intersected = a_prefix.intersect(prefix_range);
                            if intersected.is_empty() {
                                // This slab is empty, skip it
                                continue;
                            }
                            if intersected != RangeList::all() {
                                rect_dims.insert(prefix_dim.clone(), intersected);
                            }
                        }
                        // Set this dimension to the difference (escape B here)
                        if diff != RangeList::all() {
                            rect_dims.insert(dim.clone(), diff);
                        } else {
                            rect_dims.remove(dim);
                        }
                        next.push(HyperRectangleSelection {
                            dims: rect_dims,
                            empty: false,
                        });
                    }

                    // Update prefix constraint: intersect A with B in this dimension
                    let a_dim = a.dims.get(dim).cloned().unwrap_or_else(RangeList::all);
                    let b_dim = b.dims.get(dim).cloned().unwrap_or_else(RangeList::all);
                    let intersected = a_dim.intersect(&b_dim);
                    if intersected.is_empty() {
                        // A and B are disjoint in this dimension, so A \ B = A
                        // All remaining slabs from later dimensions won't contribute
                        // (their prefix would be empty)
                        // But we've already added A above... wait, no, we added (A with diff in this dim)
                        // Actually if disjoint, diff = A_dim, so we added A already. We're done with this b.
                        break;
                    }
                    prefix_constraint.insert(dim.clone(), intersected);
                }
            }
            cur = DataArraySelection(next);
            if cur.is_empty() {
                break;
            }
        }
        cur
    }

    /// Conservative DNF subtraction: returns a disjunction (OR) of hyper-rectangles.
    ///
    /// Uses: A \\ B = union_dim ( A with that dim replaced by (A_dim \\ B_dim) ).
    /// This is exact (though may produce overlapping rectangles).
    // fn difference(&self, other: &HyperRectangleSelection) -> DataArraySelection {

    //     DataArraySelection(out)
    // }

    fn exclusive_or(&self, other: &DataArraySelection) -> DataArraySelection {
        self.difference(other).union(&other.difference(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::IndexRange;

    #[test]
    fn range_list_normalize_merges_overlaps_and_adjacent() {
        let mut rl = RangeList {
            ranges: vec![
                IndexRange { start: 0, end_exclusive: 5 },
                IndexRange { start: 5, end_exclusive: 10 },
                IndexRange { start: 2, end_exclusive: 3 },
                IndexRange { start: 20, end_exclusive: 21 },
            ],
        };
        rl.normalize();
        assert_eq!(
            rl.ranges(),
            &[
                IndexRange { start: 0, end_exclusive: 10 },
                IndexRange { start: 20, end_exclusive: 21 },
            ]
        );
    }

    #[test]
    fn range_list_intersect() {
        let a = RangeList::from_index_range(IndexRange {
            start: 0,
            end_exclusive: 10,
        });
        let b = RangeList::from_index_range(IndexRange {
            start: 5,
            end_exclusive: 12,
        });
        assert_eq!(
            a.intersect(&b).ranges(),
            &[IndexRange {
                start: 5,
                end_exclusive: 10
            }]
        );
    }

    #[test]
    fn range_list_difference_splits() {
        let a = RangeList::from_index_range(IndexRange {
            start: 0,
            end_exclusive: 10,
        });
        let b = RangeList::from_index_range(IndexRange {
            start: 3,
            end_exclusive: 7,
        });
        assert_eq!(
            a.difference(&b).ranges(),
            &[
                IndexRange {
                    start: 0,
                    end_exclusive: 3
                },
                IndexRange {
                    start: 7,
                    end_exclusive: 10
                },
            ]
        );
    }

}

