use std::collections::{BTreeMap, BTreeSet};

use super::types::BoundKind;

/// Dataset-level selection: variable name -> selection on that variable.
///
/// This is intended to be a pure index-selection representation (no chunking info).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DatasetSelection(pub(crate) BTreeMap<String, DataArraySelection>);

impl DatasetSelection {
    pub(crate) fn empty() -> Self {
        Self(BTreeMap::new())
    }

    /// Returns a selection containing `vars`, each selected entirely (all indices).
    pub(crate) fn all_for_vars(vars: impl IntoIterator<Item = String>) -> Self {
        let mut m = BTreeMap::new();
        for v in vars {
            m.insert(v, DataArraySelection::all());
        }
        Self(m)
    }

    pub(crate) fn vars(&self) -> impl Iterator<Item = (&str, &DataArraySelection)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub(crate) fn for_vars_with_selection(
        vars: impl IntoIterator<Item = String>,
        sel: DataArraySelection,
    ) -> Self {
        let mut m = BTreeMap::new();
        for v in vars {
            m.insert(v, sel.clone());
        }
        Self(m)
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

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn union(&self, other: &DataArraySelection) -> DataArraySelection {
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

    pub(crate) fn intersect(&self, other: &DataArraySelection) -> DataArraySelection {
        if self.is_empty() || other.is_empty() {
            return DataArraySelection::empty();
        }
        let mut out: Vec<HyperRectangleSelection> = Vec::new();
        for a in &self.0 {
            for b in &other.0 {
                if let Some(r) = a.intersect(b) {
                    out.push(r);
                }
            }
        }
        DataArraySelection(out)
    }

    pub(crate) fn difference(&self, other: &DataArraySelection) -> DataArraySelection {
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
                next.extend(a.difference(b).0);
            }
            cur = DataArraySelection(next);
            if cur.is_empty() {
                break;
            }
        }
        cur
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

    pub(crate) fn intersect(&self, other: &HyperRectangleSelection) -> Option<HyperRectangleSelection> {
        if self.is_empty() || other.is_empty() {
            return None;
        }

        let mut out: BTreeMap<String, RangeList> = BTreeMap::new();
        for dim in self.dim_names_union(other) {
            let a = self.dims.get(&dim).cloned().unwrap_or_else(RangeList::all);
            let b = other.dims.get(&dim).cloned().unwrap_or_else(RangeList::all);
            let r = a.intersect(&b);
            if r.is_empty() {
                return None;
            }
            if r != RangeList::all() {
                out.insert(dim, r);
            }
        }
        Some(HyperRectangleSelection {
            dims: out,
            empty: false,
        })
    }

    /// Conservative DNF subtraction: returns a disjunction (OR) of hyper-rectangles.
    ///
    /// Uses: A \\ B = union_dim ( A with that dim replaced by (A_dim \\ B_dim) ).
    /// This is exact (though may produce overlapping rectangles).
    pub(crate) fn difference(&self, other: &HyperRectangleSelection) -> DataArraySelection {
        if self.is_empty() {
            return DataArraySelection::empty();
        }
        if other.is_empty() {
            return DataArraySelection(vec![self.clone()]);
        }
        if other.dims.is_empty() && !other.empty {
            // subtracting the universal rectangle => empty
            return DataArraySelection::empty();
        }

        let dims = self.dim_names_union(other);
        let mut out: Vec<HyperRectangleSelection> = Vec::new();
        for dim in dims {
            let a = self.dims.get(&dim).cloned().unwrap_or_else(RangeList::all);
            let b = other.dims.get(&dim).cloned().unwrap_or_else(RangeList::all);
            let diff = a.difference(&b);
            if diff.is_empty() {
                continue;
            }

            let mut rect_dims = self.dims.clone();
            if diff == RangeList::all() {
                rect_dims.remove(&dim);
            } else {
                rect_dims.insert(dim, diff);
            }
            out.push(HyperRectangleSelection {
                dims: rect_dims,
                empty: false,
            });
        }
        DataArraySelection(out)
    }
}

impl DatasetSelection {
    pub(crate) fn union(&self, other: &DatasetSelection) -> DatasetSelection {
        if self.0.is_empty() {
            return other.clone();
        }
        if other.0.is_empty() {
            return self.clone();
        }
        let mut out = self.0.clone();
        for (k, v) in &other.0 {
            out.entry(k.clone())
                .and_modify(|cur| *cur = cur.union(v))
                .or_insert_with(|| v.clone());
        }
        DatasetSelection(out)
    }

    pub(crate) fn intersect(&self, other: &DatasetSelection) -> DatasetSelection {
        let mut out = BTreeMap::new();
        for (k, a) in &self.0 {
            if let Some(b) = other.0.get(k) {
                let sel = a.intersect(b);
                if !sel.is_empty() {
                    out.insert(k.clone(), sel);
                }
            }
        }
        DatasetSelection(out)
    }

    pub(crate) fn difference(&self, other: &DatasetSelection) -> DatasetSelection {
        let mut out = BTreeMap::new();
        for (k, a) in &self.0 {
            let sel = if let Some(b) = other.0.get(k) {
                a.difference(b)
            } else {
                a.clone()
            };
            if !sel.is_empty() {
                out.insert(k.clone(), sel);
            }
        }
        DatasetSelection(out)
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

impl ScalarRange {
    pub(crate) fn all() -> Self {
        Self {
            start: None,
            end: None,
        }
    }
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

    pub(crate) fn from_scalar_ranges(ranges: impl IntoIterator<Item = ScalarRange>) -> Self {
        let mut out = Self::empty();
        for r in ranges {
            if let Some(ir) = scalar_range_to_index_range(r) {
                out.ranges.push(ir);
            }
        }
        out.normalize();
        out
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

    pub(crate) fn union(&self, other: &RangeList) -> RangeList {
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

    pub(crate) fn intersect(&self, other: &RangeList) -> RangeList {
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

    pub(crate) fn difference(&self, other: &RangeList) -> RangeList {
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
}

fn scalar_range_to_index_range(r: ScalarRange) -> Option<super::types::IndexRange> {
    let start = match r.start {
        None => 0,
        Some((v, BoundKind::Inclusive)) => v,
        Some((v, BoundKind::Exclusive)) => v.saturating_add(1),
    };
    let end_exclusive = match r.end {
        None => u64::MAX,
        Some((v, BoundKind::Exclusive)) => v,
        Some((v, BoundKind::Inclusive)) => v.saturating_add(1),
    };
    let out = super::types::IndexRange { start, end_exclusive };
    if out.is_empty() { None } else { Some(out) }
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
    fn scalar_range_inclusive_exclusive_normalizes() {
        let rl = RangeList::from_scalar_ranges([ScalarRange {
            start: Some((10, BoundKind::Exclusive)),
            end: Some((20, BoundKind::Inclusive)),
        }]);
        assert_eq!(rl.ranges(), &[IndexRange { start: 11, end_exclusive: 21 }]);
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

    #[test]
    fn dataset_selection_set_ops_basic() {
        let vars = vec!["a".to_string(), "b".to_string()];
        let all = DatasetSelection::all_for_vars(vars.clone());

        let only_a = DatasetSelection::all_for_vars(vec!["a".to_string()]);
        assert!(all.intersect(&only_a).0.contains_key("a"));
        assert!(!all.intersect(&only_a).0.contains_key("b"));

        let diff = all.difference(&only_a);
        assert!(!diff.0.contains_key("a"));
        assert!(diff.0.contains_key("b"));

        let back = diff.union(&only_a);
        assert!(back.0.contains_key("a"));
        assert!(back.0.contains_key("b"));
    }
}

