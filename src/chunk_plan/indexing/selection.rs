use std::collections::{BTreeMap, BTreeSet};

use smallvec::SmallVec;
use std::ops::Range;
use zarrs::array_subset::ArraySubset;

use super::types::BoundKind;

/// Dataset-level selection: variable name -> selection on that variable.
///
/// This is intended to be a pure index-selection representation (no chunking info).
// pub(crate) struct DatasetSelection(pub(crate) BTreeMap<String, DataArraySelection>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealSelection(BTreeMap<String, DataArraySelection>);
#[derive(Debug, Clone, PartialEq, Eq)]
pub (crate) enum DatasetSelection {
    /// If no selection was made:
    NoSelectionMade,
    /// If everything has been excluded
    Empty,
    /// Standard selection:
    Selection(RealSelection),
}

pub trait DSelection {
    fn vars(&self) -> impl Iterator<Item = (&str, &DataArraySelection)>;
    fn contains_dim(&self, dim: &str) -> bool;
    fn insert_dim(&mut self, dim: String, da: DataArraySelection);
}

impl From<BTreeMap<String, DataArraySelection>> for RealSelection {
    fn from(map: BTreeMap<String, DataArraySelection>) -> Self {
        RealSelection(map)
    }
}

impl RealSelection {
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn get(&self, var: &str) -> Option<&DataArraySelection> {
        self.0.get(var)
    }
}


// impl DSelection for EmptySelection {
//     fn vars(&self) -> impl Iterator<Item = (&str, &DataArraySelection)> {
//         Box::new(Vec::new().into_iter())
//     }
//     fn contains_dim(&self, dim: &str) -> bool {
//         false
//     }
//     fn insert_dim(&mut self, dim: String, da: DataArraySelection) {
//         // no-op
//     }
// }

// impl DSelection for NoSelection {
//     fn vars(&self) -> impl Iterator<Item = (&str, &DataArraySelection)> {
//         Box::new(Vec::new().into_iter())
//     }
//     fn contains_dim(&self, dim: &str) -> bool {
//         false
//     }
//     fn insert_dim(&mut self, dim: String, da: DataArraySelection) {
//         // no-op
//     }
// }

impl DSelection for RealSelection {
    fn vars(&self) -> impl Iterator<Item = (&str, &DataArraySelection)> {
        Box::new(self.0.iter().map(|(k, v)| (k.as_str(), v)))
    }
    fn contains_dim(&self, dim: &str) -> bool {
        self.0.contains_key(dim)
    }
    fn insert_dim(&mut self, dim: String, da: DataArraySelection) {
        self.0.insert(dim, da);
    }
}

impl DSelection for DatasetSelection {
    fn vars(&self) -> impl Iterator<Item = (&str, &DataArraySelection)> {
        match self {
            Self::Selection(selection) => Box::new(selection.vars()) as Box<dyn Iterator<Item = (&str, &DataArraySelection)>>,
            Self::NoSelectionMade => Box::new(std::iter::empty()),
            Self::Empty => Box::new(std::iter::empty()),
        }
    }
    fn contains_dim(&self, dim: &str) -> bool {
        match self {
            Self::Selection(selection) => selection.contains_dim(dim),
            Self::NoSelectionMade => false,
            Self::Empty => false,
        }
    }

    fn insert_dim(&mut self, dim: String, da: DataArraySelection) {
        match self {
            Self::Selection(selection) => {
                let _ = selection.0.insert(dim, da);
            }
            Self::NoSelectionMade => {
                *self = Self::Selection(RealSelection(BTreeMap::new()));
                self.insert_dim(dim, da);
            }
            Self::Empty => {
                // no-op
            }
        }
    }

}

/// Selection for a single array, expressed as a disjunction (OR) of hyper-rectangles.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DataArraySelection {
    /// SmallVec of dimension names (each name's position in the vec labels its index)
    /// So if we had [a, b, c], a would be our label for dim 0, b would be 1, c would be 2.
    dims: SmallVec<[String; 4]>,
    /// List of associated ArraySubsets
    subsets: ArraySubsetList
}

impl DataArraySelection {
    pub(crate) fn subsets_iter(&self) -> impl Iterator<Item = &ArraySubset> {
        self.subsets.0.iter()
    }
    pub (crate) fn all(
        dims: &[String],
        shape: Vec<u64>,
    ) -> Self {
        Self {
            dims: dims.iter().cloned().collect(),
            subsets: ArraySubsetList::from(vec![ArraySubset::new_with_shape(shape)]),
        }
    }

    pub (crate) fn from_subsets(dims: &[String], subsets: ArraySubsetList) -> Self {
        Self {
            dims: dims.iter().cloned().collect(),
            subsets: subsets,
        }
    }

}

pub trait Emptyable {
    fn empty() -> Self;
    fn is_empty(&self) -> bool;
}

/// Operations for sets of selections.
pub trait SetOperations: Emptyable {
    fn union(&self, other: &Self) -> Self;
    fn intersect(&self, other: &Self) -> Self;
    fn difference(&self, other: &Self) -> Self;
    fn exclusive_or(&self, other: &Self) -> Self;
}


impl Emptyable for RealSelection {
    /// TODO: slightly wrong since we would need the actual list of all vars for an empty (?)
    fn empty() -> Self {
        Self(BTreeMap::new())
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl SetOperations for RealSelection {
    fn union(&self, other: &Self) -> Self {
        let (s, o) = (self.0.clone(), other.0.clone());
        if s.is_empty() {
            return other.clone();
        }
        if o.is_empty() {
            return self.clone();
        }
        let mut out = s.clone();
        for (k, v) in o {
            out.entry(k.clone())
                .and_modify(|cur| *cur = cur.union(&v))
                .or_insert_with(|| v.clone());
        }
        RealSelection(out)
    }
    fn intersect(&self, other: &Self) -> Self {
        let (s, o) = (self.0.clone(), other.0.clone());
        let mut out = BTreeMap::new();
        for (k, a) in s {
            if let Some(b) = o.get(&k) {
                let sel = a.intersect(b);
                if !sel.is_empty() {
                    out.insert(k.clone(), sel);
                }
            }
        }
        RealSelection(out).into()
    }
    fn difference(&self, other: &Self) -> Self {
        let (s, o) = (self.0.clone(), other.0.clone());
        let mut out = BTreeMap::new();
        for (k, a) in s {
            if let Some(b) = o.get(&k) {
                let sel = a.difference(b);
                if !sel.is_empty() {
                    out.insert(k.clone(), sel);
                }
            } else {
                // Key only in self: keep it (we're subtracting nothing)
                out.insert(k.clone(), a.clone());
            }
        }
        RealSelection(out).into()
    }
    fn exclusive_or(&self, other: &Self) -> Self {
        let (s, o) = (self.0.clone(), other.0.clone());
        let mut out = BTreeMap::new();
        // Keys in self
        for (k, a) in s.iter() {
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
            if !&s.contains_key(&k) {
                out.insert(k.clone(), b.clone());
            }
        }
        RealSelection(out).into()
    }

}

impl Emptyable for DatasetSelection {
    fn empty() -> Self {
        Self::Empty
    }
    fn is_empty(&self) -> bool {
        match self {
            Self::Selection(s) => s.is_empty(),
            Self::NoSelectionMade | Self::Empty => true,
        }
    }
}


impl SetOperations for DatasetSelection {
    fn union(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Selection(s), Self::Selection(o)) => {
                Self::Selection(s.union(o))
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
                Self::Selection(s.intersect(o))
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
                Self::Selection(s.difference(o))
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

                Self::Selection(s.exclusive_or(o))
            }
            // NoSelectionMade XOR X = can't represent complement, stay conservative
            (Self::NoSelectionMade, _) | (_, Self::NoSelectionMade) => Self::NoSelectionMade,
            // Empty XOR X = X
            (Self::Empty, _) => other.clone(),
            (_, Self::Empty) => self.clone(),
        }
    }

}


impl Emptyable for DataArraySelection {
    fn empty() -> Self {
        Self {
            dims: SmallVec::new(),
            subsets: ArraySubsetList::empty(),
        }
    }
    fn is_empty(&self) -> bool {
        self.subsets.is_empty()
    }
}

impl SetOperations for DataArraySelection {

    fn union(&self, other: &DataArraySelection) -> DataArraySelection {
        if self.is_empty() && other.is_empty() {
            return DataArraySelection {
                dims: self.dims.clone(),
                subsets: ArraySubsetList::empty(),
            };
        }
        if self.is_empty() {
            return DataArraySelection {
                dims: other.dims.clone(),
                subsets: other.subsets.clone(),
            };
        }
        if other.is_empty() {
            return DataArraySelection {
                dims: self.dims.clone(),
                subsets: self.subsets.clone(),
            };
        }
        DataArraySelection {
            dims: self.dims.clone(),
            subsets: self.subsets.union(&other.subsets),
        }
    }

    fn intersect(&self, other: &DataArraySelection) -> DataArraySelection {
        if self.is_empty() || other.is_empty() {
            return DataArraySelection {
                dims: self.dims.clone(),
                subsets: ArraySubsetList::empty(),
            };
        }
        DataArraySelection {
            dims: self.dims.clone(),
            subsets: self.subsets.intersect(&other.subsets),
        }
    }

    fn difference(&self, other: &DataArraySelection) -> DataArraySelection {
        if self.is_empty() {
            return DataArraySelection {
                dims: self.dims.clone(),
                subsets: ArraySubsetList::empty(),
            };
        }
        if other.is_empty() {
            return DataArraySelection {
                dims: self.dims.clone(),
                subsets: self.subsets.clone(),
            };
        }
        DataArraySelection {
            dims: self.dims.clone(),
            subsets: self.subsets.difference(&other.subsets),
        }
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


#[derive(Debug, Clone, Default, PartialEq, Eq)]

pub (crate) struct ArraySubsetList(Vec<ArraySubset>);
impl ArraySubsetList {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }
    pub(crate) fn push(&mut self, subset: ArraySubset) {
        self.0.push(subset);
    }
    pub(crate) fn extend(&mut self, other: &ArraySubsetList) {
        self.0.extend(other.0.iter().cloned());
    }
}

impl Emptyable for ArraySubsetList {
    fn empty() -> Self {
        Self(Vec::new())
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Emptyable for ArraySubset {
    fn empty() -> Self {
        ArraySubset::new_empty(0)
    }
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

/// Compute the difference of two ranges A \ B.
/// Returns up to 2 ranges: the part of A before B and the part after B.
fn difference_ranges(a: &Range<u64>, b: &Range<u64>) -> Vec<Range<u64>> {
    // If no overlap, return A unchanged
    if b.end <= a.start || b.start >= a.end {
        return vec![a.clone()];
    }
    // If B completely contains A, return empty
    if b.start <= a.start && b.end >= a.end {
        return vec![];
    }
    
    let mut result = Vec::new();
    // Part before B
    if a.start < b.start {
        result.push(a.start..b.start);
    }
    // Part after B
    if a.end > b.end {
        result.push(b.end..a.end);
    }
    result
}

/// Compute the difference of two hyper-rectangles A \ B.
/// Returns a list of non-overlapping rectangles that together form A \ B.
///
/// The algorithm "peels off" slices of A that don't overlap with B,
/// dimension by dimension. This produces at most n rectangles for n dimensions.
fn hyper_rectangle_difference(a: &ArraySubset, b: &ArraySubset) -> Vec<ArraySubset> {
    let ndim = a.shape().len();
    if ndim != b.shape().len() {
        return vec![a.clone()];
    }
    
    // Check if there's any overlap
    match a.overlap(b) {
        Ok(overlap) if overlap.is_empty() => return vec![a.clone()],
        Err(_) => return vec![a.clone()],
        _ => {}
    }
    
    let a_ranges = a.to_ranges();
    let b_ranges = b.to_ranges();
    
    let mut result = Vec::new();
    // Track the "remaining" part of A as we peel off pieces
    let mut remaining_ranges = a_ranges.clone();
    
    for dim in 0..ndim {
        let a_range = &remaining_ranges[dim];
        let b_range = &b_ranges[dim];
        
        // Part of A before B in this dimension
        if a_range.start < b_range.start && b_range.start < a_range.end {
            let mut piece = remaining_ranges.clone();
            piece[dim] = a_range.start..b_range.start;
            result.push(ArraySubset::new_with_ranges(&piece));
        }
        
        // Part of A after B in this dimension
        if a_range.end > b_range.end && b_range.end > a_range.start {
            let mut piece = remaining_ranges.clone();
            piece[dim] = b_range.end..a_range.end;
            result.push(ArraySubset::new_with_ranges(&piece));
        }
        
        // Narrow remaining to the overlap in this dimension for next iteration
        remaining_ranges[dim] = a_range.start.max(b_range.start)..a_range.end.min(b_range.end);
        if remaining_ranges[dim].is_empty() {
            // No overlap in this dimension, we've already captured all of A
            break;
        }
    }
    
    // Filter out any empty subsets
    result.into_iter().filter(|s| !s.is_empty()).collect()
}

impl SetOperations for ArraySubset {
    fn union(&self, other: &ArraySubset) -> ArraySubset {
        // Union of two rectangles isn't a single rectangle in general
        // Return the bounding box as a conservative over-approximation
        // (The real union is handled at ArraySubsetList level)
        self.overlap(other).unwrap_or_else(|_| ArraySubset::empty())
    }
    fn intersect(&self, other: &ArraySubset) -> ArraySubset {
        self.overlap(other).unwrap_or_else(|_| ArraySubset::empty())
    }
    fn difference(&self, other: &ArraySubset) -> ArraySubset {
        // Single-rectangle difference returns multiple rectangles
        // This method is not ideal - use hyper_rectangle_difference directly
        // For now, return empty if they overlap, otherwise return self
        match self.overlap(other) {
            Ok(overlap) if !overlap.is_empty() => ArraySubset::new_empty(self.shape().len()),
            _ => self.clone(),
        }
    }
    fn exclusive_or(&self, other: &ArraySubset) -> ArraySubset {
        // XOR can't be represented as single rectangle
        ArraySubset::new_empty(self.shape().len())
    }
}

impl SetOperations for ArraySubsetList {
    fn union(&self, other: &ArraySubsetList) -> ArraySubsetList {
        let mut out = self.0.clone();
        out.extend(other.0.iter().cloned());
        Self(out)
    }
    fn intersect(&self, other: &ArraySubsetList) -> ArraySubsetList {
        let mut out = Vec::new();
        for a in &self.0 {
            for b in &other.0 {
                if let Ok(overlap) = a.overlap(b) {
                    if !overlap.is_empty() {
                        out.push(overlap);
                    }
                }
            }
        }
        Self(out)
    }
    fn difference(&self, other: &ArraySubsetList) -> ArraySubsetList {
        // For A \ B where A and B are both lists of rectangles:
        // Start with all rectangles from A, then subtract each rectangle from B
        let mut current = self.0.clone();
        
        for b in &other.0 {
            let mut next = Vec::new();
            for a in &current {
                // Subtract b from a, which may produce multiple rectangles
                let diff = hyper_rectangle_difference(a, b);
                next.extend(diff);
            }
            current = next;
        }
        
        Self(current)
    }
    fn exclusive_or(&self, other: &ArraySubsetList) -> ArraySubsetList {
        self.difference(other).union(&other.difference(self))
    }
}

impl From<Vec<ArraySubset>> for ArraySubsetList {
    fn from(subsets: Vec<ArraySubset>) -> Self {
        Self(subsets)
    }
}


