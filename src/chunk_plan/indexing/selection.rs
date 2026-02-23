use std::sync::Arc;

use smallvec::SmallVec;
use zarrs::array::ArraySubset;

use crate::IStr;

use super::grouped_selection::{
    ArraySelectionType, GroupedSelection,
};
use super::selection_base::DatasetSelectionBase;

/// Dataset-level selection: type alias for the generic `DatasetSelectionBase`.
///
/// This groups variables by their dimension signature to avoid duplication.
pub(crate) type DatasetSelection =
    DatasetSelectionBase<
        GroupedSelection<DataArraySelection>,
    >;

/// Selection for a single array, expressed as a disjunction (OR) of hyper-rectangles.
#[derive(
    Debug, Clone, Default, PartialEq, Eq,
)]
pub(crate) struct DataArraySelection {
    /// SmallVec of dimension names wrapped in Arc for cheap cloning in set operations.
    /// Each name's position in the vec labels its index.
    /// So if we had [a, b, c], a would be our label for dim 0, b would be 1, c would be 2.
    dims: Arc<SmallVec<[IStr; 4]>>,
    /// List of associated ArraySubsets
    subsets: ArraySubsetList,
}

impl DataArraySelection {
    pub(crate) fn from_subsets(
        dims: &[IStr],
        subsets: ArraySubsetList,
    ) -> Self {
        Self {
            dims: Arc::new(
                dims.iter().cloned().collect(),
            ),
            subsets,
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

// Note: Emptyable and SetOperations for DatasetSelection are provided by
// the generic DatasetSelectionBase<Sel> implementation in grouped_selection.rs

impl Emptyable for DataArraySelection {
    fn empty() -> Self {
        Self {
            dims: Arc::new(SmallVec::new()),
            subsets: ArraySubsetList::empty(),
        }
    }
    fn is_empty(&self) -> bool {
        self.subsets.is_empty()
    }
}

impl ArraySelectionType for DataArraySelection {
    fn all() -> Self {
        // For concrete selections, "all" is only meaningful when we know the shape.
        // This method exists for trait completeness but should not be called directly.
        // Use `from_subsets` with actual bounds instead.
        // We return empty here as a fallback - in practice, concrete "all" selections
        // come from materializing LazyArraySelection::all() with shape info.
        Self::empty()
    }
}

impl SetOperations for DataArraySelection {
    fn union(
        &self,
        other: &DataArraySelection,
    ) -> DataArraySelection {
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
            subsets: self
                .subsets
                .union(&other.subsets),
        }
    }

    fn intersect(
        &self,
        other: &DataArraySelection,
    ) -> DataArraySelection {
        if self.is_empty() || other.is_empty() {
            return DataArraySelection {
                dims: self.dims.clone(),
                subsets: ArraySubsetList::empty(),
            };
        }
        DataArraySelection {
            dims: self.dims.clone(),
            subsets: self
                .subsets
                .intersect(&other.subsets),
        }
    }

    fn difference(
        &self,
        other: &DataArraySelection,
    ) -> DataArraySelection {
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
            subsets: self
                .subsets
                .difference(&other.subsets),
        }
    }

    /// Conservative DNF subtraction: returns a disjunction (OR) of hyper-rectangles.
    ///
    /// Uses: A \\ B = union_dim ( A with that dim replaced by (A_dim \\ B_dim) ).
    /// This is exact (though may produce overlapping rectangles).
    // fn difference(&self, other: &HyperRectangleSelection) -> DataArraySelection {

    //     DataArraySelection(out)
    // }

    fn exclusive_or(
        &self,
        other: &DataArraySelection,
    ) -> DataArraySelection {
        self.difference(other)
            .union(&other.difference(self))
    }
}

#[derive(
    Debug, Clone, Default, PartialEq, Eq,
)]

pub(crate) struct ArraySubsetList(
    Vec<ArraySubset>,
);

impl From<Vec<ArraySubset>> for ArraySubsetList {
    fn from(subsets: Vec<ArraySubset>) -> Self {
        Self(subsets)
    }
}

impl From<DataArraySelection>
    for ArraySubsetList
{
    fn from(
        selection: DataArraySelection,
    ) -> Self {
        selection.subsets
    }
}

impl ArraySubsetList {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }
    pub(crate) fn push(
        &mut self,
        subset: ArraySubset,
    ) {
        self.0.push(subset);
    }
    pub(crate) fn subsets_iter(
        &self,
    ) -> impl Iterator<Item = &ArraySubset> {
        self.0.iter()
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

/// Compute the difference of two hyper-rectangles A \ B.
/// Returns a list of non-overlapping rectangles that together form A \ B.
///
/// The algorithm "peels off" slices of A that don't overlap with B,
/// dimension by dimension. This produces at most n rectangles for n dimensions.
fn hyper_rectangle_difference(
    a: &ArraySubset,
    b: &ArraySubset,
) -> Vec<ArraySubset> {
    let ndim = a.shape().len();
    if ndim != b.shape().len() {
        return vec![a.clone()];
    }

    // Check if there's any overlap
    match a.overlap(b) {
        Ok(overlap) if overlap.is_empty() => {
            return vec![a.clone()];
        }
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
        if a_range.start < b_range.start
            && b_range.start < a_range.end
        {
            let mut piece =
                remaining_ranges.clone();
            piece[dim] =
                a_range.start..b_range.start;
            result.push(
                ArraySubset::new_with_ranges(
                    &piece,
                ),
            );
        }

        // Part of A after B in this dimension
        if a_range.end > b_range.end
            && b_range.end > a_range.start
        {
            let mut piece =
                remaining_ranges.clone();
            piece[dim] = b_range.end..a_range.end;
            result.push(
                ArraySubset::new_with_ranges(
                    &piece,
                ),
            );
        }

        // Narrow remaining to the overlap in this dimension for next iteration
        remaining_ranges[dim] =
            a_range.start.max(b_range.start)
                ..a_range.end.min(b_range.end);
        if remaining_ranges[dim].is_empty() {
            // No overlap in this dimension, we've already captured all of A
            break;
        }
    }

    // Filter out any empty subsets
    result
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect()
}

impl SetOperations for ArraySubset {
    fn union(
        &self,
        other: &ArraySubset,
    ) -> ArraySubset {
        // Union of two rectangles isn't a single rectangle in general
        // Return the bounding box as a conservative over-approximation
        // (The real union is handled at ArraySubsetList level)
        self.overlap(other).unwrap_or_else(|_| {
            ArraySubset::empty()
        })
    }
    fn intersect(
        &self,
        other: &ArraySubset,
    ) -> ArraySubset {
        self.overlap(other).unwrap_or_else(|_| {
            ArraySubset::empty()
        })
    }
    fn difference(
        &self,
        other: &ArraySubset,
    ) -> ArraySubset {
        // Single-rectangle difference returns multiple rectangles
        // This method is not ideal - use hyper_rectangle_difference directly
        // For now, return empty if they overlap, otherwise return self
        match self.overlap(other) {
            Ok(overlap)
                if !overlap.is_empty() =>
            {
                ArraySubset::new_empty(
                    self.shape().len(),
                )
            }
            _ => self.clone(),
        }
    }
    fn exclusive_or(
        &self,
        _other: &ArraySubset,
    ) -> ArraySubset {
        // XOR can't be represented as single rectangle
        ArraySubset::new_empty(self.shape().len())
    }
}

impl SetOperations for ArraySubsetList {
    fn union(
        &self,
        other: &ArraySubsetList,
    ) -> ArraySubsetList {
        let mut out = self.0.clone();
        out.extend(other.0.iter().cloned());
        Self(out)
    }
    fn intersect(
        &self,
        other: &ArraySubsetList,
    ) -> ArraySubsetList {
        let mut out = Vec::new();
        for a in &self.0 {
            for b in &other.0 {
                if let Ok(overlap) = a.overlap(b)
                {
                    if !overlap.is_empty() {
                        out.push(overlap);
                    }
                }
            }
        }
        Self(out)
    }
    fn difference(
        &self,
        other: &ArraySubsetList,
    ) -> ArraySubsetList {
        // For A \ B where A and B are both lists of rectangles:
        // Start with all rectangles from A, then subtract each rectangle from B
        let mut current = self.0.clone();

        for b in &other.0 {
            let mut next = Vec::new();
            for a in &current {
                // Subtract b from a, which may produce multiple rectangles
                let diff =
                    hyper_rectangle_difference(
                        a, b,
                    );
                next.extend(diff);
            }
            current = next;
        }

        Self(current)
    }
    fn exclusive_or(
        &self,
        other: &ArraySubsetList,
    ) -> ArraySubsetList {
        self.difference(other)
            .union(&other.difference(self))
    }
}
