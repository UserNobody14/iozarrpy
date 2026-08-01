//! N-dimensional hyper-rectangle set primitive used by the new
//! `Expr -> GridJoinTree` chunk-plan builder.
//!
//! A [`RectangleSet`] is a union of [`HyperRectangle`]s whose dimensions are
//! already resolved to `std::ops::Range<u64>` index ranges. It supports the
//! usual boolean set operations (union, intersect, difference, negate, xor)
//! and exposes `iter_subsets` to feed the chunk grid downstream.

use std::ops::Range;

use smallvec::SmallVec;
use zarrs::array::ArraySubset;

use crate::shared::IStr;

/// One N-dimensional hyper-rectangle: a per-dimension range list whose
/// cross-product yields the represented index cells.
#[derive(Debug, Clone, Default)]
pub(crate) struct HyperRectangle {
    /// One entry per dimension, in dim order; each entry is a list of
    /// non-overlapping ranges along that dimension.
    pub(crate) per_dim:
        SmallVec<[Vec<Range<u64>>; 4]>,
}

/// A union of [`HyperRectangle`]s sharing the same dim ordering. Used by
/// the new chunk-plan builder to accumulate index-range constraints across
/// logical OR / AND / NOT.
#[derive(Debug, Clone, Default)]
pub(crate) struct RectangleSet {
    pub(crate) dims: SmallVec<[IStr; 4]>,
    pub(crate) shape: SmallVec<[u64; 4]>,
    pub(crate) rects: Vec<HyperRectangle>,
}

impl HyperRectangle {
    /// True iff at least one dimension has no surviving range.
    fn is_empty(&self) -> bool {
        self.per_dim.iter().any(|d| {
            d.is_empty()
                || d.iter().all(|r| r.is_empty())
        })
    }
}

impl RectangleSet {
    /// Empty set (no rectangles) for the given dim/shape signature.
    pub(crate) fn empty(
        dims: SmallVec<[IStr; 4]>,
        shape: SmallVec<[u64; 4]>,
    ) -> Self {
        debug_assert_eq!(dims.len(), shape.len());
        Self {
            dims,
            shape,
            rects: Vec::new(),
        }
    }

    /// Full set covering the entire `0..shape[d]` cube.
    pub(crate) fn full(
        dims: SmallVec<[IStr; 4]>,
        shape: SmallVec<[u64; 4]>,
    ) -> Self {
        debug_assert_eq!(dims.len(), shape.len());
        let per_dim: SmallVec<
            [Vec<Range<u64>>; 4],
        > = shape
            .iter()
            .map(|&s| vec![0..s])
            .collect();
        let rect = HyperRectangle { per_dim };
        Self {
            dims,
            shape,
            rects: vec![rect],
        }
    }

    /// A single-rectangle set built from per-dim range lists.
    pub(crate) fn from_per_dim(
        dims: SmallVec<[IStr; 4]>,
        shape: SmallVec<[u64; 4]>,
        per_dim: SmallVec<[Vec<Range<u64>>; 4]>,
    ) -> Self {
        debug_assert_eq!(dims.len(), shape.len());
        debug_assert_eq!(
            dims.len(),
            per_dim.len()
        );
        Self {
            dims,
            shape,
            rects: vec![HyperRectangle {
                per_dim,
            }],
        }
    }

    /// True when no cells are represented.
    pub(crate) fn is_empty(&self) -> bool {
        self.rects.is_empty()
            || self
                .rects
                .iter()
                .all(HyperRectangle::is_empty)
    }

    /// Union: concatenate the two rectangle lists.
    pub(crate) fn union(
        &self,
        other: &Self,
    ) -> Self {
        assert_eq!(
            self.dims, other.dims,
            "RectangleSet::union requires matching dims"
        );
        assert_eq!(
            self.shape, other.shape,
            "RectangleSet::union requires matching shape"
        );
        let mut rects = Vec::with_capacity(
            self.rects.len() + other.rects.len(),
        );
        rects.extend(self.rects.iter().cloned());
        rects.extend(other.rects.iter().cloned());
        Self {
            dims: self.dims.clone(),
            shape: self.shape.clone(),
            rects,
        }
    }

    /// Intersection: per (a, b) pair, intersect each per-dim range list and
    /// emit one rectangle for the cross-product (rect_a × rect_b).
    pub(crate) fn intersect(
        &self,
        other: &Self,
    ) -> Self {
        assert_eq!(
            self.dims, other.dims,
            "RectangleSet::intersect requires matching dims"
        );
        assert_eq!(
            self.shape, other.shape,
            "RectangleSet::intersect requires matching shape"
        );
        let mut rects = Vec::new();
        for a in &self.rects {
            for b in &other.rects {
                let inter = intersect_rects(a, b);
                if !inter.is_empty() {
                    rects.push(inter);
                }
            }
        }
        Self {
            dims: self.dims.clone(),
            shape: self.shape.clone(),
            rects,
        }
    }

    /// Difference: subtract every rect of `other` from every rect of `self`.
    pub(crate) fn difference(
        &self,
        other: &Self,
    ) -> Self {
        assert_eq!(
            self.dims, other.dims,
            "RectangleSet::difference requires matching dims"
        );
        assert_eq!(
            self.shape, other.shape,
            "RectangleSet::difference requires matching shape"
        );

        // Start with all rectangles from self; subtract each rect of other.
        let mut current: Vec<HyperRectangle> =
            self.rects.iter().cloned().collect();

        for s in &other.rects {
            let mut next = Vec::new();
            for a in &current {
                next.extend(
                    hyper_rectangle_difference(
                        a, s,
                    ),
                );
            }
            current = next;
            if current.is_empty() {
                break;
            }
        }

        Self {
            dims: self.dims.clone(),
            shape: self.shape.clone(),
            rects: current,
        }
    }

    /// Set complement relative to the full cube `0..shape[d]`.
    pub(crate) fn negate(&self) -> Self {
        Self::full(
            self.dims.clone(),
            self.shape.clone(),
        )
        .difference(self)
    }

    /// Symmetric difference: `(self \ other) ∪ (other \ self)`.
    pub(crate) fn exclusive_or(
        &self,
        other: &Self,
    ) -> Self {
        self.difference(other)
            .union(&other.difference(self))
    }

    /// Yield one [`ArraySubset`] per cross-product cell of every rectangle.
    /// Empty cells (any dim with start >= end) are skipped.
    pub(crate) fn iter_subsets(
        &self,
    ) -> impl Iterator<Item = ArraySubset> + '_
    {
        self.rects.iter().flat_map(|rect| {
            iter_rect_subsets(rect)
        })
    }
}

/// Intersect two hyper-rectangles by intersecting each pair of per-dim
/// ranges. The result is a single rectangle whose per-dim list is the
/// cartesian product of `a.per_dim[d] × b.per_dim[d]` (after dropping empty
/// intersections).
fn intersect_rects(
    a: &HyperRectangle,
    b: &HyperRectangle,
) -> HyperRectangle {
    debug_assert_eq!(
        a.per_dim.len(),
        b.per_dim.len()
    );
    let mut per_dim: SmallVec<
        [Vec<Range<u64>>; 4],
    > = SmallVec::with_capacity(a.per_dim.len());
    for (a_d, b_d) in
        a.per_dim.iter().zip(b.per_dim.iter())
    {
        let mut dim_out = Vec::new();
        for ar in a_d {
            for br in b_d {
                let lo = ar.start.max(br.start);
                let hi = ar.end.min(br.end);
                if lo < hi {
                    dim_out.push(lo..hi);
                }
            }
        }
        per_dim.push(dim_out);
    }
    HyperRectangle { per_dim }
}

/// Iterate the cartesian product of a single rectangle's per-dim ranges,
/// yielding one [`ArraySubset`] per cell.
fn iter_rect_subsets(
    rect: &HyperRectangle,
) -> Box<dyn Iterator<Item = ArraySubset> + '_> {
    if rect.is_empty() {
        return Box::new(std::iter::empty());
    }

    let mut acc: Vec<Vec<Range<u64>>> =
        vec![Vec::new()];
    for dim_ranges in &rect.per_dim {
        let mut next = Vec::new();
        for prefix in &acc {
            for r in dim_ranges {
                if r.is_empty() {
                    continue;
                }
                let mut new_prefix =
                    Vec::with_capacity(
                        prefix.len() + 1,
                    );
                new_prefix
                    .extend_from_slice(prefix);
                new_prefix.push(r.clone());
                next.push(new_prefix);
            }
        }
        acc = next;
    }

    Box::new(acc.into_iter().map(|ranges| {
        ArraySubset::new_with_ranges(&ranges)
    }))
}

/// Subtract `subtractor` from `minuend` along the shared dim ordering.
/// Returns "remainder" rectangles that together cover `minuend \ subtractor`
/// and do not overlap with `subtractor`.
///
/// Algorithm: expand both inputs into single-range-per-dim sub-rectangles
/// (their internal per-dim cartesian product), then for each pair apply the
/// classic "peel off" dimension-by-dimension difference that splits each
/// dim's range into "before" and "after" fragments around the overlap.
pub(crate) fn hyper_rectangle_difference(
    minuend: &HyperRectangle,
    subtractor: &HyperRectangle,
) -> Vec<HyperRectangle> {
    if minuend.is_empty() {
        return Vec::new();
    }
    if subtractor.is_empty() {
        return vec![minuend.clone()];
    }
    if minuend.per_dim.len()
        != subtractor.per_dim.len()
    {
        return vec![minuend.clone()];
    }

    let m_simple = expand_to_simple(minuend);
    let s_simple = expand_to_simple(subtractor);

    let mut current: Vec<Vec<Range<u64>>> =
        m_simple;
    for s in &s_simple {
        let mut next = Vec::new();
        for a in &current {
            next.extend(simple_rect_difference(
                a, s,
            ));
        }
        current = next;
        if current.is_empty() {
            break;
        }
    }

    current
        .into_iter()
        .map(|ranges| {
            let per_dim: SmallVec<
                [Vec<Range<u64>>; 4],
            > = ranges
                .into_iter()
                .map(|r| vec![r])
                .collect();
            HyperRectangle { per_dim }
        })
        .collect()
}

/// Expand a [`HyperRectangle`] into the list of single-range-per-dim
/// rectangles that constitute its internal cartesian product.
fn expand_to_simple(
    rect: &HyperRectangle,
) -> Vec<Vec<Range<u64>>> {
    let mut acc: Vec<Vec<Range<u64>>> =
        vec![Vec::new()];
    for dim_ranges in &rect.per_dim {
        let mut next = Vec::new();
        for prefix in &acc {
            for r in dim_ranges {
                if r.is_empty() {
                    continue;
                }
                let mut new_prefix =
                    Vec::with_capacity(
                        prefix.len() + 1,
                    );
                new_prefix
                    .extend_from_slice(prefix);
                new_prefix.push(r.clone());
                next.push(new_prefix);
            }
        }
        acc = next;
    }
    acc
}

/// Difference of two single-range-per-dim rectangles using the peel-off
/// algorithm salvaged from the legacy `selection.rs`.
fn simple_rect_difference(
    a: &[Range<u64>],
    b: &[Range<u64>],
) -> Vec<Vec<Range<u64>>> {
    let ndim = a.len();
    if ndim != b.len() {
        return vec![a.to_vec()];
    }

    // No overlap -> minuend survives intact.
    for d in 0..ndim {
        let lo = a[d].start.max(b[d].start);
        let hi = a[d].end.min(b[d].end);
        if lo >= hi {
            return vec![a.to_vec()];
        }
    }

    let mut result: Vec<Vec<Range<u64>>> =
        Vec::new();
    let mut remaining: Vec<Range<u64>> =
        a.to_vec();

    for dim in 0..ndim {
        let a_range = remaining[dim].clone();
        let b_range = b[dim].clone();

        // Slice of A strictly before B in this dim.
        if a_range.start < b_range.start
            && b_range.start < a_range.end
        {
            let mut piece = remaining.clone();
            piece[dim] =
                a_range.start..b_range.start;
            result.push(piece);
        }

        // Slice of A strictly after B in this dim.
        if a_range.end > b_range.end
            && b_range.end > a_range.start
        {
            let mut piece = remaining.clone();
            piece[dim] = b_range.end..a_range.end;
            result.push(piece);
        }

        // Narrow remaining to the per-dim overlap for the next iteration.
        remaining[dim] =
            a_range.start.max(b_range.start)
                ..a_range.end.min(b_range.end);
        if remaining[dim].is_empty() {
            break;
        }
    }

    result
        .into_iter()
        .filter(|p| {
            p.iter().all(|r| !r.is_empty())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::IntoIStr;

    fn dims1() -> SmallVec<[IStr; 4]> {
        let mut v = SmallVec::new();
        v.push("x".istr());
        v
    }

    fn dims2() -> SmallVec<[IStr; 4]> {
        let mut v = SmallVec::new();
        v.push("x".istr());
        v.push("y".istr());
        v
    }

    fn shape_of(
        values: &[u64],
    ) -> SmallVec<[u64; 4]> {
        values.iter().copied().collect()
    }

    fn rect1d(
        dims: SmallVec<[IStr; 4]>,
        shape: SmallVec<[u64; 4]>,
        ranges: Vec<Range<u64>>,
    ) -> RectangleSet {
        let mut per_dim: SmallVec<
            [Vec<Range<u64>>; 4],
        > = SmallVec::new();
        per_dim.push(ranges);
        RectangleSet::from_per_dim(
            dims, shape, per_dim,
        )
    }

    fn rect2d(
        dims: SmallVec<[IStr; 4]>,
        shape: SmallVec<[u64; 4]>,
        x: Vec<Range<u64>>,
        y: Vec<Range<u64>>,
    ) -> RectangleSet {
        let mut per_dim: SmallVec<
            [Vec<Range<u64>>; 4],
        > = SmallVec::new();
        per_dim.push(x);
        per_dim.push(y);
        RectangleSet::from_per_dim(
            dims, shape, per_dim,
        )
    }

    /// Total number of cells across all subsets yielded by `iter_subsets`.
    fn cell_count(set: &RectangleSet) -> u64 {
        set.iter_subsets()
            .map(|s| {
                s.shape().iter().product::<u64>()
            })
            .sum()
    }

    /// Materialize all (dim0, dim1, ...) cells covered by `set` for
    /// equality testing in tests.
    fn collect_cells(
        set: &RectangleSet,
    ) -> std::collections::BTreeSet<Vec<u64>>
    {
        let mut out =
            std::collections::BTreeSet::new();
        for sub in set.iter_subsets() {
            let ranges = sub.to_ranges();
            let mut acc: Vec<Vec<u64>> =
                vec![Vec::new()];
            for r in &ranges {
                let mut next = Vec::new();
                for prefix in &acc {
                    for v in r.clone() {
                        let mut np =
                            prefix.clone();
                        np.push(v);
                        next.push(np);
                    }
                }
                acc = next;
            }
            out.extend(acc);
        }
        out
    }

    #[test]
    fn empty_constructor_is_empty() {
        let s = RectangleSet::empty(
            dims1(),
            shape_of(&[10]),
        );
        assert!(s.is_empty());
        assert_eq!(s.iter_subsets().count(), 0);
    }

    #[test]
    fn full_constructor_covers_everything() {
        let s = RectangleSet::full(
            dims2(),
            shape_of(&[3, 4]),
        );
        assert!(!s.is_empty());
        assert_eq!(cell_count(&s), 12);
        let cells = collect_cells(&s);
        assert_eq!(cells.len(), 12);
    }

    #[test]
    fn union_concatenates_rects() {
        let a = rect1d(
            dims1(),
            shape_of(&[10]),
            vec![0..3],
        );
        let b = rect1d(
            dims1(),
            shape_of(&[10]),
            vec![5..8],
        );
        let u = a.union(&b);
        assert_eq!(u.rects.len(), 2);
        assert_eq!(cell_count(&u), 6);
    }

    #[test]
    fn intersect_single_dim() {
        let a = rect1d(
            dims1(),
            shape_of(&[10]),
            vec![0..6],
        );
        let b = rect1d(
            dims1(),
            shape_of(&[10]),
            vec![3..9],
        );
        let i = a.intersect(&b);
        let cells = collect_cells(&i);
        let expected: std::collections::BTreeSet<
            Vec<u64>,
        > = (3..6).map(|v| vec![v]).collect();
        assert_eq!(cells, expected);
    }

    #[test]
    fn intersect_two_dims_cross_product() {
        let a = rect2d(
            dims2(),
            shape_of(&[10, 10]),
            vec![0..5],
            vec![0..5],
        );
        let b = rect2d(
            dims2(),
            shape_of(&[10, 10]),
            vec![3..8],
            vec![2..6],
        );
        let i = a.intersect(&b);
        let cells = collect_cells(&i);
        let mut expected =
            std::collections::BTreeSet::new();
        for x in 3..5 {
            for y in 2..5 {
                expected.insert(vec![x, y]);
            }
        }
        assert_eq!(cells, expected);
    }

    #[test]
    fn difference_partial_overlap_1d() {
        let a = rect1d(
            dims1(),
            shape_of(&[10]),
            vec![0..6],
        );
        let b = rect1d(
            dims1(),
            shape_of(&[10]),
            vec![3..8],
        );
        let d = a.difference(&b);
        let cells = collect_cells(&d);
        let expected: std::collections::BTreeSet<
            Vec<u64>,
        > = (0..3).map(|v| vec![v]).collect();
        assert_eq!(cells, expected);
    }

    #[test]
    fn difference_partial_overlap_2d() {
        let a = rect2d(
            dims2(),
            shape_of(&[10, 10]),
            vec![0..4],
            vec![0..4],
        );
        let b = rect2d(
            dims2(),
            shape_of(&[10, 10]),
            vec![2..6],
            vec![2..6],
        );
        let d = a.difference(&b);
        let cells = collect_cells(&d);
        let mut expected =
            std::collections::BTreeSet::new();
        for x in 0..4 {
            for y in 0..4 {
                if !(x >= 2 && y >= 2) {
                    expected.insert(vec![x, y]);
                }
            }
        }
        assert_eq!(cells, expected);
    }

    #[test]
    fn difference_minuend_contains_subtractor() {
        let a = rect2d(
            dims2(),
            shape_of(&[10, 10]),
            vec![0..6],
            vec![0..6],
        );
        let b = rect2d(
            dims2(),
            shape_of(&[10, 10]),
            vec![1..5],
            vec![1..5],
        );
        let d = a.difference(&b);
        let cells = collect_cells(&d);
        // Outer ring of the 6x6 minus inner 4x4.
        let mut expected =
            std::collections::BTreeSet::new();
        for x in 0..6 {
            for y in 0..6 {
                if !(x >= 1
                    && x < 5
                    && y >= 1
                    && y < 5)
                {
                    expected.insert(vec![x, y]);
                }
            }
        }
        assert_eq!(cells, expected);
        // 36 - 16 = 20.
        assert_eq!(cells.len(), 20);
    }

    #[test]
    fn difference_subtractor_contains_minuend() {
        let a = rect2d(
            dims2(),
            shape_of(&[10, 10]),
            vec![2..4],
            vec![2..4],
        );
        let b = rect2d(
            dims2(),
            shape_of(&[10, 10]),
            vec![0..6],
            vec![0..6],
        );
        let d = a.difference(&b);
        assert!(d.is_empty());
        assert_eq!(cell_count(&d), 0);
    }

    #[test]
    fn negate_is_involutive_on_full_and_empty() {
        let full = RectangleSet::full(
            dims2(),
            shape_of(&[3, 3]),
        );
        let empty = RectangleSet::empty(
            dims2(),
            shape_of(&[3, 3]),
        );

        let neg_full = full.negate();
        assert!(neg_full.is_empty());
        let neg_neg_full = neg_full.negate();
        let cells_full = collect_cells(&full);
        let cells_neg_neg =
            collect_cells(&neg_neg_full);
        assert_eq!(cells_full, cells_neg_neg);

        let neg_empty = empty.negate();
        assert_eq!(cell_count(&neg_empty), 9);
        let neg_neg_empty = neg_empty.negate();
        assert!(neg_neg_empty.is_empty());
    }

    #[test]
    fn exclusive_or_is_symmetric() {
        let a = rect2d(
            dims2(),
            shape_of(&[10, 10]),
            vec![0..4],
            vec![0..4],
        );
        let b = rect2d(
            dims2(),
            shape_of(&[10, 10]),
            vec![2..6],
            vec![2..6],
        );
        let xor_ab = a.exclusive_or(&b);
        let xor_ba = b.exclusive_or(&a);
        assert_eq!(
            collect_cells(&xor_ab),
            collect_cells(&xor_ba),
        );
        // Symmetric difference of two 4x4s overlapping in a 2x2.
        // |A| + |B| - 2|A∩B| = 16 + 16 - 2*4 = 24.
        assert_eq!(
            collect_cells(&xor_ab).len(),
            24
        );
    }

    #[test]
    fn iter_subsets_yields_cartesian_product() {
        // Single rectangle with multi-range per dim:
        //   x: [0..2, 5..7], y: [0..3]
        // Expected subsets: (0..2, 0..3) and (5..7, 0..3).
        let mut per_dim: SmallVec<
            [Vec<Range<u64>>; 4],
        > = SmallVec::new();
        per_dim.push(vec![0..2, 5..7]);
        per_dim.push(vec![0..3]);
        let set = RectangleSet::from_per_dim(
            dims2(),
            shape_of(&[10, 10]),
            per_dim,
        );

        let subsets: Vec<_> =
            set.iter_subsets().collect();
        assert_eq!(subsets.len(), 2);
        let first_ranges = subsets[0].to_ranges();
        let second_ranges =
            subsets[1].to_ranges();
        assert_eq!(
            first_ranges,
            vec![0..2, 0..3]
        );
        assert_eq!(
            second_ranges,
            vec![5..7, 0..3]
        );

        // Total cell count = 2 * 3 + 2 * 3 = 12.
        assert_eq!(cell_count(&set), 12);
    }

    #[test]
    fn hyper_rectangle_difference_disjoint_returns_minuend()
     {
        let mut a_per: SmallVec<
            [Vec<Range<u64>>; 4],
        > = SmallVec::new();
        a_per.push(vec![0..3]);
        a_per.push(vec![0..3]);
        let a = HyperRectangle { per_dim: a_per };

        let mut b_per: SmallVec<
            [Vec<Range<u64>>; 4],
        > = SmallVec::new();
        b_per.push(vec![5..8]);
        b_per.push(vec![5..8]);
        let b = HyperRectangle { per_dim: b_per };

        let diff =
            hyper_rectangle_difference(&a, &b);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0].per_dim.len(), 2);
    }
}
