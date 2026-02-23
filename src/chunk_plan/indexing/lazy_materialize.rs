//! Collection and materialization of lazy selections.
//!
//! This module provides functions to:
//! 1. Collect all resolution requests from a lazy selection
//! 2. Materialize a lazy selection into a concrete selection using a resolution cache

use std::collections::{
    BTreeMap, HashMap, HashSet,
};
use std::ops::Range;
use std::sync::Arc;

use zarrs::array::ArraySubset;

use super::grouped_selection::GroupedSelection;
use super::lazy_selection::{
    LazyArraySelection, LazyDatasetSelection,
    LazyDimConstraint, LazyHyperRectangle,
};
use super::resolver_traits::{
    ResolutionCache, ResolutionError,
    ResolutionRequest,
};
use super::selection::{
    DataArraySelection, DatasetSelection,
    Emptyable, SetOperations,
};
use super::types::DimSignature;
use crate::meta::ZarrMeta;

/// Materialize a lazy dataset selection into a concrete selection.
///
/// Uses the provided cache to resolve value ranges to index ranges.
pub(crate) fn materialize(
    selection: &LazyDatasetSelection,
    meta: &ZarrMeta,
    cache: &dyn ResolutionCache,
) -> Result<DatasetSelection, ResolutionError> {
    match selection {
        LazyDatasetSelection::NoSelectionMade => {
            Ok(DatasetSelection::NoSelectionMade)
        }
        LazyDatasetSelection::Empty => {
            Ok(DatasetSelection::Empty)
        }
        LazyDatasetSelection::Selection(sel) => {
            // Materialize once per dimension signature
            let mut by_dims: BTreeMap<
                Arc<DimSignature>,
                DataArraySelection,
            > = BTreeMap::new();

            for (sig, array_sel) in
                sel.by_signature()
            {
                // Use the signature's dims to determine shape
                let dims = sig.dims();
                let shape =
                    dims_to_shape(dims, meta)?;

                let materialized =
                    materialize_array(
                        array_sel,
                        &dims
                            .iter()
                            .cloned()
                            .collect(),
                        shape,
                        cache,
                    )?;
                if !materialized.is_empty() {
                    // We need an Arc for the key - find it from the lazy selection
                    if let Some(arc_sig) =
                        sel.by_dims().keys().find(
                            |k| k.as_ref() == sig,
                        )
                    {
                        by_dims.insert(
                            arc_sig.clone(),
                            materialized,
                        );
                    }
                }
            }

            if by_dims.is_empty() {
                Ok(DatasetSelection::Empty)
            } else {
                // Preserve the var_to_sig mapping
                let var_to_sig =
                    sel.var_to_sig().clone();
                let grouped =
                    GroupedSelection::from_parts(
                        by_dims, var_to_sig,
                    );
                Ok(DatasetSelection::Selection(
                    grouped,
                ))
            }
        }
    }
}

/// Get shape for a dimension signature by looking up the first matching array.
fn dims_to_shape(
    dims: &[crate::IStr],
    meta: &ZarrMeta,
) -> Result<Arc<[u64]>, ResolutionError> {
    let dim_shape_reduced: Option<Vec<u64>> =
        dims.iter()
            .map(|dim| {
                meta.dim_analysis
                    .dim_lengths
                    .get(dim)
                    .copied()
            })
            .collect();

    if let Some(reduced) = dim_shape_reduced {
        return Ok(reduced.into());
    }
    // Fallback: construct shape from coordinate arrays
    let mut shape =
        Vec::with_capacity(dims.len());
    for dim in dims {
        if let Some(coord_array) =
            meta.array_by_path(dim)
        {
            if let Some(&len) =
                coord_array.shape.first()
            {
                shape.push(len);
            } else {
                return Err(
                    ResolutionError::Unresolvable(
                        format!(
                            "dimension '{}' has no shape",
                            dim
                        ),
                    ),
                );
            }
        } else {
            return Err(
                ResolutionError::Unresolvable(
                    format!(
                        "cannot determine shape for dimension '{}'",
                        dim
                    ),
                ),
            );
        }
    }

    Ok(shape.into())
}

fn materialize_array(
    selection: &LazyArraySelection,
    dims: &smallvec::SmallVec<[crate::IStr; 4]>,
    shape: std::sync::Arc<[u64]>,
    cache: &dyn ResolutionCache,
) -> Result<DataArraySelection, ResolutionError> {
    match selection {
        LazyArraySelection::Rectangles(rects) => {
            let mut out = ArraySubsetList::new();
            for rect in rects {
                let materialized =
                    materialize_rectangle(
                        rect,
                        dims,
                        shape.clone(),
                        cache,
                    )?;
                out = out.union(&materialized);
            }

            Ok(DataArraySelection::from_subsets(
                dims, out,
            ))
        }
        LazyArraySelection::Difference(a, b) => {
            let a_mat = materialize_array(
                a,
                dims,
                shape.clone(),
                cache,
            )?;
            let b_mat = materialize_array(
                b,
                dims,
                shape.clone(),
                cache,
            )?;
            Ok(a_mat.difference(&b_mat))
        }
        LazyArraySelection::Union(a, b) => {
            let a_mat = materialize_array(
                a,
                dims,
                shape.clone(),
                cache,
            )?;
            let b_mat = materialize_array(
                b,
                dims,
                shape.clone(),
                cache,
            )?;
            Ok(a_mat.union(&b_mat))
        }
    }
}

fn materialize_rectangle(
    rect: &LazyHyperRectangle,
    dims: &smallvec::SmallVec<[crate::IStr; 4]>,
    shape: std::sync::Arc<[u64]>,
    cache: &dyn ResolutionCache,
) -> Result<ArraySubsetList, ResolutionError> {
    if rect.is_empty() {
        return Ok(ArraySubsetList::empty());
    }

    // Start with a single subset representing "all" for each dimension
    // Each element in current_subsets is a vec of ranges, one per dimension
    let mut current_subsets: Vec<
        Vec<Range<u64>>,
    > = vec![
        (0..dims.len())
            .map(|i| 0..shape[i])
            .collect(),
    ];

    for (dim, constraint) in rect.dims() {
        let dim_idx_option =
            dims.iter().position(|d| d == dim);
        if let Some(dim_idx) = dim_idx_option {
            let range_list =
                materialize_constraint(
                    dim,
                    0..shape[dim_idx],
                    constraint,
                    cache,
                )?;
            if range_list.is_empty() {
                return Ok(
                    ArraySubsetList::empty(),
                );
            }

            // For each (range, existing subset) pair, create a new subset
            let mut new_subsets = Vec::new();
            for subset in current_subsets.iter() {
                for range in range_list.iter() {
                    if !range.is_empty() {
                        let mut new_subset =
                            subset.clone();
                        new_subset[dim_idx] =
                            range.clone();
                        new_subsets
                            .push(new_subset);
                    }
                }
            }

            if new_subsets.is_empty() {
                return Ok(
                    ArraySubsetList::empty(),
                );
            }
            current_subsets = new_subsets;
        }
    }

    let mut out_list = ArraySubsetList::new();
    for subset in current_subsets {
        out_list.push(
            ArraySubset::new_with_ranges(&subset),
        );
    }
    Ok(out_list)
}

/// Merge overlapping or adjacent ranges into a minimal set.
fn merge_ranges(
    mut ranges: Vec<Range<u64>>,
) -> Vec<Range<u64>> {
    if ranges.len() <= 1 {
        return ranges;
    }
    ranges.sort_unstable_by_key(|r| r.start);
    let mut result =
        Vec::with_capacity(ranges.len());
    let mut current = ranges[0].clone();
    for range in ranges.into_iter().skip(1) {
        if range.start <= current.end {
            current.end =
                current.end.max(range.end);
        } else {
            result.push(current);
            current = range;
        }
    }
    result.push(current);
    result
}

/// Look up a resolution request in the cache, returning a fallback on non-monotonic arrays.
fn resolve_from_cache(
    dim: &crate::IStr,
    vr: &ValueRangePresent,
    dim_range: &Range<u64>,
    cache: &dyn ResolutionCache,
) -> Result<Range<u64>, ResolutionError> {
    let request =
        ResolutionRequest::new(dim, vr.clone());
    match cache.get(&request) {
        Some(Some(idx_range)) => {
            Ok(idx_range.clone())
        }
        Some(None) => Ok(dim_range.clone()),
        None => Err(ResolutionError::NotFound(
            request,
        )),
    }
}

/// Materialize a constraint to one or more index ranges.
fn materialize_constraint(
    dim: &crate::IStr,
    dim_range: Range<u64>,
    constraint: &LazyDimConstraint,
    cache: &dyn ResolutionCache,
) -> Result<Vec<Range<u64>>, ResolutionError> {
    match constraint {
        LazyDimConstraint::All => Ok(vec![dim_range]),
        LazyDimConstraint::Empty => Ok(vec![]),
        LazyDimConstraint::Resolved(rl) => Ok(vec![rl.clone()]),
        LazyDimConstraint::Unresolved(vr) => {
            Ok(vec![resolve_from_cache(dim, vr, &dim_range, cache)?])
        }
        LazyDimConstraint::InterpolationRange(vr) => {
            let r = resolve_from_cache(dim, vr, &dim_range, cache)?;
            let start = r.start.saturating_sub(1);
            let end = r.end.saturating_add(1);
            Ok(vec![start..end])
        }
        LazyDimConstraint::InterpolationPoints(points) => {
            let mut ranges = Vec::with_capacity(points.len());
            for point in points.iter() {
                let vr_max = ValueRangePresent::from_max_exclusive(point.clone());
                let req_max = ResolutionRequest::new(dim, vr_max);
                let vr_min = ValueRangePresent::from_min_inclusive(point.clone());
                let req_min = ResolutionRequest::new(dim, vr_min);

                let left_end = match cache.get(&req_max) {
                    Some(Some(r)) => r.end,
                    Some(None) => return Ok(vec![dim_range]),
                    None => return Err(ResolutionError::NotFound(req_max)),
                };
                let right_start = match cache.get(&req_min) {
                    Some(Some(r)) => r.start,
                    Some(None) => return Ok(vec![dim_range]),
                    None => return Err(ResolutionError::NotFound(req_min)),
                };

                let left_idx = left_end.saturating_sub(1);
                let right_idx = right_start;
                let start = left_idx.min(right_idx);
                let end_exclusive = (left_idx.max(right_idx) + 1).min(dim_range.end);
                ranges.push(start..end_exclusive);
            }
            Ok(merge_ranges(ranges))
        }
    }
}

// ============================================================================
// Collection with context for index-only dimensions
// ============================================================================

use crate::chunk_plan::indexing::selection::ArraySubsetList;
use crate::chunk_plan::indexing::types::ValueRangePresent;

/// Collect requests and handle index-only dimensions.
///
/// For dimensions that don't have coordinate arrays (index-only dims),
/// we can resolve them immediately using the dimension length.
pub(crate) fn collect_requests_with_meta(
    selection: &LazyDatasetSelection,
    meta: &ZarrMeta,
    dim_lengths: &[u64],
    dims: &[crate::IStr],
) -> (
    Vec<ResolutionRequest>,
    super::resolver_traits::HashMapCache,
) {
    let mut requests = HashSet::new();
    let mut immediate_cache =
        super::resolver_traits::HashMapCache::new(
        );
    let dim_index: HashMap<&crate::IStr, usize> =
        dims.iter()
            .enumerate()
            .map(|(i, d)| (d, i))
            .collect();

    if let LazyDatasetSelection::Selection(sel) =
        selection
    {
        let mut ctx = CollectCtx {
            meta,
            dim_index: &dim_index,
            dim_lengths,
            requests: &mut requests,
            immediate_cache: &mut immediate_cache,
        };
        for (_, array_sel) in sel.by_signature() {
            collect_array_requests(
                array_sel, &mut ctx,
            );
        }
    }

    (
        requests.into_iter().collect(),
        immediate_cache,
    )
}

/// Try to resolve a value range immediately for an index-only dimension (no coordinate array).
fn try_resolve_index_only(
    dim: &crate::IStr,
    meta: &ZarrMeta,
    dim_index: &HashMap<&crate::IStr, usize>,
    dim_lengths: &[u64],
    vr: &ValueRangePresent,
) -> Option<std::ops::Range<u64>> {
    if meta.array_by_path_contains(dim) {
        return None;
    }
    let &idx = dim_index.get(dim)?;
    let &dim_len = dim_lengths.get(idx)?;
    vr.index_range_for_index_dim(dim_len)
}

/// Try to resolve or enqueue a single value range request.
fn resolve_or_enqueue(
    dim: &crate::IStr,
    vr: &ValueRangePresent,
    meta: &ZarrMeta,
    dim_index: &HashMap<&crate::IStr, usize>,
    dim_lengths: &[u64],
    requests: &mut HashSet<ResolutionRequest>,
    immediate_cache: &mut super::resolver_traits::HashMapCache,
) {
    let request =
        ResolutionRequest::new(dim, vr.clone());
    if let Some(idx_range) =
        try_resolve_index_only(
            dim,
            meta,
            dim_index,
            dim_lengths,
            vr,
        )
    {
        immediate_cache
            .insert(request, Some(idx_range));
    } else {
        requests.insert(request);
    }
}

struct CollectCtx<'a> {
    meta: &'a ZarrMeta,
    dim_index: &'a HashMap<&'a crate::IStr, usize>,
    dim_lengths: &'a [u64],
    requests: &'a mut HashSet<ResolutionRequest>,
    immediate_cache: &'a mut super::resolver_traits::HashMapCache,
}

fn collect_array_requests(
    selection: &LazyArraySelection,
    ctx: &mut CollectCtx<'_>,
) {
    match selection {
        LazyArraySelection::Rectangles(rects) => {
            for rect in rects {
                collect_rectangle_requests(
                    rect, ctx,
                );
            }
        }
        LazyArraySelection::Difference(a, b)
        | LazyArraySelection::Union(a, b) => {
            collect_array_requests(a, ctx);
            collect_array_requests(b, ctx);
        }
    }
}

fn collect_rectangle_requests(
    rect: &LazyHyperRectangle,
    ctx: &mut CollectCtx<'_>,
) {
    for (dim, constraint) in rect.dims() {
        match constraint {
            LazyDimConstraint::Unresolved(vr)
            | LazyDimConstraint::InterpolationRange(vr) => {
                resolve_or_enqueue(
                    dim, vr, ctx.meta, ctx.dim_index, ctx.dim_lengths,
                    ctx.requests, ctx.immediate_cache,
                );
            }
            LazyDimConstraint::InterpolationPoints(points) => {
                for point in points.iter() {
                    let vr_max = ValueRangePresent::from_max_exclusive(point.clone());
                    let vr_min = ValueRangePresent::from_min_inclusive(point.clone());
                    resolve_or_enqueue(
                        dim, &vr_max, ctx.meta, ctx.dim_index, ctx.dim_lengths,
                        ctx.requests, ctx.immediate_cache,
                    );
                    resolve_or_enqueue(
                        dim, &vr_min, ctx.meta, ctx.dim_index, ctx.dim_lengths,
                        ctx.requests, ctx.immediate_cache,
                    );
                }
            }
            _ => {}
        }
    }
}

/// Merge two resolution caches.
///
/// Creates a combined cache that first checks the primary cache, then falls back
/// to the secondary.
pub(crate) struct MergedCache<'a> {
    primary: &'a dyn ResolutionCache,
    secondary: &'a dyn ResolutionCache,
}

impl<'a> MergedCache<'a> {
    pub(crate) fn new(
        primary: &'a dyn ResolutionCache,
        secondary: &'a dyn ResolutionCache,
    ) -> Self {
        Self { primary, secondary }
    }
}

impl std::fmt::Debug for MergedCache<'_> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("MergedCache").finish()
    }
}

impl ResolutionCache for MergedCache<'_> {
    fn get(
        &self,
        request: &ResolutionRequest,
    ) -> Option<Option<Range<u64>>> {
        self.primary.get(request).or_else(|| {
            self.secondary.get(request)
        })
    }

    fn insert(
        &mut self,

        #[allow(unused_variables)]
        request: ResolutionRequest,
        #[allow(unused_variables)] result: Option<
            Range<u64>,
        >,
    ) {
        panic!(
            "MergedCache does not support insert"
        );
    }
}
