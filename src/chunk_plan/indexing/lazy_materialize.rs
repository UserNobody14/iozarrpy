//! Collection and materialization of lazy selections.
//!
//! This module provides functions to:
//! 1. Collect all resolution requests from a lazy selection
//! 2. Materialize a lazy selection into a concrete selection using a resolution cache

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use zarrs::array_subset::ArraySubset;

use super::lazy_selection::{
    LazyArraySelection, LazyDatasetSelection, LazyDimConstraint, LazyHyperRectangle,
};
use super::resolver_traits::{ResolutionCache, ResolutionError, ResolutionRequest};
use super::selection::{
    DataArraySelection, DatasetSelection, Emptyable,
    SetOperations,
};
use super::types::{IndexRange, ValueRange};

/// Materialize a lazy dataset selection into a concrete selection.
///
/// Uses the provided cache to resolve value ranges to index ranges.
pub(crate) fn materialize(
    selection: &LazyDatasetSelection,
    meta: &ZarrDatasetMeta,
    cache: &dyn ResolutionCache,
) -> Result<DatasetSelection, ResolutionError> {
    match selection {
        LazyDatasetSelection::NoSelectionMade => Ok(DatasetSelection::NoSelectionMade),
        LazyDatasetSelection::Empty => Ok(DatasetSelection::Empty),
        LazyDatasetSelection::Selection(sel) => {
            let mut out = BTreeMap::new();
            for (var, array_sel) in sel {
                let Some(var_meta) = meta.arrays.get(var) else {
                    return Err(ResolutionError::Unresolvable(format!(
                        "variable '{}' not found",
                        var
                    )));
                };
                let materialized =
                    materialize_array(array_sel, &var_meta.dims, var_meta.shape.clone(), cache)?;
                if !materialized.is_empty() {
                    out.insert(var.clone(), materialized);
                }
            }
            if out.is_empty() {
                Ok(DatasetSelection::Empty)
            } else {
                Ok(DatasetSelection::Selection(out.into()))
            }
        }
    }
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
                let materialized = materialize_rectangle(rect, dims, shape.clone(), cache)?;
                out = out.union(&materialized);
            }

            Ok(DataArraySelection::from_subsets(dims, out))
        }
        LazyArraySelection::Difference(a, b) => {
            let a_mat = materialize_array(a, dims, shape.clone(), cache)?;
            let b_mat = materialize_array(b, dims, shape.clone(), cache)?;
            Ok(a_mat.difference(&b_mat))
        }
        LazyArraySelection::Union(a, b) => {
            let a_mat = materialize_array(a, dims, shape.clone(), cache)?;
            let b_mat = materialize_array(b, dims, shape.clone(), cache)?;
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
    let mut current_subsets: Vec<Vec<Range<u64>>> = vec![
        (0..dims.len()).map(|i| 0..shape[i]).collect()
    ];

    for (dim, constraint) in rect.dims() {
        let dim_idx_option = dims.iter().position(|d| d == dim);
        if let Some(dim_idx) = dim_idx_option {
            let range_list = materialize_constraint_multi(dim, 0..shape[dim_idx], constraint, cache)?;
            if range_list.is_empty() {
                return Ok(ArraySubsetList::empty());
            }
            
            // For each (range, existing subset) pair, create a new subset
            let mut new_subsets = Vec::new();
            for subset in current_subsets.iter() {
                for range in range_list.iter() {
                    if !range.is_empty() {
                        let mut new_subset = subset.clone();
                        new_subset[dim_idx] = range.clone();
                        new_subsets.push(new_subset);
                    }
                }
            }
            
            if new_subsets.is_empty() {
                return Ok(ArraySubsetList::empty());
            }
            current_subsets = new_subsets;
        }
    }

    let mut out_list = ArraySubsetList::new();
    for subset in current_subsets {
        out_list.push(ArraySubset::new_with_ranges(&subset));
    }
    Ok(out_list)
}

/// Materialize a constraint returning potentially multiple ranges.
/// For interpolation points, each point may produce a disjoint range.
fn materialize_constraint_multi(
    dim: &crate::IStr,
    dim_range: Range<u64>,
    constraint: &LazyDimConstraint,
    cache: &dyn ResolutionCache,
) -> Result<Vec<Range<u64>>, ResolutionError> {
    match constraint {
        LazyDimConstraint::UnresolvedInterpolationPoints(points) => {
            // For each point, compute bracketing indices and collect disjoint ranges
            let mut ranges = Vec::new();

            for point in points.iter() {
                // Look up the <= v request (to find left bracket)
                let vr_max = ValueRange {
                    max: Some((point.clone(), super::types::BoundKind::Inclusive)),
                    ..Default::default()
                };
                let req_max = ResolutionRequest::new(dim.as_ref(), vr_max);

                // Look up the >= v request (to find right bracket)
                let vr_min = ValueRange {
                    min: Some((point.clone(), super::types::BoundKind::Inclusive)),
                    ..Default::default()
                };
                let req_min = ResolutionRequest::new(dim.as_ref(), vr_min);

                let left_end = match cache.get(&req_max) {
                    Some(Some(r)) => r.end_exclusive,
                    Some(None) => return Ok(vec![dim_range]), // Can't resolve
                    None => return Err(ResolutionError::NotFound(req_max)),
                };

                let right_start = match cache.get(&req_min) {
                    Some(Some(r)) => r.start,
                    Some(None) => return Ok(vec![dim_range]), // Can't resolve
                    None => return Err(ResolutionError::NotFound(req_min)),
                };

                // Compute bracketing indices
                let left_idx = if left_end == 0 { 0 } else { left_end - 1 };
                let right_idx = right_start;

                // The bracket spans from min to max + 1
                let start = left_idx.min(right_idx);
                let end_exclusive = left_idx.max(right_idx).saturating_add(1);

                // Expand by 1 on each side to be safe near chunk boundaries
                let expanded_start = start.saturating_sub(1);
                let expanded_end = end_exclusive.saturating_add(1).min(dim_range.end);

                ranges.push(expanded_start..expanded_end);
            }

            // Merge adjacent/overlapping ranges for efficiency
            ranges = merge_ranges(ranges);
            
            Ok(ranges)
        }
        // For all other constraint types, delegate to the single-range version
        _ => {
            let single = materialize_constraint(dim, dim_range, constraint, cache)?;
            Ok(vec![single])
        }
    }
}

/// Merge overlapping or adjacent ranges into a minimal set
fn merge_ranges(mut ranges: Vec<Range<u64>>) -> Vec<Range<u64>> {
    if ranges.is_empty() {
        return ranges;
    }
    
    // Sort by start
    ranges.sort_by_key(|r| r.start);
    
    let mut result = Vec::new();
    let mut current = ranges[0].clone();
    
    for range in ranges.into_iter().skip(1) {
        if range.start <= current.end {
            // Overlapping or adjacent - extend current
            current.end = current.end.max(range.end);
        } else {
            // Disjoint - save current and start new
            result.push(current);
            current = range;
        }
    }
    result.push(current);
    
    result
}

fn materialize_constraint(
    dim: &crate::IStr,
    dim_range: Range<u64>,
    constraint: &LazyDimConstraint,
    cache: &dyn ResolutionCache,
) -> Result<Range<u64>, ResolutionError> {
    match constraint {
        LazyDimConstraint::All => Ok(dim_range),
        LazyDimConstraint::Empty => Ok(0..0),
        LazyDimConstraint::Resolved(rl) => Ok(rl.clone()),
        LazyDimConstraint::Unresolved(vr) => {
            let request = ResolutionRequest::new(dim.as_ref(), vr.clone());
            match cache.get(&request) {
                Some(Some(idx_range)) => Ok(
                    idx_range.start..idx_range.end_exclusive
                ),
                Some(None) => {
                    // Resolution was attempted but couldn't determine a range
                    // (e.g., non-monotonic array) - return All conservatively
                    Ok(dim_range)
                }
                None => Err(ResolutionError::NotFound(request)),
            }
        }
        LazyDimConstraint::UnresolvedInterpolation(vr) => {
            // Interpolation needs bracketing indices - expand by 1 on each side.
            let request = ResolutionRequest::new(dim.as_ref(), (**vr).clone());
            match cache.get(&request) {
                Some(Some(idx_range)) => {
                    // Expand by 1 on each side for interpolation bracketing
                    let expanded = IndexRange {
                        start: idx_range.start.saturating_sub(1),
                        // Note: we don't know dim_len here, so we just add 1
                        // The caller should clamp this if needed
                        end_exclusive: idx_range.end_exclusive.saturating_add(1),
                    };
                    Ok(expanded.start..expanded.end_exclusive)
                }
                Some(None) => {
                    // Resolution was attempted but couldn't determine a range
                    Ok(dim_range)
                }
                None => Err(ResolutionError::NotFound(request)),
            }
        }
        LazyDimConstraint::UnresolvedInterpolationPoints(_) => {
            // This case is handled by materialize_constraint_multi, so if we reach here
            // it means we were called through the wrong path. Return conservative dim_range.
            Ok(dim_range)
        }
    }
}

// ============================================================================
// Collection with context for index-only dimensions
// ============================================================================

use crate::chunk_plan::indexing::index_ranges::index_range_for_index_dim;
use crate::chunk_plan::indexing::selection::ArraySubsetList;
use crate::meta::ZarrDatasetMeta;

/// Collect requests and handle index-only dimensions.
///
/// For dimensions that don't have coordinate arrays (index-only dims),
/// we can resolve them immediately using the dimension length.
pub(crate) fn collect_requests_with_meta(
    selection: &LazyDatasetSelection,
    meta: &ZarrDatasetMeta,
    dim_lengths: &[u64],
    dims: &[crate::IStr],
) -> (Vec<ResolutionRequest>, super::resolver_traits::HashMapCache) {
    let mut requests = HashSet::new();
    let mut immediate_cache = super::resolver_traits::HashMapCache::new();

    match selection {
        LazyDatasetSelection::NoSelectionMade | LazyDatasetSelection::Empty => {}
        LazyDatasetSelection::Selection(sel) => {
            for (_, array_sel) in sel {
                collect_array_requests_with_meta(
                    array_sel,
                    meta,
                    dim_lengths,
                    dims,
                    &mut requests,
                    &mut immediate_cache,
                );
            }
        }
    }

    (requests.into_iter().collect(), immediate_cache)
}

fn collect_array_requests_with_meta(
    selection: &LazyArraySelection,
    meta: &ZarrDatasetMeta,
    dim_lengths: &[u64],
    dims: &[crate::IStr],
    requests: &mut HashSet<ResolutionRequest>,
    immediate_cache: &mut super::resolver_traits::HashMapCache,
) {
    match selection {
        LazyArraySelection::Rectangles(rects) => {
            for rect in rects {
                collect_rectangle_requests_with_meta(
                    rect,
                    meta,
                    dim_lengths,
                    dims,
                    requests,
                    immediate_cache,
                );
            }
        }
        LazyArraySelection::Difference(a, b) | LazyArraySelection::Union(a, b) => {
            collect_array_requests_with_meta(a, meta, dim_lengths, dims, requests, immediate_cache);
            collect_array_requests_with_meta(b, meta, dim_lengths, dims, requests, immediate_cache);
        }
    }
}

fn collect_rectangle_requests_with_meta(
    rect: &LazyHyperRectangle,
    meta: &ZarrDatasetMeta,
    dim_lengths: &[u64],
    dims: &[crate::IStr],
    requests: &mut HashSet<ResolutionRequest>,
    immediate_cache: &mut super::resolver_traits::HashMapCache,
) {
    for (dim, constraint) in rect.dims() {
        match constraint {
            LazyDimConstraint::Unresolved(vr) => {
                let request = ResolutionRequest::new(dim.as_ref(), vr.clone());

                // Check if this is an index-only dimension (no coordinate array)
                if meta.arrays.get(dim).is_none() {
                    // Index-only dimension - resolve immediately
                    if let Some(dim_idx) = dims.iter().position(|d| d == dim) {
                        if let Some(&dim_len) = dim_lengths.get(dim_idx) {
                            if let Some(idx_range) = index_range_for_index_dim(vr, dim_len) {
                                immediate_cache.insert(request.clone(), Some(idx_range));
                                continue;
                            }
                        }
                    }
                    // Couldn't resolve - add to requests (will fail later)
                    requests.insert(request);
                } else {
                    // Has coordinate array - needs resolution
                    requests.insert(request);
                }
            }
            LazyDimConstraint::UnresolvedInterpolation(vr_arc) => {
                let vr = (**vr_arc).clone();
                let request = ResolutionRequest::new(dim.as_ref(), vr.clone());

                // Check if this is an index-only dimension (no coordinate array)
                if meta.arrays.get(dim).is_none() {
                    // Index-only dimension - resolve immediately
                    if let Some(dim_idx) = dims.iter().position(|d| d == dim) {
                        if let Some(&dim_len) = dim_lengths.get(dim_idx) {
                            if let Some(idx_range) = index_range_for_index_dim(&vr, dim_len) {
                                immediate_cache.insert(request.clone(), Some(idx_range));
                                continue;
                            }
                        }
                    }
                    // Couldn't resolve - add to requests (will fail later)
                    requests.insert(request);
                } else {
                    // Has coordinate array - needs resolution
                    requests.insert(request);
                }
            }
            LazyDimConstraint::UnresolvedInterpolationPoints(points) => {
                // For interpolation points, create bracketing requests
                for point in points.iter() {
                    let vr_max = ValueRange {
                        max: Some((point.clone(), super::types::BoundKind::Inclusive)),
                        ..Default::default()
                    };
                    let vr_min = ValueRange {
                        min: Some((point.clone(), super::types::BoundKind::Inclusive)),
                        ..Default::default()
                    };

                    // Check if this is an index-only dimension
                    if meta.arrays.get(dim).is_none() {
                        if let Some(dim_idx) = dims.iter().position(|d| d == dim) {
                            if let Some(&dim_len) = dim_lengths.get(dim_idx) {
                                // For index-only dims, resolve immediately
                                if let Some(r) = index_range_for_index_dim(&vr_max, dim_len) {
                                    immediate_cache.insert(
                                        ResolutionRequest::new(dim.as_ref(), vr_max.clone()),
                                        Some(r),
                                    );
                                }
                                if let Some(r) = index_range_for_index_dim(&vr_min, dim_len) {
                                    immediate_cache.insert(
                                        ResolutionRequest::new(dim.as_ref(), vr_min.clone()),
                                        Some(r),
                                    );
                                }
                                continue;
                            }
                        }
                    }

                    requests.insert(ResolutionRequest::new(dim.as_ref(), vr_max));
                    requests.insert(ResolutionRequest::new(dim.as_ref(), vr_min));
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergedCache").finish()
    }
}

impl ResolutionCache for MergedCache<'_> {
    fn get(&self, request: &ResolutionRequest) -> Option<Option<super::types::IndexRange>> {
        self.primary
            .get(request)
            .or_else(|| self.secondary.get(request))
    }
}
