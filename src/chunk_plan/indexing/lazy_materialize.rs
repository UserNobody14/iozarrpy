//! Collection and materialization of lazy selections.
//!
//! This module provides functions to:
//! 1. Collect all resolution requests from a lazy selection
//! 2. Materialize a lazy selection into a concrete selection using a resolution cache

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use smallvec::SmallVec;

use super::lazy_selection::{
    LazyArraySelection, LazyDatasetSelection, LazyDimConstraint, LazyHyperRectangle,
};
use super::resolver_traits::{ResolutionCache, ResolutionError, ResolutionRequest};
use super::selection::{
    DataArraySelection, DatasetSelection, HyperRectangleSelection, RangeList, SetOperations,
};
use super::types::{IndexRange, ValueRange};

/// Collect all resolution requests from a lazy dataset selection.
///
/// Returns a deduplicated set of requests that need to be resolved.
pub(crate) fn collect_requests(selection: &LazyDatasetSelection) -> Vec<ResolutionRequest> {
    let mut requests = HashSet::new();

    match selection {
        LazyDatasetSelection::NoSelectionMade | LazyDatasetSelection::Empty => {}
        LazyDatasetSelection::Selection(sel) => {
            for (_, array_sel) in sel {
                collect_array_requests(array_sel, &mut requests);
            }
        }
    }

    requests.into_iter().collect()
}

fn collect_array_requests(selection: &LazyArraySelection, requests: &mut HashSet<ResolutionRequest>) {
    match selection {
        LazyArraySelection::Rectangles(rects) => {
            for rect in rects {
                collect_rectangle_requests(rect, requests);
            }
        }
        LazyArraySelection::Difference(a, b) | LazyArraySelection::Union(a, b) => {
            collect_array_requests(a, requests);
            collect_array_requests(b, requests);
        }
    }
}

fn collect_rectangle_requests(rect: &LazyHyperRectangle, requests: &mut HashSet<ResolutionRequest>) {
    for (dim, constraint) in rect.dims() {
        match constraint {
            LazyDimConstraint::Unresolved(vr) => {
                requests.insert(ResolutionRequest::new(dim.clone(), vr.clone()));
            }
            LazyDimConstraint::UnresolvedInterpolation(vr) => {
                requests.insert(ResolutionRequest::new(dim.clone(), (**vr).clone()));
            }
            LazyDimConstraint::UnresolvedInterpolationPoints(points) => {
                // For interpolation points, we need to create requests for each point's
                // bracketing lookups. We create two requests per point: <= v and >= v.
                for point in points.iter() {
                    // Request for <= v (to find left bracket)
                    let vr_max = ValueRange {
                        max: Some((point.clone(), super::types::BoundKind::Inclusive)),
                        ..Default::default()
                    };
                    requests.insert(ResolutionRequest::new(dim.clone(), vr_max));
                    
                    // Request for >= v (to find right bracket)
                    let vr_min = ValueRange {
                        min: Some((point.clone(), super::types::BoundKind::Inclusive)),
                        ..Default::default()
                    };
                    requests.insert(ResolutionRequest::new(dim.clone(), vr_min));
                }
            }
            _ => {}
        }
    }
}

/// Materialize a lazy dataset selection into a concrete selection.
///
/// Uses the provided cache to resolve value ranges to index ranges.
pub(crate) fn materialize(
    selection: &LazyDatasetSelection,
    cache: &dyn ResolutionCache,
) -> Result<DatasetSelection, ResolutionError> {
    match selection {
        LazyDatasetSelection::NoSelectionMade => Ok(DatasetSelection::NoSelectionMade),
        LazyDatasetSelection::Empty => Ok(DatasetSelection::Empty),
        LazyDatasetSelection::Selection(sel) => {
            let mut out = BTreeMap::new();
            for (var, array_sel) in sel {
                let materialized = materialize_array(array_sel, cache)?;
                if !materialized.is_empty() {
                    out.insert(var.clone(), materialized);
                }
            }
            if out.is_empty() {
                Ok(DatasetSelection::Empty)
            } else {
                Ok(DatasetSelection::Selection(out))
            }
        }
    }
}

fn materialize_array(
    selection: &LazyArraySelection,
    cache: &dyn ResolutionCache,
) -> Result<DataArraySelection, ResolutionError> {
    match selection {
        LazyArraySelection::Rectangles(rects) => {
            let mut out = Vec::with_capacity(rects.len());
            for rect in rects {
                let materialized = materialize_rectangle(rect, cache)?;
                if !materialized.is_empty() {
                    out.push(materialized);
                }
            }
            Ok(DataArraySelection(out))
        }
        LazyArraySelection::Difference(a, b) => {
            let a_mat = materialize_array(a, cache)?;
            let b_mat = materialize_array(b, cache)?;
            Ok(a_mat.difference(&b_mat))
        }
        LazyArraySelection::Union(a, b) => {
            let a_mat = materialize_array(a, cache)?;
            let b_mat = materialize_array(b, cache)?;
            Ok(a_mat.union(&b_mat))
        }
    }
}

fn materialize_rectangle(
    rect: &LazyHyperRectangle,
    cache: &dyn ResolutionCache,
) -> Result<HyperRectangleSelection, ResolutionError> {
    if rect.is_empty() {
        return Ok(HyperRectangleSelection::empty());
    }

    let mut result = HyperRectangleSelection::all();

    for (dim, constraint) in rect.dims() {
        let range_list = materialize_constraint(dim, constraint, cache)?;
        if range_list.is_empty() {
            return Ok(HyperRectangleSelection::empty());
        }
        if range_list != RangeList::all() {
            result = result.with_dim(dim.to_string(), range_list);
        }
    }

    Ok(result)
}

fn materialize_constraint(
    dim: &Arc<str>,
    constraint: &LazyDimConstraint,
    cache: &dyn ResolutionCache,
) -> Result<RangeList, ResolutionError> {
    match constraint {
        LazyDimConstraint::All => Ok(RangeList::all()),
        LazyDimConstraint::Empty => Ok(RangeList::empty()),
        LazyDimConstraint::Resolved(rl) => Ok(rl.clone()),
        LazyDimConstraint::Unresolved(vr) => {
            let request = ResolutionRequest::new(dim.clone(), vr.clone());
            match cache.get(&request) {
                Some(Some(idx_range)) => Ok(RangeList::from_index_range(idx_range)),
                Some(None) => {
                    // Resolution was attempted but couldn't determine a range
                    // (e.g., non-monotonic array) - return All conservatively
                    Ok(RangeList::all())
                }
                None => Err(ResolutionError::NotFound(request)),
            }
        }
        LazyDimConstraint::UnresolvedInterpolation(vr) => {
            // Interpolation needs bracketing indices - expand by 1 on each side.
            let request = ResolutionRequest::new(dim.clone(), (**vr).clone());
            match cache.get(&request) {
                Some(Some(idx_range)) => {
                    // Expand by 1 on each side for interpolation bracketing
                    let expanded = IndexRange {
                        start: idx_range.start.saturating_sub(1),
                        // Note: we don't know dim_len here, so we just add 1
                        // The caller should clamp this if needed
                        end_exclusive: idx_range.end_exclusive.saturating_add(1),
                    };
                    Ok(RangeList::from_index_range(expanded))
                }
                Some(None) => {
                    // Resolution was attempted but couldn't determine a range
                    Ok(RangeList::all())
                }
                None => Err(ResolutionError::NotFound(request)),
            }
        }
        LazyDimConstraint::UnresolvedInterpolationPoints(points) => {
            // For each point, we need to find bracketing indices and union them.
            // We look up the <= v and >= v requests from the cache.
            use super::selection::SetOperations;

            let mut result = RangeList::empty();

            for point in points.iter() {
                // Look up the <= v request (to find left bracket)
                let vr_max = ValueRange {
                    max: Some((point.clone(), super::types::BoundKind::Inclusive)),
                    ..Default::default()
                };
                let req_max = ResolutionRequest::new(dim.clone(), vr_max);

                // Look up the >= v request (to find right bracket)
                let vr_min = ValueRange {
                    min: Some((point.clone(), super::types::BoundKind::Inclusive)),
                    ..Default::default()
                };
                let req_min = ResolutionRequest::new(dim.clone(), vr_min);

                let left_end = match cache.get(&req_max) {
                    Some(Some(r)) => r.end_exclusive,
                    Some(None) => return Ok(RangeList::all()), // Can't resolve
                    None => return Err(ResolutionError::NotFound(req_max)),
                };

                let right_start = match cache.get(&req_min) {
                    Some(Some(r)) => r.start,
                    Some(None) => return Ok(RangeList::all()), // Can't resolve
                    None => return Err(ResolutionError::NotFound(req_min)),
                };

                // Compute bracketing indices with clamping
                // left_end is the exclusive end of the range <= v
                // right_start is the start of the range >= v
                //
                // For interpolation:
                // - left_idx is the last index <= v (i.e., left_end - 1, clamped to 0)
                // - right_idx is the first index >= v (i.e., right_start)
                //
                // We need to include both for bracketing.
                // Clamp to valid range: we don't know dim_len here, but the resolver
                // should have returned valid ranges. If left_end == 0, clamp left_idx to 0.
                let left_idx = if left_end == 0 { 0 } else { left_end - 1 };
                let right_idx = right_start;

                // The bracket spans from min(left_idx, right_idx) to max(left_idx, right_idx) + 1
                let start = left_idx.min(right_idx);
                let end_exclusive = left_idx.max(right_idx).saturating_add(1);

                // Expand by 1 on each side to be safe near chunk boundaries
                let expanded = IndexRange {
                    start: start.saturating_sub(1),
                    end_exclusive: end_exclusive.saturating_add(1),
                };

                result = result.union(&RangeList::from_index_range(expanded));
            }

            Ok(result)
        }
    }
}

/// Helper to materialize with dimension length clamping.
#[allow(dead_code)]
pub(crate) fn materialize_with_dim_lengths(
    selection: &LazyDatasetSelection,
    cache: &dyn ResolutionCache,
    dim_lengths: &BTreeMap<String, u64>,
) -> Result<DatasetSelection, ResolutionError> {
    let mut base = materialize(selection, cache)?;

    // Clamp all range lists to their dimension lengths
    if let DatasetSelection::Selection(ref mut sel) = base {
        for array_sel in sel.values_mut() {
            for rect in &mut array_sel.0 {
                // Collect updates first to avoid borrow conflict
                let updates: Vec<_> = rect
                    .dims()
                    .filter_map(|(dim_name, range_list)| {
                        dim_lengths.get(dim_name).map(|&len| {
                            (dim_name.to_string(), range_list.clamp_to_len(len))
                        })
                    })
                    .collect();

                // Apply updates
                let mut new_rect = rect.clone();
                for (dim_name, clamped) in updates {
                    new_rect = new_rect.with_dim(dim_name, clamped);
                }
                *rect = new_rect;
            }
        }
    }

    Ok(base)
}

// ============================================================================
// Collection with context for index-only dimensions
// ============================================================================

use crate::chunk_plan::indexing::index_ranges::index_range_for_index_dim;
use crate::meta::ZarrDatasetMeta;

/// Collect requests and handle index-only dimensions.
///
/// For dimensions that don't have coordinate arrays (index-only dims),
/// we can resolve them immediately using the dimension length.
pub(crate) fn collect_requests_with_meta(
    selection: &LazyDatasetSelection,
    meta: &ZarrDatasetMeta,
    dim_lengths: &[u64],
    dims: &[String],
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
    dims: &[String],
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
    dims: &[String],
    requests: &mut HashSet<ResolutionRequest>,
    immediate_cache: &mut super::resolver_traits::HashMapCache,
) {
    for (dim, constraint) in rect.dims() {
        match constraint {
            LazyDimConstraint::Unresolved(vr) => {
                let request = ResolutionRequest::new(dim.clone(), vr.clone());

                // Check if this is an index-only dimension (no coordinate array)
                if meta.arrays.get(dim.as_ref()).is_none() {
                    // Index-only dimension - resolve immediately
                    if let Some(dim_idx) = dims.iter().position(|d| d == dim.as_ref()) {
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
                let request = ResolutionRequest::new(dim.clone(), vr.clone());

                // Check if this is an index-only dimension (no coordinate array)
                if meta.arrays.get(dim.as_ref()).is_none() {
                    // Index-only dimension - resolve immediately
                    if let Some(dim_idx) = dims.iter().position(|d| d == dim.as_ref()) {
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
                    if meta.arrays.get(dim.as_ref()).is_none() {
                        if let Some(dim_idx) = dims.iter().position(|d| d == dim.as_ref()) {
                            if let Some(&dim_len) = dim_lengths.get(dim_idx) {
                                // For index-only dims, resolve immediately
                                if let Some(r) = index_range_for_index_dim(&vr_max, dim_len) {
                                    immediate_cache.insert(
                                        ResolutionRequest::new(dim.clone(), vr_max.clone()),
                                        Some(r),
                                    );
                                }
                                if let Some(r) = index_range_for_index_dim(&vr_min, dim_len) {
                                    immediate_cache.insert(
                                        ResolutionRequest::new(dim.clone(), vr_min.clone()),
                                        Some(r),
                                    );
                                }
                                continue;
                            }
                        }
                    }

                    requests.insert(ResolutionRequest::new(dim.clone(), vr_max));
                    requests.insert(ResolutionRequest::new(dim.clone(), vr_min));
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
    pub(crate) fn new(primary: &'a dyn ResolutionCache, secondary: &'a dyn ResolutionCache) -> Self {
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
