use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use smallvec::SmallVec;
use zarrs::array::ChunkGrid;

use crate::chunk_plan::indexing::selection::ArraySubsetList;
use crate::chunk_plan::indexing::types::ChunkGridSignature;
use crate::errors::{
    BackendError, BackendResult,
};
use crate::shared::{IStr, IntoIStr};
use snafu::prelude::*;

// =============================================================================
// Chunk Subset
// =============================================================================

/// Per-chunk local bounding box: ranges in chunk-local coordinates.
///
/// Represents the portion of a chunk that intersects the user's selection.
/// Used to constrain the `KeepMask` so only relevant elements are processed.
#[derive(Debug, Clone)]
pub struct ChunkSubset {
    pub(crate) ranges: SmallVec<[Range<u64>; 4]>,
}

impl ChunkSubset {
    fn is_full_chunk(
        &self,
        chunk_shape: &[u64],
    ) -> bool {
        self.ranges.iter().zip(chunk_shape).all(
            |(r, &s)| r.start == 0 && r.end >= s,
        )
    }
}

/// Compute the chunk-local subset for a given chunk index.
///
/// Intersects every `ArraySubset` in the plan with this chunk's extent,
/// then returns the bounding box of all intersections in chunk-local
/// coordinates. Returns `None` when the bbox covers the full chunk.
fn compute_chunk_subset(
    chunk_idx: &[u64],
    chunk_shape: &[u64],
    array_shape: &[u64],
    subsets: &ArraySubsetList,
) -> Option<ChunkSubset> {
    let ndim = chunk_idx.len();

    let chunk_start: SmallVec<[u64; 4]> =
        chunk_idx
            .iter()
            .zip(chunk_shape)
            .map(|(i, s)| i * s)
            .collect();
    let chunk_end: SmallVec<[u64; 4]> =
        chunk_start
            .iter()
            .zip(chunk_shape)
            .zip(array_shape)
            .map(|((s, cs), a)| (s + cs).min(*a))
            .collect();

    let mut bbox_start: SmallVec<[u64; 4]> =
        chunk_end.clone();
    let mut bbox_end: SmallVec<[u64; 4]> =
        chunk_start.clone();

    for subset in subsets.subsets_iter() {
        let ranges = subset.to_ranges();
        for d in 0..ndim {
            let inter_start = ranges[d]
                .start
                .max(chunk_start[d]);
            let inter_end =
                ranges[d].end.min(chunk_end[d]);
            if inter_start < inter_end {
                bbox_start[d] = bbox_start[d]
                    .min(inter_start);
                bbox_end[d] =
                    bbox_end[d].max(inter_end);
            }
        }
    }

    let local_ranges: SmallVec<[Range<u64>; 4]> =
        bbox_start
            .iter()
            .zip(bbox_end.iter())
            .zip(chunk_start.iter())
            .map(|((s, e), cs)| {
                (s - cs)..(e - cs)
            })
            .collect();

    let actual_chunk_shape: SmallVec<[u64; 4]> =
        chunk_end
            .iter()
            .zip(chunk_start.iter())
            .map(|(e, s)| e - s)
            .collect();

    let subset = ChunkSubset {
        ranges: local_ranges,
    };
    if subset.is_full_chunk(&actual_chunk_shape) {
        None
    } else {
        Some(subset)
    }
}

// =============================================================================
// Grouped Chunk Plan
// =============================================================================

/// A grid group with deduplicated chunk indices, ready for reading.
/// Owned version of the grid group.
#[derive(Debug)]
pub struct OwnedGridGroup {
    pub sig: Arc<
        crate::chunk_plan::ChunkGridSignature,
    >,
    pub vars: Vec<IStr>,
    pub chunk_indices: Vec<Vec<u64>>,
    pub chunk_subsets: Vec<Option<ChunkSubset>>,
    pub array_shape: Vec<u64>,
}

impl OwnedGridGroup {
    pub fn new(
        sig: Arc<ChunkGridSignature>,
        vars: Vec<IStr>,
        chunk_indices: Vec<Vec<u64>>,
        chunk_subsets: Vec<Option<ChunkSubset>>,
        array_shape: Vec<u64>,
    ) -> Self {
        Self {
            sig,
            vars,
            chunk_indices,
            chunk_subsets,
            array_shape,
        }
    }
}

/// Grouped chunk plan - maps chunk grid signatures to plans.
///
/// This allows heterogeneous chunk layouts: variables with the same dimensions
/// but different chunk shapes will have different plans.
#[derive(Debug, Clone)]
pub struct GroupedChunkPlan {
    /// ChunkPlan by grid signature
    by_grid: BTreeMap<
        Arc<ChunkGridSignature>,
        ArraySubsetList,
    >,
    /// Variable name to grid signature lookup
    var_to_grid:
        BTreeMap<IStr, Arc<ChunkGridSignature>>,

    vars_by_grid: BTreeMap<
        Arc<ChunkGridSignature>,
        Vec<IStr>,
    >,

    /// Chunk grid by signature
    chunk_grid: BTreeMap<
        Arc<ChunkGridSignature>,
        Arc<ChunkGrid>,
    >,
}

#[allow(private_interfaces)]
impl GroupedChunkPlan {
    /// Create a new empty grouped chunk plan.
    pub fn new() -> Self {
        Self {
            by_grid: BTreeMap::new(),
            var_to_grid: BTreeMap::new(),
            vars_by_grid: BTreeMap::new(),
            chunk_grid: BTreeMap::new(),
        }
    }

    /// Insert a plan for a variable with the given grid signature.
    pub fn insert<T: IntoIStr>(
        &mut self,
        var: T,
        sig: Arc<ChunkGridSignature>,
        plan: ArraySubsetList,
        chunk_grid: Arc<ChunkGrid>,
    ) {
        let var = var.istr();
        self.var_to_grid.insert(var, sig.clone());
        self.by_grid
            .entry(sig.clone())
            .or_insert(plan);
        self.vars_by_grid
            .entry(sig.clone())
            .or_insert(vec![])
            .push(var);
        self.chunk_grid
            .entry(sig.clone())
            .or_insert(chunk_grid);
    }

    /// Get the chunk grid for a signature.
    pub fn get_chunk_grid(
        &self,
        sig: &ChunkGridSignature,
    ) -> Option<&Arc<ChunkGrid>> {
        self.chunk_grid.get(sig)
    }

    /// Get all variables for a grid signature.
    pub fn vars_for_grid(
        &self,
        sig: &ChunkGridSignature,
    ) -> Vec<IStr> {
        self.vars_by_grid
            .get(sig)
            .cloned()
            .unwrap_or_default()
    }

    /// Iterate over grid groups with deduplicated chunk indices.
    ///
    /// Each group yields its signature, associated variables,
    /// deduplicated (sorted) chunk indices, per-chunk local subsets,
    /// and the array shape. This avoids reading the same physical
    /// chunk multiple times when overlapping array subsets map to
    /// the same chunk.
    pub fn iter_consolidated_chunks(
        &self,
    ) -> impl Iterator<
        Item = BackendResult<OwnedGridGroup>,
    > + '_ {
        self.by_grid.iter().map(
            move |(sig, subsets)| {
                let vars = self
                    .vars_for_grid(sig.as_ref());
                let chunkgrid = self
                    .get_chunk_grid(sig.as_ref())
                    .ok_or_else(|| {
                        BackendError::MissingChunkGrid {
                            sig: sig.as_ref().clone(),
                        }
                    })?;
                let array_shape =
                    chunkgrid.array_shape().to_vec();
                let chunk_shape = sig.retrieval_shape();

                let mut seen: BTreeSet<Vec<u64>> =
                    BTreeSet::new();
                for subset in
                    subsets.subsets_iter()
                {
                    let indices = chunkgrid
                        .chunks_in_array_subset(
                            subset,
                        ).context(
                            crate::errors::backend::IncompatibleDimensionalitySnafu {
                                dims: sig.dims().to_vec(),
                                shape: chunkgrid.array_shape().to_vec(),
                                paths: vars.to_vec(),
                            }
                        )?;
                    if let Some(indices) = indices
                    {
                        for idx in
                            indices.indices()
                        {
                            seen.insert(
                                idx.to_vec(),
                            );
                        }
                    }
                }

                let chunk_indices: Vec<Vec<u64>> =
                    seen.into_iter().collect();

                let chunk_subsets: Vec<
                    Option<ChunkSubset>,
                > = chunk_indices
                    .iter()
                    .map(|idx| {
                        compute_chunk_subset(
                            idx,
                            chunk_shape,
                            &array_shape,
                            subsets,
                        )
                    })
                    .collect();

                Ok(OwnedGridGroup::new(
                    sig.clone(),
                    vars,
                    chunk_indices,
                    chunk_subsets,
                    array_shape,
                ))
            },
        )
    }

    /// Consolidated groups, optionally clearing for a literal-false predicate.
    ///
    /// Drops "redundant dim coordinate" groups: a 1D group whose only variable
    /// has the same name as its dim **and** that dim already appears in a
    /// higher-dim variable in `meta`. The higher-dim group materializes the
    /// dim column via
    /// [`crate::scan::column_policy::DimMaterialization::FromArray`], so
    /// scheduling the standalone dim-coord group adds duplicate reads and
    /// forces an extra `Independent` subtree at plan time.
    ///
    /// Auxiliary 1D coords sharing a dim with a larger grid (e.g. `latitude`,
    /// `longitude`, `station_id` along `point`) are *kept* because the larger
    /// grid does not materialize them.
    pub(crate) fn owned_grid_groups_for_io(
        &self,
        literal_false_clear: bool,
        meta: &crate::meta::ZarrMeta,
    ) -> BackendResult<Vec<OwnedGridGroup>> {
        if literal_false_clear {
            return Ok(Vec::new());
        }
        let multi_dim_dim_set: BTreeSet<IStr> =
            meta.all_array_paths()
                .into_iter()
                .filter_map(|p| {
                    meta.array_by_path(p)
                })
                .filter(|m| m.dims.len() > 1)
                .flat_map(|m| {
                    m.dims
                        .iter()
                        .copied()
                        .collect::<Vec<_>>()
                })
                .collect();
        let groups: Vec<OwnedGridGroup> = self
            .iter_consolidated_chunks()
            .collect::<BackendResult<_>>()?;
        Ok(groups
            .into_iter()
            .filter(|g| {
                let dims = g.sig.dims();
                if dims.len() != 1 {
                    return true;
                }
                let dim = dims[0];
                if !multi_dim_dim_set
                    .contains(&dim)
                {
                    return true;
                }
                // Only drop when the group is exactly the dim's coordinate.
                !(g.vars.len() == 1
                    && g.vars[0] == dim)
            })
            .collect())
    }

    /// Get the internal var_to_grid map.
    pub fn var_to_grid(
        &self,
    ) -> &BTreeMap<IStr, Arc<ChunkGridSignature>>
    {
        &self.var_to_grid
    }
}

impl Default for GroupedChunkPlan {
    fn default() -> Self {
        Self::new()
    }
}
