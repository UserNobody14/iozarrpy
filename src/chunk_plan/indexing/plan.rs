use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use zarrs::array::ChunkGrid;

use crate::IStr;
use crate::chunk_plan::indexing::selection::ArraySubsetList;
use crate::chunk_plan::indexing::types::ChunkGridSignature;

// =============================================================================
// Grouped Chunk Plan
// =============================================================================

/// A grid group with deduplicated chunk indices, ready for reading.
///
/// Produced by [`GroupedChunkPlan::iter_consolidated_chunks`]. Each group
/// corresponds to one chunk-grid signature and contains the sorted, unique
/// set of chunk indices that need to be read.
pub struct ConsolidatedGridGroup<'a> {
    /// The chunk grid signature for this group.
    pub sig: &'a ChunkGridSignature,
    /// Variables sharing this chunk grid.
    pub vars: Vec<&'a IStr>,
    /// Deduplicated, sorted chunk indices.
    pub chunk_indices: Vec<Vec<u64>>,
    /// Array shape from the chunk grid.
    pub array_shape: Vec<u64>,
}

/// Grouped chunk plan - maps chunk grid signatures to plans.
///
/// This allows heterogeneous chunk layouts: variables with the same dimensions
/// but different chunk shapes will have different plans.
#[derive(Debug, Clone)]
pub(crate) struct GroupedChunkPlan {
    /// ChunkPlan by grid signature
    by_grid: BTreeMap<
        Arc<ChunkGridSignature>,
        ArraySubsetList,
    >,
    /// Variable name to grid signature lookup
    var_to_grid:
        BTreeMap<IStr, Arc<ChunkGridSignature>>,

    /// Chunk grid by signature
    chunk_grid: BTreeMap<
        Arc<ChunkGridSignature>,
        Arc<ChunkGrid>,
    >,
}

impl GroupedChunkPlan {
    /// Create a new empty grouped chunk plan.
    pub fn new() -> Self {
        Self {
            by_grid: BTreeMap::new(),
            var_to_grid: BTreeMap::new(),
            chunk_grid: BTreeMap::new(),
        }
    }

    /// Insert a plan for a variable with the given grid signature.
    pub fn insert(
        &mut self,
        var: IStr,
        sig: Arc<ChunkGridSignature>,
        plan: ArraySubsetList,
        chunk_grid: Arc<ChunkGrid>,
    ) {
        self.var_to_grid.insert(var, sig.clone());
        self.by_grid
            .entry(sig.clone())
            .or_insert(plan);
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
    ) -> Vec<&IStr> {
        self.var_to_grid
            .iter()
            .filter(|(_, s)| s.as_ref() == sig)
            .map(|(v, _)| v)
            .collect()
    }

    /// Iterate over (grid_signature, variables, chunk_plan, chunk_grid) tuples.
    ///
    /// This yields the *raw* array subsets without deduplication.
    /// For reading, prefer [`iter_consolidated_chunks`] which deduplicates
    /// chunk indices across potentially overlapping subsets.
    pub fn iter_grids(
        &self,
    ) -> impl Iterator<
        Item = (
            &ChunkGridSignature,
            Vec<&IStr>,
            &ArraySubsetList,
            &Arc<ChunkGrid>,
        ),
    > {
        self.by_grid.iter().map(
            move |(sig, plan)| {
                let vars = self
                    .vars_for_grid(sig.as_ref());
                (
                    sig.as_ref(),
                    vars,
                    plan,
                    self.get_chunk_grid(
                        sig.as_ref(),
                    )
                    .unwrap(),
                )
            },
        )
    }

    /// Count total unique chunks across all grid groups.
    ///
    /// This deduplicates chunk indices from potentially overlapping
    /// array subsets within each grid group.
    pub fn total_unique_chunks(
        &self,
    ) -> Result<usize, String> {
        let mut total = 0;
        for (_, _, subsets, chunkgrid) in
            self.iter_grids()
        {
            let mut seen: BTreeSet<Vec<u64>> =
                BTreeSet::new();
            for subset in subsets.subsets_iter() {
                let indices = chunkgrid
                    .chunks_in_array_subset(subset)
                    .map_err(|e| {
                        format!(
                        "chunk grid traversal error: {e}"
                    )
                    })?;
                if let Some(indices) = indices {
                    for idx in indices.indices() {
                        seen.insert(idx.to_vec());
                    }
                }
            }
            total += seen.len();
        }
        Ok(total)
    }

    /// Iterate over grid groups with deduplicated chunk indices.
    ///
    /// Each group yields its signature, associated variables,
    /// deduplicated (sorted) chunk indices, and the array shape.
    /// This avoids reading the same physical chunk multiple times
    /// when overlapping array subsets map to the same chunk.
    pub fn iter_consolidated_chunks(
        &self,
    ) -> impl Iterator<
        Item = Result<
            ConsolidatedGridGroup<'_>,
            String,
        >,
    > + '_ {
        self.by_grid.iter().map(
            move |(sig, subsets)| {
                let vars = self
                    .vars_for_grid(sig.as_ref());
                let chunkgrid = self
                    .get_chunk_grid(sig.as_ref())
                    .ok_or_else(|| {
                        "missing chunk grid for signature"
                            .to_string()
                    })?;
                let array_shape =
                    chunkgrid.array_shape().to_vec();

                let mut seen: BTreeSet<Vec<u64>> =
                    BTreeSet::new();
                for subset in
                    subsets.subsets_iter()
                {
                    let indices = chunkgrid
                        .chunks_in_array_subset(
                            subset,
                        )
                        .map_err(|e| {
                            format!(
                            "chunk grid traversal error: {e}"
                        )
                        })?;
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

                Ok(ConsolidatedGridGroup {
                    sig: sig.as_ref(),
                    vars,
                    chunk_indices: seen
                        .into_iter()
                        .collect(),
                    array_shape,
                })
            },
        )
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
