use std::collections::BTreeMap;
use std::sync::Arc;

use zarrs::array::ChunkGrid;

use crate::IStr;
use crate::chunk_plan::indexing::selection::ArraySubsetList;
use crate::chunk_plan::indexing::types::ChunkGridSignature;

// =============================================================================
// Grouped Chunk Plan
// =============================================================================

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

    /// Iterate over (grid_signature, variables, chunk_plan) tuples.
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
