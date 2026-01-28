use std::collections::BTreeMap;
use std::sync::Arc;

use crate::chunk_plan::indexing::selection::ArraySubsetList;
use crate::chunk_plan::indexing::types::ChunkGridSignature;
use crate::IStr;

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
}

impl GroupedChunkPlan {
    /// Create a new empty grouped chunk plan.
    pub fn new() -> Self {
        Self {
            by_grid: BTreeMap::new(),
            var_to_grid: BTreeMap::new(),
        }
    }

    /// Insert a plan for a variable with the given grid signature.
    pub fn insert(
        &mut self,
        var: IStr,
        sig: Arc<ChunkGridSignature>,
        plan: ArraySubsetList,
    ) {
        self.var_to_grid.insert(var, sig.clone());
        self.by_grid.entry(sig).or_insert(plan);
    }

    /// Get the plan for a specific variable.
    pub fn get_plan(
        &self,
        var: &str,
    ) -> Option<&ArraySubsetList> {
        let sig = self
            .var_to_grid
            .get(&crate::IntoIStr::istr(var))?;
        self.by_grid.get(sig)
    }

    /// Get the grid signature for a variable.
    pub fn get_signature(
        &self,
        var: &str,
    ) -> Option<&Arc<ChunkGridSignature>> {
        self.var_to_grid
            .get(&crate::IntoIStr::istr(var))
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

    /// Get the number of unique chunk grids.
    pub fn num_grids(&self) -> usize {
        self.by_grid.len()
    }

    /// Get the number of variables.
    pub fn num_vars(&self) -> usize {
        self.var_to_grid.len()
    }

    /// Check if this plan is empty.
    pub fn is_empty(&self) -> bool {
        self.by_grid.is_empty()
    }

    /// Total number of chunks across all grids.
    pub fn total_chunks(&self) -> usize {
        self.by_grid
            .values()
            .map(|plan| {
                plan.clone().num_elements_usize()
            })
            .sum()
    }

    /// Iterate over (grid_signature, variables, chunk_plan) tuples.
    pub fn iter_grids(
        &self,
    ) -> impl Iterator<
        Item = (
            &ChunkGridSignature,
            Vec<&IStr>,
            &ArraySubsetList,
        ),
    > {
        self.by_grid.iter().map(
            move |(sig, plan)| {
                let vars = self
                    .vars_for_grid(sig.as_ref());
                (sig.as_ref(), vars, plan)
            },
        )
    }

    /// Iterate over (grid_signature, chunk_plan) pairs.
    pub fn iter_plans(
        &self,
    ) -> impl Iterator<
        Item = (
            &Arc<ChunkGridSignature>,
            &ArraySubsetList,
        ),
    > {
        self.by_grid.iter()
    }

    /// Get the internal by_grid map.
    pub fn by_grid(
        &self,
    ) -> &BTreeMap<
        Arc<ChunkGridSignature>,
        ArraySubsetList,
    > {
        &self.by_grid
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
