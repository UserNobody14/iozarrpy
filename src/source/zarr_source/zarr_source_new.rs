use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_polars::PySchema;
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;

use crate::chunk_plan::{ChunkGridSignature, GroupedChunkPlan};
use crate::store::StoreInput;
use crate::{IStr, IntoIStr};

use super::{GridIterState, DEFAULT_BATCH_SIZE, ZarrSource};

impl ZarrSource {
    pub(super) fn new_impl(
        store_input: StoreInput,
        batch_size: Option<usize>,
        n_rows: Option<usize>,
        variables: Option<Vec<String>>,
        max_chunks_to_read: Option<usize>,
    ) -> PyResult<Self> {
        // Try hierarchical loading first, with fallback to legacy loading
        let (opened, meta, unified_meta, is_hierarchical) =
            match crate::meta::open_and_load_zarr_meta_from_input(store_input.clone()) {
                Ok((opened, unified)) => {
                    let is_hier = unified.is_hierarchical();
                    let legacy = crate::meta::ZarrDatasetMeta::from(&unified);
                    // Check if we got any data vars from hierarchical loading
                    if !legacy.data_vars.is_empty() {
                        (opened, legacy, Some(unified), is_hier)
                    } else {
                        // Hierarchical loading found no data vars, fall back to legacy
                        let (opened2, legacy2) =
                            crate::meta::open_and_load_dataset_meta_from_input(store_input)
                                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                        (opened2, legacy2, None, false)
                    }
                }
                Err(_) => {
                    // Hierarchical loading failed, use legacy
                    let (opened, legacy) =
                        crate::meta::open_and_load_dataset_meta_from_input(store_input)
                            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                    (opened, legacy, None, false)
                }
            };

        let store = opened.store.clone();

        let vars: Vec<IStr> = if let Some(v) = variables {
            v.into_iter().map(|s| s.istr()).collect()
        } else {
            meta.data_vars.clone()
        };

        if vars.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "no variables found/selected",
            ));
        }

        // Polars may pass 0 to mean "unspecified"; interpret it as the default.
        let n_rows_left = match n_rows {
            None | Some(0) => usize::MAX,
            Some(n) => n,
        };
        let batch_size = match batch_size {
            None | Some(0) => DEFAULT_BATCH_SIZE,
            Some(n) => n,
        };

        // Build per-grid iteration state for all requested variables
        let grid_states = build_initial_grid_states(&meta, &vars, &store)?;
        
        // Get dims from first variable (for backward compat)
        let dims: Vec<IStr> = meta
            .arrays
            .get(&vars[0])
            .map(|m| m.dims.iter().cloned().collect())
            .filter(|d: &Vec<IStr>| !d.is_empty())
            .unwrap_or_else(|| {
                // Fallback: generate dim names
                let primary_path = &meta.arrays[&vars[0]].path;
                if let Ok(primary) = Array::open(store.clone(), primary_path.as_ref()) {
                    (0..primary.dimensionality()).map(|i| format!("dim_{i}").istr()).collect()
                } else {
                    vec![]
                }
            });

        Ok(Self {
            meta,
            unified_meta,
            store,
            dims,
            vars,
            batch_size,
            n_rows_left,
            predicate: None,
            with_columns: None,
            grid_states,
            current_grid_idx: 0,
            done: false,
            chunks_left: max_chunks_to_read,
            is_hierarchical,
        })
    }

    pub(super) fn consume_chunk_budget(&mut self) -> PyResult<()> {
        if let Some(left) = self.chunks_left {
            if left == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "max_chunks_to_read limit hit",
                ));
            }
            self.chunks_left = Some(left - 1);
        }
        Ok(())
    }

    pub(super) fn schema_impl(&self) -> PySchema {
        // Use unified meta for schema if available (supports struct columns)
        // For hierarchical stores, pass None to include all child groups as struct columns
        // unless specific variables were selected by the user
        let schema = if let Some(ref unified) = self.unified_meta {
            if self.is_hierarchical {
                // For hierarchical stores, always show full schema with struct columns
                // The with_columns filter will be applied during data retrieval
                unified.tidy_schema(None)
            } else {
                unified.tidy_schema(Some(&self.vars))
            }
        } else {
            self.meta.tidy_schema(Some(&self.vars))
        };
        PySchema(Arc::new(schema))
    }
    
    /// Rebuild grid states from a new GroupedChunkPlan (after predicate pushdown)
    pub(super) fn set_grid_states_from_plan(&mut self, plan: GroupedChunkPlan) {
        use std::collections::HashSet;
        
        // Only include variables that were requested
        let requested: HashSet<&IStr> = self.vars.iter().collect();
        let mut grid_states = Vec::new();
        
        for (sig, vars, subsets) in plan.iter_grids() {
            // Filter to only requested variables
            let variables: Vec<IStr> = vars
                .iter()
                .filter(|v| requested.contains(*v))
                .map(|v| (*v).clone())
                .collect();
            
            // Skip grids with no requested variables
            if variables.is_empty() {
                continue;
            }
            
            let sig_arc = Arc::new(sig.clone());
            let mut state = GridIterState::new(sig_arc, variables);
            state.add_subsets(subsets.subsets_iter().cloned());
            grid_states.push(state);
        }
        
        self.grid_states = grid_states;
        self.current_grid_idx = 0;
    }
}

/// Build initial grid states for all variables (no predicate = scan all elements).
fn build_initial_grid_states(
    meta: &crate::meta::ZarrDatasetMeta,
    vars: &[IStr],
    store: &zarrs::storage::ReadableWritableListableStorage,
) -> PyResult<Vec<GridIterState>> {
    use std::collections::BTreeMap;
    
    // Group variables by their chunk grid signature
    let mut grids: BTreeMap<ChunkGridSignature, Vec<IStr>> = BTreeMap::new();
    
    for var in vars {
        let Some(var_meta) = meta.arrays.get(var) else {
            continue;
        };
        
        let arr = Array::open(store.clone(), var_meta.path.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        // Get chunk shape from first chunk
        let zero = vec![0u64; arr.dimensionality()];
        let chunk_shape: Vec<u64> = if arr.dimensionality() > 0 {
            arr.chunk_shape(&zero)
                .map(|cs| cs.iter().map(|x| x.get()).collect())
                .unwrap_or_else(|_| arr.shape().to_vec())
        } else {
            vec![]
        };
        
        let sig = ChunkGridSignature::new(var_meta.dims.clone(), chunk_shape);
        grids.entry(sig).or_default().push(var.clone());
    }
    
    // Build grid states with full array subsets (scan all)
    let mut grid_states = Vec::new();
    
    for (sig, variables) in grids {
        // Get array shape for the first variable in this grid
        let first_var = &variables[0];
        let var_meta = &meta.arrays[first_var];
        let arr = Array::open(store.clone(), var_meta.path.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        let shape = arr.shape().to_vec();
        
        let mut state = GridIterState::new(Arc::new(sig), variables);
        
        // Add full array as a single subset (scan all)
        if !shape.is_empty() {
            let full_subset = ArraySubset::new_with_start_shape(
                vec![0u64; shape.len()],
                shape,
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            state.pending_subsets.push_back(full_subset);
        }
        
        grid_states.push(state);
    }
    
    Ok(grid_states)
}
