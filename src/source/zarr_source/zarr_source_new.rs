use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_polars::PySchema;
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;

use crate::chunk_plan::{
    ChunkGridSignature, DatasetSelection,
    GroupedChunkPlan,
    selection_to_grouped_chunk_plan,
    selection_to_grouped_chunk_plan_unified,
};
use crate::store::StoreInput;
use crate::{IStr, IntoIStr};

use super::{
    DEFAULT_BATCH_SIZE, GridIterState,
    ZarrSource, to_py_err,
};

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
                    let legacy = unified.planning_meta();
                    (opened, legacy, Some(unified), is_hier)
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

        let vars: Vec<IStr> =
            if let Some(v) = variables {
                v.into_iter()
                    .map(|s| s.istr())
                    .collect()
            } else if let Some(ref unified) =
                unified_meta
            {
                if is_hierarchical {
                    unified.all_data_var_paths()
                } else {
                    meta.data_vars.clone()
                }
            } else {
                meta.data_vars.clone()
            };

        if vars.is_empty() && !is_hierarchical {
            return Err(PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(
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
        let grid_states = Vec::new();

        // Get dims from unified analysis for hierarchical stores
        let dims: Vec<IStr> = if is_hierarchical {
            unified_meta
                .as_ref()
                .map(|m| {
                    m.dim_analysis
                        .all_dims
                        .clone()
                })
                .unwrap_or_default()
        } else {
            meta.arrays
                .get(&vars[0])
                .map(|m| {
                    m.dims
                        .iter()
                        .cloned()
                        .collect()
                })
                .filter(|d: &Vec<IStr>| {
                    !d.is_empty()
                })
                .unwrap_or_else(|| {
                    // Fallback: generate dim names
                    let primary_path = &meta
                        .arrays[&vars[0]]
                        .path;
                    if let Ok(primary) =
                        Array::open(
                            store.clone(),
                            primary_path.as_ref(),
                        )
                    {
                        (0..primary
                            .dimensionality())
                            .map(|i| {
                                format!("dim_{i}")
                                    .istr()
                            })
                            .collect()
                    } else {
                        vec![]
                    }
                })
        };

        let mut source = Self {
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
        };

        let plan = if source.is_hierarchical {
            selection_to_grouped_chunk_plan_unified(
                &DatasetSelection::NoSelectionMade,
                source
                    .unified_meta
                    .as_ref()
                    .expect("hierarchical meta missing"),
                source.store.clone(),
            )
        } else {
            selection_to_grouped_chunk_plan(
                &DatasetSelection::NoSelectionMade,
                &source.meta,
                source.store.clone(),
            )
        }
        .map_err(to_py_err)?;
        source.set_grid_states_from_plan(plan);

        Ok(source)
    }

    pub(super) fn consume_chunk_budget(
        &mut self,
    ) -> PyResult<()> {
        if let Some(left) = self.chunks_left {
            if left == 0 {
                return Err(PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >(
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
        let schema = if let Some(ref unified) =
            self.unified_meta
        {
            if self.is_hierarchical {
                // For hierarchical stores, always show full schema with struct columns
                // The with_columns filter will be applied during data retrieval
                unified.tidy_schema(None)
            } else {
                unified
                    .tidy_schema(Some(&self.vars))
            }
        } else {
            self.meta
                .tidy_schema(Some(&self.vars))
        };
        PySchema(Arc::new(schema))
    }

    /// Rebuild grid states from a new GroupedChunkPlan (after predicate pushdown)
    pub(super) fn set_grid_states_from_plan(
        &mut self,
        plan: GroupedChunkPlan,
    ) {
        use std::collections::HashSet;

        // Only include variables that were requested
        let requested: HashSet<&IStr> =
            self.vars.iter().collect();
        let mut by_dims: std::collections::BTreeMap<
            Vec<IStr>,
            GridIterState,
        > = std::collections::BTreeMap::new();
        let target_dims = if self.is_hierarchical
        {
            plan.iter_grids()
                .map(|(sig, _, _, _)| {
                    sig.dims().to_vec()
                })
                .max_by_key(|d| d.len())
        } else {
            None
        };
        let use_only_target =
            target_dims.is_some();

        for (sig, vars, subsets, chunk_grid) in
            plan.iter_grids()
        {
            if use_only_target
                && target_dims
                    .as_ref()
                    .map(|d| {
                        sig.dims() != d.as_slice()
                    })
                    .unwrap_or(false)
            {
                continue;
            }
            // Filter to only requested variables
            let variables: Vec<IStr> = vars
                .iter()
                .filter(|v| {
                    requested.contains(*v)
                })
                .map(|v| (*v).clone())
                .collect();

            // Skip grids with no requested variables
            if variables.is_empty() {
                continue;
            }

            let dims_key = sig.dims().to_vec();
            let entry = by_dims
                .entry(dims_key.clone())
                .or_insert_with(|| {
                    let sig = ChunkGridSignature::from_dims_only(
                        dims_key.clone(),
                    );
                    GridIterState::new(
                        Arc::new(sig),
                        Vec::new(),
                        chunk_grid.clone(),
                    )
                });
            for var in variables {
                if !entry.variables.contains(&var)
                {
                    entry.variables.push(var);
                }
            }
            if entry.pending_subsets.is_empty()
                && entry.current_subset.is_none()
                && subsets
                    .subsets_iter()
                    .next()
                    .is_some()
            {
                entry.add_subsets(
                    subsets
                        .subsets_iter()
                        .cloned(),
                );
            }
        }

        let mut grid_states: Vec<GridIterState> =
            by_dims.into_values().collect();
        for state in &mut grid_states {
            if state.pending_subsets.is_empty()
                && state.current_subset.is_none()
            {
                if let Some(var_name) =
                    state.variables.first()
                {
                    if let Some(var_meta) = self
                        .meta
                        .arrays
                        .get(var_name)
                    {
                        if let Ok(arr) =
                            Array::open(
                                self.store
                                    .clone(),
                                var_meta
                                    .path
                                    .as_ref(),
                            )
                        {
                            let shape = arr
                                .shape()
                                .to_vec();
                            if let Ok(full) = ArraySubset::new_with_start_shape(
                                vec![0; shape.len()],
                                shape,
                            ) {
                                state.add_subsets(
                                    std::iter::once(
                                        full,
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }

        self.grid_states = grid_states;
        self.current_grid_idx = 0;
    }
}
