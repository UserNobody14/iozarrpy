use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_polars::PySchema;
use zarrs::array::Array;

use crate::chunk_plan::ChunkPlan;
use crate::store::StoreInput;
use crate::{IStr, IntoIStr};

use super::{DEFAULT_BATCH_SIZE, ZarrSource};

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

        // Primary var defines the chunk iteration.
        let primary_path = meta
            .arrays
            .get(&vars[0])
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown variable"))?
            .path
            .clone();
        let primary = Array::open(store.clone(), primary_path.as_ref()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })?;

        let primary_grid_shape = primary.chunk_grid().grid_shape().to_vec();

        // Polars may pass 0 to mean "unspecified"; interpret it as the default.
        let n_rows_left = match n_rows {
            None | Some(0) => usize::MAX,
            Some(n) => n,
        };
        let batch_size = match batch_size {
            None | Some(0) => DEFAULT_BATCH_SIZE,
            Some(n) => n,
        };

        let dims: Vec<IStr> = meta
            .arrays
            .get(&vars[0])
            .map(|m| m.dims.iter().cloned().collect())
            .filter(|d: &Vec<IStr>| !d.is_empty())
            .unwrap_or_else(|| (0..primary.dimensionality()).map(|i| format!("dim_{i}").istr()).collect());
        let chunk_iter = ChunkPlan::all(primary_grid_shape.clone()).into_index_iter();

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
            primary_grid_shape,
            chunk_iter,
            current_chunk_indices: None,
            chunk_offset: 0,
            done: primary.dimensionality() == 0 && false,
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
}

