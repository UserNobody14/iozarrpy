use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use zarrs::array::Array;

use crate::chunk_plan::{compile_expr_to_dataset_selection, DSelection};
use crate::meta::{open_and_load_zarr_meta, ZarrDatasetMeta};
use crate::py::expr_extract::extract_expr;
use crate::{IStr, IntoIStr};

/// Helper to compute chunk indices for a specific variable based on its selection.
fn compute_chunks_for_variable(
    ref_array: &Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
    selection: &crate::chunk_plan::DatasetSelection,
    ref_var: &str,
) -> Vec<Vec<u64>> {
    let grid_shape = ref_array.chunk_grid().grid_shape();

    match selection {
        crate::chunk_plan::DatasetSelection::NoSelectionMade => {
            // Select all chunks - generate all indices
            let mut chunk_indices = Vec::new();
            let mut idx = vec![0u64; grid_shape.len()];
            loop {
                chunk_indices.push(idx.clone());
                // Increment (last dim fastest)
                let mut carry = true;
                for d in (0..idx.len()).rev() {
                    if carry {
                        idx[d] += 1;
                        if idx[d] < grid_shape[d] {
                            carry = false;
                        } else {
                            idx[d] = 0;
                        }
                    }
                }
                if carry {
                    break;
                }
            }
            chunk_indices
        }
        crate::chunk_plan::DatasetSelection::Empty => {
            // No chunks
            vec![]
        }
        crate::chunk_plan::DatasetSelection::Selection(grouped) => {
            // Get selection for this variable
            let Some(darray) = grouped.get(ref_var) else {
                // Variable not in selection - return empty
                return vec![];
            };

            let mut chunk_set = Vec::new();
            for subset in darray.subsets_iter() {
                if let Ok(Some(chunks)) = ref_array.chunks_in_array_subset(subset) {
                    for chunk_index in chunks.indices().iter() {
                        chunk_set.push(chunk_index.iter().copied().collect());
                    }
                }
            }
            chunk_set
        }
    }
}

#[pyfunction]
#[pyo3(signature = (zarr_url, predicate, variables=None))]
pub(crate) fn selected_chunks(
    py: Python<'_>,
    zarr_url: String,
    predicate: &Bound<'_, PyAny>,
    variables: Option<Vec<String>>,
) -> PyResult<Vec<Py<PyAny>>> {
    let (opened, zarr_meta) = open_and_load_zarr_meta(&zarr_url)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    // Convert to ZarrDatasetMeta - preserves hierarchical paths from path_to_array
    let meta = ZarrDatasetMeta::from(&zarr_meta);

    // Convert to IStr at the Python boundary
    let vars: Vec<IStr> = variables
        .map(|v| v.into_iter().map(|s| s.istr()).collect())
        .unwrap_or_else(|| meta.data_vars.clone());
    if vars.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "no variables found/selected",
        ));
    }

    let expr = extract_expr(predicate)?;

    // Pick the first requested variable as the reference
    let ref_var = &vars[0];
    let ref_meta = meta
        .arrays
        .get(ref_var)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown variable"))?;
    let ref_array = Array::open(opened.store.clone(), ref_meta.path.as_ref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Compile to dataset selection (not chunk plan - we'll compute chunks ourselves)
    let (selection, _stats) = compile_expr_to_dataset_selection(&expr, &meta, opened.store.clone())
        .unwrap_or_else(|_| {
            (
                crate::chunk_plan::DatasetSelection::NoSelectionMade,
                crate::chunk_plan::PlannerStats { coord_reads: 0 },
            )
        });

    // Compute chunk indices specifically for the requested variable
    let ref_var_str: &str = ref_var.as_ref();
    let indices = compute_chunks_for_variable(&ref_array, &selection, ref_var_str);

    let mut out: Vec<Py<PyAny>> = Vec::new();
    for idx in indices {
        let chunk_shape_nz = ref_array
            .chunk_shape(&idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let shape: Vec<u64> = chunk_shape_nz.iter().map(|x| x.get()).collect();
        let origin = ref_array
            .chunk_grid()
            .chunk_origin(&idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .unwrap_or_else(|| vec![0; shape.len()]);

        let chunk = crate::chunk_plan::ChunkId {
            indices: idx,
            origin,
            shape,
        };
        let d = pyo3::types::PyDict::new(py);
        d.set_item("indices", &chunk.indices)?;
        d.set_item("origin", &chunk.origin)?;
        d.set_item("shape", &chunk.shape)?;
        out.push(d.into());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (zarr_url, predicate, variables=None))]
pub(crate) fn _selected_chunks_debug(
    py: Python<'_>,
    zarr_url: String,
    predicate: &Bound<'_, PyAny>,
    variables: Option<Vec<String>>,
) -> PyResult<(Vec<Py<PyAny>>, u64)> {
    let (opened, zarr_meta) = open_and_load_zarr_meta(&zarr_url)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    // Convert to ZarrDatasetMeta - preserves hierarchical paths from path_to_array
    let meta = ZarrDatasetMeta::from(&zarr_meta);

    // Convert to IStr at the Python boundary
    let vars: Vec<IStr> = variables
        .map(|v| v.into_iter().map(|s| s.istr()).collect())
        .unwrap_or_else(|| meta.data_vars.clone());
    if vars.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "no variables found/selected",
        ));
    }

    let expr = extract_expr(predicate)?;

    // Pick the first requested variable as the reference
    let ref_var = &vars[0];
    let ref_meta = meta
        .arrays
        .get(ref_var)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown variable"))?;
    let ref_array = Array::open(opened.store.clone(), ref_meta.path.as_ref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Compile to dataset selection
    let (selection, stats) = compile_expr_to_dataset_selection(&expr, &meta, opened.store.clone())
        .map_err(|e| match e {
            crate::chunk_plan::CompileError::Unsupported(e) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
            }
            crate::chunk_plan::CompileError::MissingPrimaryDims(e) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
            }
        })?;

    // Compute chunk indices specifically for the requested variable
    let ref_var_str: &str = ref_var.as_ref();
    let indices = compute_chunks_for_variable(&ref_array, &selection, ref_var_str);

    let mut out: Vec<Py<PyAny>> = Vec::new();
    for idx in indices {
        let chunk_shape_nz = ref_array
            .chunk_shape(&idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let shape: Vec<u64> = chunk_shape_nz.iter().map(|x| x.get()).collect();
        let origin = ref_array
            .chunk_grid()
            .chunk_origin(&idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .unwrap_or_else(|| vec![0; shape.len()]);

        let chunk = crate::chunk_plan::ChunkId {
            indices: idx,
            origin,
            shape,
        };
        let d = pyo3::types::PyDict::new(py);
        d.set_item("indices", &chunk.indices)?;
        d.set_item("origin", &chunk.origin)?;
        d.set_item("shape", &chunk.shape)?;
        out.push(d.into());
    }

    Ok((out, stats.coord_reads))
}

/// Debug function that returns per-variable chunk selections.
/// 
/// Returns:
/// - `inferred_variables`: List of variable names found in the DatasetSelection
/// - `per_variable_chunks`: Dict mapping variable name -> list of chunk dicts
/// - `coord_reads`: Number of coordinate array reads performed
#[pyfunction]
#[pyo3(signature = (zarr_url, expr))]
pub(crate) fn _selected_variables_debug(
    py: Python<'_>,
    zarr_url: String,
    expr: &Bound<'_, PyAny>,
) -> PyResult<(Vec<String>, HashMap<String, Vec<Py<PyAny>>>, u64)> {
    let (opened, zarr_meta) = open_and_load_zarr_meta(&zarr_url)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    // Convert to ZarrDatasetMeta - preserves hierarchical paths from path_to_array
    let meta = ZarrDatasetMeta::from(&zarr_meta);

    let parsed_expr = extract_expr(expr)?;

    let (selection, stats) = compile_expr_to_dataset_selection(
        &parsed_expr,
        &meta,
        opened.store.clone(),
    )
    .map_err(|e| match e {
        crate::chunk_plan::CompileError::Unsupported(e) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
        }
        crate::chunk_plan::CompileError::MissingPrimaryDims(e) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
        }
    })?;

    // Get the variable names from the selection (convert IStr -> String at Python boundary)
    let inferred_vars: Vec<String> = selection.vars().map(|(k, _)| k.to_string()).collect();

    // Plan chunk indices for all variables in the selection
    let grouped_plan = crate::chunk_plan::selection_to_grouped_chunk_plan(&selection, &meta, opened.store.clone())
        .map_err(|e| match e {
            crate::chunk_plan::CompileError::Unsupported(e) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
            }
            crate::chunk_plan::CompileError::MissingPrimaryDims(e) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
            }
        })?;

    // Convert to Python-friendly format
    let mut out: HashMap<String, Vec<Py<PyAny>>> = HashMap::new();
    
    // Iterate over each variable's plan
    for (var_istr, sig) in grouped_plan.var_to_grid() {
        let var = var_istr.to_string();
        let Some(var_meta) = meta.arrays.get(var_istr) else {
            continue;
        };
        let arr = Array::open(opened.store.clone(), var_meta.path.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Get the subsets for this variable's grid signature
        let Some(subsets) = grouped_plan.by_grid().get(sig) else {
            continue;
        };

        // Convert array subsets to chunk indices
        let mut var_chunks: Vec<Py<PyAny>> = Vec::new();
        for subset in subsets.subsets_iter() {
            if let Ok(Some(chunks)) = arr.chunks_in_array_subset(subset) {
                for chunk_idx in chunks.indices().iter() {
                    let idx: Vec<u64> = chunk_idx.iter().copied().collect();
                    let chunk_shape_nz = arr
                        .chunk_shape(&idx)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                    let shape: Vec<u64> = chunk_shape_nz.iter().map(|x| x.get()).collect();
                    let origin = arr
                        .chunk_grid()
                        .chunk_origin(&idx)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                        .unwrap_or_else(|| vec![0; shape.len()]);

                    let d = pyo3::types::PyDict::new(py);
                    d.set_item("indices", &idx)?;
                    d.set_item("origin", &origin)?;
                    d.set_item("shape", &shape)?;
                    var_chunks.push(d.into());
                }
            }
        }
        out.insert(var, var_chunks);
    }

    Ok((inferred_vars, out, stats.coord_reads))
}
