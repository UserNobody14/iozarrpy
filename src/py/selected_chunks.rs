use pyo3::prelude::*;
use pyo3::types::PyAny;
use zarrs::array::Array;

use crate::chunk_plan::{compile_expr_to_chunk_plan, ChunkPlan};
use crate::meta::open_and_load_dataset_meta;
use crate::py::expr_extract::extract_expr;

#[pyfunction]
#[pyo3(signature = (zarr_url, predicate, variables=None))]
pub(crate) fn selected_chunks(
    py: Python<'_>,
    zarr_url: String,
    predicate: &Bound<'_, PyAny>,
    variables: Option<Vec<String>>,
) -> PyResult<Vec<Py<PyAny>>> {
    let (opened, meta) = open_and_load_dataset_meta(&zarr_url)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let vars = variables.unwrap_or_else(|| meta.data_vars.clone());
    if vars.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "no variables found/selected",
        ));
    }
    let primary_var = &vars[0];

    let expr = extract_expr(predicate)?;

    let primary_meta = meta
        .arrays
        .get(primary_var)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown primary variable"))?;
    let primary = Array::open(opened.store.clone(), &primary_meta.path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let (plan, _stats) =
        compile_expr_to_chunk_plan(&expr, &meta, opened.store.clone(), primary_var)
            .unwrap_or_else(|_| {
                let grid_shape = primary.chunk_grid().grid_shape().to_vec();
                (
                    ChunkPlan::all(grid_shape),
                    crate::chunk_plan::PlannerStats { coord_reads: 0 },
                )
            });

    let mut out: Vec<Py<PyAny>> = Vec::new();
    for idx in plan.into_index_iter() {
        let chunk_shape_nz = primary
            .chunk_shape(&idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let shape: Vec<u64> = chunk_shape_nz.iter().map(|x| x.get()).collect();
        let origin = primary
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
    let (opened, meta) = open_and_load_dataset_meta(&zarr_url)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let vars = variables.unwrap_or_else(|| meta.data_vars.clone());
    if vars.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "no variables found/selected",
        ));
    }
    let primary_var = &vars[0];

    let expr = extract_expr(predicate)?;

    let primary_meta = meta
        .arrays
        .get(primary_var)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown primary variable"))?;
    let primary = Array::open(opened.store.clone(), &primary_meta.path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let (plan, stats) = compile_expr_to_chunk_plan(&expr, &meta, opened.store.clone(), primary_var)
        .map_err(|e| match e {
            crate::chunk_plan::CompileError::Unsupported(e) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
            }
            crate::chunk_plan::CompileError::MissingPrimaryDims(e) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(e)
            }
        })?;

    let mut out: Vec<Py<PyAny>> = Vec::new();
    for idx in plan.into_index_iter() {
        let chunk_shape_nz = primary
            .chunk_shape(&idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let shape: Vec<u64> = chunk_shape_nz.iter().map(|x| x.get()).collect();
        let origin = primary
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
