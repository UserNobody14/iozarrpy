use pyo3::wrap_pyfunction;



mod zarr_meta;
mod zarr_source;
mod zarr_store;
mod chunk_plan;
mod zarr_scan_async;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_polars::PyExpr;
use zarrs::array::Array;

use crate::zarr_source::ZarrSource;
use crate::zarr_store::open_store;
use crate::zarr_meta::load_dataset_meta_from_opened;
use crate::chunk_plan::{compile_expr_to_chunk_plan, ChunkPlan};
use crate::zarr_scan_async::scan_zarr_async;

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from rainbear!".to_string()
}

#[pyfunction]
#[pyo3(signature = (zarr_url, predicate, variables=None))]
fn selected_chunks(py: Python<'_>, zarr_url: String, predicate: &Bound<'_, PyAny>, variables: Option<Vec<String>>) -> PyResult<Vec<Py<PyAny>>> {
    let opened = open_store(&zarr_url).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let meta = load_dataset_meta_from_opened(&opened)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let vars = variables.unwrap_or_else(|| meta.data_vars.clone());
    if vars.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("no variables found/selected"));
    }
    let primary_var = &vars[0];

    let expr = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let pyexpr: PyExpr = predicate.extract()?;
        Ok::<polars::prelude::Expr, PyErr>(pyexpr.0.clone())
    }))
    .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("panic while converting predicate Expr"))??;

    let primary_meta = meta
        .arrays
        .get(primary_var)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown primary variable"))?;
    let primary = Array::open(opened.store.clone(), &primary_meta.path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let (plan, _stats) = compile_expr_to_chunk_plan(&expr, &meta, opened.store.clone(), primary_var)
        .unwrap_or_else(|_| {
            let grid_shape = primary.chunk_grid().grid_shape().to_vec();
            let zero = vec![0u64; primary.dimensionality()];
            let regular_chunk_shape = primary
                .chunk_shape(&zero)
                .map(|v| v.iter().map(|x| x.get()).collect::<Vec<u64>>())
                .unwrap_or_else(|_| vec![1; primary.dimensionality()]);
            let dims = primary_meta.dims.clone();
            (ChunkPlan::all(dims, grid_shape, regular_chunk_shape), crate::chunk_plan::PlannerStats { coord_reads: 0 })
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

        let d = pyo3::types::PyDict::new(py);
        d.set_item("indices", idx)?;
        d.set_item("origin", origin)?;
        d.set_item("shape", shape)?;
        out.push(d.into());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (zarr_url, predicate, variables=None))]
fn _selected_chunks_debug(py: Python<'_>, zarr_url: String, predicate: &Bound<'_, PyAny>, variables: Option<Vec<String>>) -> PyResult<(Vec<Py<PyAny>>, u64)> {
    let opened = open_store(&zarr_url).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let meta = load_dataset_meta_from_opened(&opened)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let vars = variables.unwrap_or_else(|| meta.data_vars.clone());
    if vars.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("no variables found/selected"));
    }
    let primary_var = &vars[0];

    let expr = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let pyexpr: PyExpr = predicate.extract()?;
        Ok::<polars::prelude::Expr, PyErr>(pyexpr.0.clone())
    }))
    .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("panic while converting predicate Expr"))??;

    let primary_meta = meta
        .arrays
        .get(primary_var)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown primary variable"))?;
    let primary = Array::open(opened.store.clone(), &primary_meta.path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let (plan, stats) = compile_expr_to_chunk_plan(&expr, &meta, opened.store.clone(), primary_var)
        .map_err(|e| match e {
            crate::chunk_plan::CompileError::Unsupported(e) => {
                return PyErr::new::<pyo3::exceptions::PyValueError, _>(e);
            }
            crate::chunk_plan::CompileError::MissingPrimaryDims(e) => {
                return PyErr::new::<pyo3::exceptions::PyValueError, _>(e);
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

        let d = pyo3::types::PyDict::new(py);
        d.set_item("indices", idx)?;
        d.set_item("origin", origin)?;
        d.set_item("shape", shape)?;
        out.push(d.into());
    }
    Ok((out, stats.coord_reads))
}

#[pyfunction]
fn _debug_expr_ast(predicate: &Bound<'_, PyAny>) -> PyResult<String> {
    let expr = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let pyexpr: PyExpr = predicate.extract()?;
        Ok::<polars::prelude::Expr, PyErr>(pyexpr.0.clone())
    }))
    .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("panic while converting predicate Expr"))??;
    Ok(format!("{expr:?}"))
}


#[pymodule]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_bin, m)?)?;
    m.add_function(wrap_pyfunction!(selected_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(_selected_chunks_debug, m)?)?;
    m.add_function(wrap_pyfunction!(_debug_expr_ast, m)?)?;
    m.add_function(wrap_pyfunction!(scan_zarr_async, m)?)?;

    m.add_class::<ZarrSource>()?;

    Ok(())
}