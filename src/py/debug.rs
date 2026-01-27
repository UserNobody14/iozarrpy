//! Debug utilities for chunk planning inspection.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_async_runtimes::tokio::future_into_py;
use std::collections::HashMap;

use crate::chunk_plan::{
    DatasetSelection, collect_requests_with_meta,
    compile_expr_to_grouped_chunk_plan_async, compile_expr_to_lazy_selection,
    resolve_lazy_selection_async, resolve_lazy_selection_sync,
};
use crate::meta::{ZarrDatasetMeta, open_and_load_zarr_meta, open_and_load_zarr_meta_async};
use crate::py::expr_extract::extract_expr;
use crate::{IStr, IntoIStr};

#[pyfunction]
pub(crate) fn print_extension_info() -> String {
    "Rainbear extension module loaded successfully".to_string()
}

#[pyfunction]
pub(crate) fn _debug_expr_ast(predicate: &Bound<'_, PyAny>) -> PyResult<String> {
    let expr = extract_expr(predicate)?;
    Ok(format!("{expr:#?}"))
}

/// Comprehensive debug function for chunk planning.
///
/// Returns a dictionary with:
/// - `meta`: Dataset metadata including time encoding info for each array
/// - `dims`: List of dimensions
/// - `dim_lengths`: Map of dimension name to length
/// - `expr_ast`: Parsed expression AST
/// - `lazy_selection`: String representation of the lazy selection (before resolution)
/// - `resolution_requests`: List of resolution requests that will be made
/// - `resolution_results`: Map of request to result (after resolution)
/// - `materialized_selection`: The final materialized selection
/// - `coord_reads`: Number of coordinate array reads performed
/// - `error`: Any error that occurred (if applicable)
#[pyfunction]
#[pyo3(signature = (zarr_url, predicate, primary_var=None))]
pub(crate) fn _debug_chunk_planning(
    py: Python<'_>,
    zarr_url: String,
    predicate: &Bound<'_, PyAny>,
    primary_var: Option<String>,
) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);

    // Open the store and load metadata
    let (opened, zarr_meta) = match open_and_load_zarr_meta(&zarr_url) {
        Ok(x) => x,
        Err(e) => {
            result.set_item("error", format!("Failed to open store: {}", e))?;
            return Ok(result.into());
        }
    };
    // Convert to ZarrDatasetMeta - preserves hierarchical paths from path_to_array
    let meta = ZarrDatasetMeta::from(&zarr_meta);

    // Add metadata info
    let meta_dict = PyDict::new(py);
    for (name, arr_meta) in &meta.arrays {
        let arr_dict = PyDict::new(py);
        arr_dict.set_item("path", arr_meta.path.to_string())?;
        arr_dict.set_item("shape", arr_meta.shape.to_vec())?;
        let dims_strs: Vec<String> = arr_meta.dims.iter().map(|d| d.to_string()).collect();
        arr_dict.set_item("dims", &dims_strs)?;
        arr_dict.set_item("polars_dtype", format!("{:?}", arr_meta.polars_dtype))?;

        if let Some(te) = &arr_meta.time_encoding {
            let te_dict = PyDict::new(py);
            te_dict.set_item("epoch_ns", te.epoch_ns)?;
            te_dict.set_item("unit_ns", te.unit_ns)?;
            te_dict.set_item("is_duration", te.is_duration)?;
            arr_dict.set_item("time_encoding", te_dict)?;
        } else {
            arr_dict.set_item("time_encoding", py.None())?;
        }

        meta_dict.set_item(name.to_string(), arr_dict)?;
    }
    result.set_item("meta", meta_dict)?;
    let dims_strs: Vec<String> = meta.dims.iter().map(|d| d.to_string()).collect();
    result.set_item("dims", &dims_strs)?;
    let data_vars_strs: Vec<String> = meta.data_vars.iter().map(|d| d.to_string()).collect();
    result.set_item("data_vars", &data_vars_strs)?;

    // Extract expression
    let expr = match extract_expr(predicate) {
        Ok(e) => e,
        Err(e) => {
            result.set_item("error", format!("Failed to extract expression: {}", e))?;
            return Ok(result.into());
        }
    };
    result.set_item("expr_ast", format!("{:#?}", expr))?;

    // Determine primary variable (convert String -> IStr at boundary)
    let pvar: IStr = primary_var
        .map(|s| s.istr())
        .unwrap_or_else(|| meta.data_vars.first().cloned().unwrap_or_else(|| "".istr()));
    result.set_item("primary_var", pvar.to_string())?;

    let Some(primary_meta) = meta.arrays.get(&pvar) else {
        result.set_item("error", format!("Primary variable '{}' not found", pvar))?;
        return Ok(result.into());
    };

    let dims: Vec<IStr> = if !primary_meta.dims.is_empty() {
        primary_meta.dims.iter().cloned().collect()
    } else {
        meta.dims.clone()
    };

    let dim_lengths: Vec<u64> = if primary_meta.shape.len() == dims.len() {
        primary_meta.shape.to_vec()
    } else {
        result.set_item(
            "error",
            format!(
                "Primary variable shape {:?} doesn't match dims {:?}",
                primary_meta.shape, dims
            ),
        )?;
        return Ok(result.into());
    };

    // Create dimension length map (convert IStr -> String for Python)
    let dim_lengths_map: HashMap<String, u64> = dims
        .iter()
        .zip(dim_lengths.iter())
        .map(|(d, l)| (d.to_string(), *l))
        .collect();
    result.set_item("dim_lengths", dim_lengths_map)?;

    // Compile to lazy selection
    let lazy_selection = match compile_expr_to_lazy_selection(&expr, &meta) {
        Ok(sel) => sel,
        Err(e) => {
            result.set_item("error", format!("Failed to compile expression: {:?}", e))?;
            return Ok(result.into());
        }
    };
    result.set_item("lazy_selection", format!("{:#?}", lazy_selection))?;

    // Collect resolution requests
    let (requests, immediate_cache) =
        collect_requests_with_meta(&lazy_selection, &meta, &dim_lengths, &dims);

    let request_strs: Vec<String> = requests.iter().map(|r| format!("{:?}", r)).collect();
    result.set_item("resolution_requests", request_strs)?;
    result.set_item("num_requests", requests.len())?;

    // Show immediate cache (index-only dims resolved immediately)
    let immediate_cache_str = format!("{:?}", immediate_cache);
    result.set_item("immediate_cache", immediate_cache_str)?;

    // Resolve the lazy selection
    let (materialized, stats) =
        match resolve_lazy_selection_sync(&lazy_selection, &meta, opened.store.clone()) {
            Ok(x) => x,
            Err(e) => {
                result.set_item(
                    "error",
                    format!("Failed to resolve lazy selection: {:?}", e),
                )?;
                return Ok(result.into());
            }
        };

    result.set_item("materialized_selection", format!("{:#?}", materialized))?;
    result.set_item("coord_reads", stats.coord_reads)?;

    // Check if the selection is empty or "no selection made" (which means all chunks)
    let selection_summary = match &materialized {
        DatasetSelection::NoSelectionMade => "NoSelectionMade (will select ALL chunks)".to_string(),
        DatasetSelection::Empty => "Empty (will select NO chunks)".to_string(),
        DatasetSelection::Selection(sel) => {
            format!("Selection with {} variables", sel.len())
        }
    };
    result.set_item("selection_summary", selection_summary)?;

    Ok(result.into())
}

/// Async version of chunk planning debug.
///
/// Tests the async resolution path to diagnose issues with async chunk planning.
#[pyfunction]
#[pyo3(signature = (zarr_url, predicate, primary_var=None))]
pub(crate) fn _debug_chunk_planning_async(
    py: Python<'_>,
    zarr_url: String,
    predicate: &Bound<'_, PyAny>,
    primary_var: Option<String>,
) -> PyResult<Py<PyAny>> {
    // Extract expression under the GIL
    let expr = extract_expr(predicate)?;

    let awaitable = future_into_py(py, async move {
        _debug_chunk_planning_async_inner(zarr_url, expr, primary_var).await
    })?;

    Ok(awaitable.unbind())
}

async fn _debug_chunk_planning_async_inner(
    zarr_url: String,
    expr: polars::prelude::Expr,
    primary_var: Option<String>,
) -> PyResult<HashMap<String, String>> {
    let mut result: HashMap<String, String> = HashMap::new();

    // Open the store async and load metadata
    let (opened_async, zarr_meta) = match open_and_load_zarr_meta_async(&zarr_url).await {
        Ok(x) => x,
        Err(e) => {
            result.insert("error".to_string(), format!("Failed to open store: {}", e));
            return Ok(result);
        }
    };
    // Convert to ZarrDatasetMeta - preserves hierarchical paths from path_to_array
    let meta = ZarrDatasetMeta::from(&zarr_meta);

    result.insert("meta_loaded".to_string(), "true".to_string());
    result.insert("dims".to_string(), format!("{:?}", meta.dims));
    result.insert("data_vars".to_string(), format!("{:?}", meta.data_vars));

    // Determine primary variable (convert String -> IStr at boundary)
    let pvar: IStr = primary_var
        .map(|s| s.istr())
        .unwrap_or_else(|| meta.data_vars.first().cloned().unwrap_or_else(|| "".istr()));
    result.insert("primary_var".to_string(), pvar.to_string());

    let Some(primary_meta) = meta.arrays.get(&pvar) else {
        result.insert(
            "error".to_string(),
            format!("Primary variable '{}' not found", pvar),
        );
        return Ok(result);
    };

    let dims: Vec<IStr> = if !primary_meta.dims.is_empty() {
        primary_meta.dims.iter().cloned().collect()
    } else {
        meta.dims.clone()
    };
    result.insert("primary_dims".to_string(), format!("{:?}", dims));
    result.insert(
        "primary_shape".to_string(),
        format!("{:?}", primary_meta.shape),
    );

    // Compile to lazy selection (no I/O)
    let lazy_selection = match compile_expr_to_lazy_selection(&expr, &meta) {
        Ok(sel) => sel,
        Err(e) => {
            result.insert(
                "error".to_string(),
                format!("Failed to compile expression: {:?}", e),
            );
            return Ok(result);
        }
    };
    result.insert(
        "lazy_selection".to_string(),
        format!("{:#?}", lazy_selection),
    );
    result.insert("lazy_compilation_success".to_string(), "true".to_string());

    // Now test async resolution
    result.insert("starting_async_resolution".to_string(), "true".to_string());

    let start = std::time::Instant::now();
    let resolve_result =
        resolve_lazy_selection_async(&lazy_selection, &meta, opened_async.store.clone()).await;
    let elapsed = start.elapsed();
    result.insert(
        "async_resolution_time_ms".to_string(),
        format!("{}", elapsed.as_millis()),
    );

    match resolve_result {
        Ok((selection, stats)) => {
            result.insert("async_resolution_success".to_string(), "true".to_string());
            result.insert("coord_reads".to_string(), format!("{}", stats.coord_reads));

            let selection_summary = match &selection {
                DatasetSelection::NoSelectionMade => "NoSelectionMade".to_string(),
                DatasetSelection::Empty => "Empty".to_string(),
                DatasetSelection::Selection(sel) => {
                    format!("Selection with {} variables", sel.len())
                }
            };
            result.insert("selection_summary".to_string(), selection_summary);
            result.insert(
                "materialized_selection".to_string(),
                format!("{:#?}", selection),
            );
        }
        Err(e) => {
            result.insert("async_resolution_success".to_string(), "false".to_string());
            result.insert("async_resolution_error".to_string(), format!("{:?}", e));
        }
    }

    // Also test the full chunk plan compilation
    result.insert("starting_chunk_plan".to_string(), "true".to_string());

    let start = std::time::Instant::now();
    let plan_result =
        compile_expr_to_grouped_chunk_plan_async(&expr, &meta, opened_async.store.clone()).await;
    let elapsed = start.elapsed();
    result.insert(
        "chunk_plan_time_ms".to_string(),
        format!("{}", elapsed.as_millis()),
    );

    match plan_result {
        Ok((plan, stats)) => {
            result.insert("chunk_plan_success".to_string(), "true".to_string());
            result.insert(
                "chunk_plan_coord_reads".to_string(),
                format!("{}", stats.coord_reads),
            );

            // Report grouped chunk plan summary
            result.insert("num_grids".to_string(), format!("{}", plan.num_grids()));
            result.insert("num_vars".to_string(), format!("{}", plan.num_vars()));
            result.insert("total_chunks".to_string(), format!("{}", plan.total_chunks()));

            // List variables per grid signature
            let mut grid_info: Vec<String> = Vec::new();
            for (sig, vars, subsets) in plan.iter_grids() {
                let var_names: Vec<&str> = vars.iter().map(|v| v.as_ref()).collect();
                let num_subsets = subsets.subsets_iter().count();
                grid_info.push(format!(
                    "dims={:?}, vars={:?}, subsets={}",
                    sig.dims(),
                    var_names,
                    num_subsets
                ));
            }
            result.insert("grids".to_string(), format!("{:?}", grid_info));
        }
        Err(e) => {
            result.insert("chunk_plan_success".to_string(), "false".to_string());
            result.insert("chunk_plan_error".to_string(), format!("{:?}", e));
        }
    }

    Ok(result)
}

/// Debug function to inspect a coordinate array's values and time encoding.
///
/// Returns a dictionary with:
/// - `dim_name`: The dimension name
/// - `shape`: The array shape
/// - `time_encoding`: Time encoding info (if present)
/// - `sample_raw_values`: First and last few raw values from the array
/// - `sample_decoded_values`: The same values after time encoding is applied
#[pyfunction]
#[pyo3(signature = (zarr_url, dim_name, num_samples=5))]
pub(crate) fn _debug_coord_array(
    py: Python<'_>,
    zarr_url: String,
    dim_name: String,
    num_samples: Option<usize>,
) -> PyResult<Py<PyDict>> {
    use zarrs::array::Array;

    let result = PyDict::new(py);
    let num_samples = num_samples.unwrap_or(5);

    // Open the store and load metadata
    let (opened, zarr_meta) = match open_and_load_zarr_meta(&zarr_url) {
        Ok(x) => x,
        Err(e) => {
            result.set_item("error", format!("Failed to open store: {}", e))?;
            return Ok(result.into());
        }
    };
    // Convert to ZarrDatasetMeta - preserves hierarchical paths from path_to_array
    let meta = ZarrDatasetMeta::from(&zarr_meta);

    let dim_key = dim_name.as_str().istr();
    let Some(arr_meta) = meta.arrays.get(&dim_key) else {
        result.set_item("error", format!("Array '{}' not found", dim_name))?;
        return Ok(result.into());
    };

    result.set_item("dim_name", &dim_name)?;
    result.set_item("shape", arr_meta.shape.to_vec())?;
    let dims_strs: Vec<String> = arr_meta.dims.iter().map(|d| d.to_string()).collect();
    result.set_item("dims", &dims_strs)?;
    result.set_item("path", arr_meta.path.to_string())?;

    if let Some(te) = &arr_meta.time_encoding {
        let te_dict = PyDict::new(py);
        te_dict.set_item("epoch_ns", te.epoch_ns)?;
        te_dict.set_item("unit_ns", te.unit_ns)?;
        te_dict.set_item("is_duration", te.is_duration)?;

        // Add human-readable info
        let epoch_datetime = chrono::DateTime::from_timestamp_nanos(te.epoch_ns);
        te_dict.set_item("epoch_datetime", format!("{}", epoch_datetime))?;

        let unit_human = match te.unit_ns {
            1 => "nanoseconds",
            1_000 => "microseconds",
            1_000_000 => "milliseconds",
            1_000_000_000 => "seconds",
            60_000_000_000 => "minutes",
            3_600_000_000_000 => "hours",
            86_400_000_000_000 => "days",
            _ => "unknown",
        };
        te_dict.set_item("unit_human", unit_human)?;

        result.set_item("time_encoding", te_dict)?;
    } else {
        result.set_item("time_encoding", py.None())?;
    }

    // Open the array and read sample values
    let arr = match Array::open(opened.store.clone(), arr_meta.path.as_ref()) {
        Ok(a) => a,
        Err(e) => {
            result.set_item("error", format!("Failed to open array: {}", e))?;
            return Ok(result.into());
        }
    };

    let dtype_id = arr.data_type().identifier();
    result.set_item("dtype", dtype_id)?;

    if arr_meta.shape.len() != 1 {
        result.set_item("note", "Array is not 1D, skipping value sampling")?;
        return Ok(result.into());
    }

    let n = arr_meta.shape[0] as usize;
    if n == 0 {
        result.set_item("note", "Array is empty")?;
        return Ok(result.into());
    }

    // Sample indices: first num_samples and last num_samples
    let mut sample_indices: Vec<usize> = Vec::new();
    for i in 0..num_samples.min(n) {
        sample_indices.push(i);
    }
    if n > num_samples {
        for i in (n - num_samples)..n {
            if !sample_indices.contains(&i) {
                sample_indices.push(i);
            }
        }
    }
    sample_indices.sort();

    result.set_item("sample_indices", &sample_indices)?;

    // Read values based on dtype
    match dtype_id {
        "int64" => {
            let mut raw_values: Vec<i64> = Vec::new();
            let mut decoded_values: Vec<i64> = Vec::new();

            for &idx in &sample_indices {
                let start = [idx as u64];
                let shape = [1u64];
                match arr.retrieve_array_subset_ndarray::<i64>(
                    &zarrs::array_subset::ArraySubset::new_with_start_shape(
                        start.to_vec(),
                        shape.to_vec(),
                    )
                    .unwrap(),
                ) {
                    Ok(data) => {
                        if let Some(&v) = data.as_slice().and_then(|s| s.first()) {
                            raw_values.push(v);
                            if let Some(te) = &arr_meta.time_encoding {
                                decoded_values.push(te.decode(v));
                            } else {
                                decoded_values.push(v);
                            }
                        }
                    }
                    Err(e) => {
                        result.set_item("sample_error", format!("Failed to read: {}", e))?;
                        break;
                    }
                }
            }

            result.set_item("sample_raw_values", raw_values)?;

            // If we have time encoding, show decoded as datetime strings
            if arr_meta.time_encoding.is_some() {
                let decoded_datetime_strs: Vec<String> = decoded_values
                    .iter()
                    .map(|&ns| format!("{}", chrono::DateTime::from_timestamp_nanos(ns)))
                    .collect();
                result.set_item("sample_decoded_datetimes", decoded_datetime_strs)?;
            }
            result.set_item("sample_decoded_values", decoded_values)?;
        }
        _ => {
            result.set_item(
                "note",
                format!("Value sampling not implemented for dtype '{}'", dtype_id),
            )?;
        }
    }

    Ok(result.into())
}

/// Debug function to test literal conversion.
///
/// Shows how a Python value would be converted to a CoordScalar for comparison.
#[pyfunction]
pub(crate) fn _debug_literal_conversion(
    py: Python<'_>,
    zarr_url: String,
    dim_name: String,
    test_value: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    #![allow(unused_imports)]
    use polars::prelude::*;

    let result = PyDict::new(py);

    // Open the store and load metadata
    let (_, zarr_meta) = match open_and_load_zarr_meta(&zarr_url) {
        Ok(x) => x,
        Err(e) => {
            result.set_item("error", format!("Failed to open store: {}", e))?;
            return Ok(result.into());
        }
    };
    // Convert to ZarrDatasetMeta - preserves hierarchical paths from path_to_array
    let meta = ZarrDatasetMeta::from(&zarr_meta);

    let dim_key = dim_name.as_str().istr();
    let time_encoding = meta
        .arrays
        .get(&dim_key)
        .and_then(|a| a.time_encoding.as_ref());

    result.set_item("dim_name", &dim_name)?;

    if let Some(te) = time_encoding {
        let te_dict = PyDict::new(py);
        te_dict.set_item("epoch_ns", te.epoch_ns)?;
        te_dict.set_item("unit_ns", te.unit_ns)?;
        te_dict.set_item("is_duration", te.is_duration)?;
        result.set_item("time_encoding", te_dict)?;
    } else {
        result.set_item("time_encoding", py.None())?;
    }

    // Try to extract the value through the expression extraction path
    // Build a simple comparison expression: pl.col(dim_name) == test_value
    let polars_mod = py.import("polars")?;
    let col_fn = polars_mod.getattr("col")?;
    let col_expr = col_fn.call1((&dim_name,))?;
    let eq_expr = col_expr.call_method1("__eq__", (test_value,))?;

    result.set_item("test_expr_repr", format!("{}", eq_expr))?;

    // Extract the expression
    match extract_expr(&eq_expr) {
        Ok(expr) => {
            result.set_item("extracted_expr", format!("{:#?}", expr))?;

            // Try to compile it to a lazy selection
            match compile_expr_to_lazy_selection(&expr, &meta) {
                Ok(sel) => {
                    result.set_item("lazy_selection", format!("{:#?}", sel))?;
                    result.set_item("compilation_success", true)?;
                }
                Err(e) => {
                    result.set_item("compilation_error", format!("{:?}", e))?;
                    result.set_item("compilation_success", false)?;
                }
            }
        }
        Err(e) => {
            result.set_item("extraction_error", format!("{}", e))?;
        }
    }

    Ok(result.into())
}
