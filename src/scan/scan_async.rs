use super::chunk_to_df::chunk_to_df;
use super::open_arrays::open_arrays_async;
use super::prelude::*;
use crate::IntoIStr;
use crate::chunk_plan::compile_expr_to_grouped_chunk_plan_async;

pub(crate) async fn scan_zarr_df_async(
    store_input: StoreInput,
    expr: Expr,
    variables: Option<Vec<String>>,
    max_concurrency: Option<usize>,
    with_columns: Option<BTreeSet<String>>,
) -> Result<DataFrame, PyErr> {
    // Async open + async meta traversal.
    let (opened_async, zarr_meta) =
        open_and_load_zarr_meta_from_input_async(store_input).await.map_err(to_py_err)?;
    // Convert to ZarrDatasetMeta - preserves hierarchical paths from path_to_array
    let meta = ZarrDatasetMeta::from(&zarr_meta);

    // Convert from Python String to IStr at the boundary
    let vars: Vec<IStr> = variables
        .map(|v| v.into_iter().map(|s| s.istr()).collect())
        .unwrap_or_else(|| meta.data_vars.clone());
    if vars.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "no variables found/selected",
        ));
    }

    // Use dataset dims directly
    let dims = meta.dims.clone();

    // Open arrays once (async) for reading.
    let (var_arrays, coord_arrays) =
        open_arrays_async(opened_async.store.clone(), &meta, &vars, &dims)
            .await
            .map_err(to_py_err)?;

    // Compile chunk plan using truly async I/O (concurrent coordinate resolution).
    let (grouped_plan, _stats) = match compile_expr_to_grouped_chunk_plan_async(
        &expr,
        &meta,
        opened_async.store.clone(),
    )
    .await
    {
        Ok(x) => x,
        Err(_) => {
            // Fall back to empty plan - will scan no chunks
            (
                GroupedChunkPlan::new(),
                crate::chunk_plan::PlannerStats { coord_reads: 0 },
            )
        }
    };

    // Pick a reference variable for chunk iteration geometry
    // TODO: Eventually iterate per-grid, but for now pick the first variable
    let ref_var = &vars[0];
    let ref_meta = meta.arrays.get(ref_var).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown variable")
    })?;
    let ref_array = zarrs::array::Array::async_open(opened_async.store.clone(), ref_meta.path.as_ref())
        .await
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let ref_array = Arc::new(ref_array);

    // Convert array subsets to chunk indices for the reference variable
    let mut chunk_indices: Vec<Vec<u64>> = Vec::new();
    
    // Get the plan for the reference variable, or scan all chunks if not in plan
    if let Some(subsets) = grouped_plan.get_plan(ref_var.as_ref()) {
        for subset in subsets.subsets_iter() {
            if let Ok(Some(chunks)) = ref_array.chunks_in_array_subset(subset) {
                for chunk_idx in chunks.indices().iter() {
                    chunk_indices.push(chunk_idx.iter().copied().collect());
                }
            }
        }
    } else if grouped_plan.is_empty() {
        // No selection made - scan all chunks
        let grid_shape = ref_array.chunk_grid().grid_shape();
        // Generate all chunk indices
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
    }

    let max_conc = max_concurrency
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CONCURRENCY);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_conc));

    let meta = Arc::new(meta);
    let dims = Arc::new(dims);
    let vars = Arc::new(vars);
    let var_arrays = Arc::new(var_arrays);
    let coord_arrays = Arc::new(coord_arrays);
    // Convert with_columns to IStr
    let with_columns: Arc<Option<BTreeSet<IStr>>> = Arc::new(
        with_columns.map(|s| s.into_iter().map(|c| c.istr()).collect())
    );

    let mut futs = FuturesUnordered::new();
    for idx in chunk_indices {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let ref_array = ref_array.clone();
        let meta = Arc::clone(&meta);
        let dims = Arc::clone(&dims);
        let vars = Arc::clone(&vars);
        let var_arrays = Arc::clone(&var_arrays);
        let coord_arrays = Arc::clone(&coord_arrays);
        let with_columns = Arc::clone(&with_columns);
        futs.push(async move {
            let _permit = permit;
            chunk_to_df(idx, ref_array, meta, dims, vars, var_arrays, coord_arrays, with_columns).await
        });
    }

    let mut out: Option<DataFrame> = None;
    while let Some(r) = futs.next().await {
        let df = r?;
        if let Some(acc) = &mut out {
            acc.vstack_mut(&df).map_err(PyPolarsErr::from)?;
        } else {
            out = Some(df);
        }
    }

    Ok(out.unwrap_or_else(|| DataFrame::new(0, vec![]).unwrap()))
}
