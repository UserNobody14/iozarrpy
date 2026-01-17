use super::chunk_to_df::chunk_to_df;
use super::open_arrays::open_arrays_async;
use super::prelude::*;

pub(crate) async fn scan_zarr_df_async(
    zarr_url: String,
    expr: Expr,
    variables: Option<Vec<String>>,
    max_concurrency: Option<usize>,
    with_columns: Option<BTreeSet<String>>,
) -> Result<DataFrame, PyErr> {
    // Async open + async meta traversal.
    let (opened_async, meta) =
        open_and_load_dataset_meta_async(&zarr_url).await.map_err(to_py_err)?;

    let vars = variables.unwrap_or_else(|| meta.data_vars.clone());
    if vars.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "no variables found/selected",
        ));
    }

    let primary_var = &vars[0];
    let primary_meta = meta.arrays.get(primary_var).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown primary variable")
    })?;
    let dims = if !primary_meta.dims.is_empty() {
        primary_meta.dims.clone()
    } else {
        meta.dims.clone()
    };

    // Open arrays once (async) for reading.
    let (primary, var_arrays, coord_arrays) =
        open_arrays_async(opened_async.store.clone(), &meta, &vars, &dims)
            .await
            .map_err(to_py_err)?;

    // Compile chunk plan off-thread using the existing (sync) planner.
    // This keeps behavior identical to the current predicate pushdown planning.
    let zarr_url_plan = zarr_url.clone();
    let meta_plan = meta.clone();
    let expr_plan = expr.clone();
    let primary_var_plan = primary_var.to_string();
    let (plan, _stats) = tokio::task::spawn_blocking(move || -> Result<(ChunkPlan, crate::chunk_plan::PlannerStats), PyErr> {
        let opened_sync = open_store(&zarr_url_plan).map_err(to_py_err)?;
        match compile_expr_to_chunk_plan(
            &expr_plan,
            &meta_plan,
            opened_sync.store.clone(),
            &primary_var_plan,
        ) {
            Ok(x) => Ok(x),
            Err(_) => {
                // Fall back to scanning all chunks if planning fails.
                let arr = Array::open(
                    opened_sync.store.clone(),
                    &meta_plan.arrays[&primary_var_plan].path,
                )
                .map_err(to_py_err)?;
                let grid_shape = arr.chunk_grid().grid_shape().to_vec();
                Ok((
                    ChunkPlan::all(grid_shape),
                    crate::chunk_plan::PlannerStats { coord_reads: 0 },
                ))
            }
        }
    })
    .await
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "chunk planner join error: {e}"
        ))
    })??;

    let indices: Vec<Vec<u64>> = plan.into_index_iter().collect();

    let max_conc = max_concurrency
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CONCURRENCY);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_conc));

    let meta = Arc::new(meta);
    let dims = Arc::new(dims);
    let vars = Arc::new(vars);
    let var_arrays = Arc::new(var_arrays);
    let coord_arrays = Arc::new(coord_arrays);
    let with_columns = Arc::new(with_columns);

    let mut futs = FuturesUnordered::new();
    for idx in indices {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let primary = primary.clone();
        let meta = Arc::clone(&meta);
        let dims = Arc::clone(&dims);
        let vars = Arc::clone(&vars);
        let var_arrays = Arc::clone(&var_arrays);
        let coord_arrays = Arc::clone(&coord_arrays);
        let with_columns = Arc::clone(&with_columns);
        futs.push(async move {
            let _permit = permit;
            chunk_to_df(idx, primary, meta, dims, vars, var_arrays, coord_arrays, with_columns).await
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
