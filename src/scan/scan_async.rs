use super::chunk_to_df::chunk_to_df;
use super::open_arrays::open_arrays_async;
use super::prelude::*;
use crate::IntoIStr;
use crate::chunk_plan::compile_expr_to_grouped_chunk_plan_async;
use zarrs::array_subset::ArraySubset;

pub(crate) async fn scan_zarr_df_async(
    store_input: StoreInput,
    expr: Expr,
    variables: Option<Vec<String>>,
    max_concurrency: Option<usize>,
    with_columns: Option<BTreeSet<String>>,
) -> Result<DataFrame, PyErr> {
    // Async open + async meta traversal.
    let (opened_async, zarr_meta) =
        open_and_load_zarr_meta_from_input_async(
            store_input,
        )
        .await
        .map_err(to_py_err)?;
    // Convert to ZarrDatasetMeta - preserves hierarchical paths from path_to_array
    let meta = ZarrDatasetMeta::from(&zarr_meta);

    // Convert from Python String to IStr at the boundary
    let vars: Vec<IStr> = variables
        .map(|v| {
            v.into_iter()
                .map(|s| s.istr())
                .collect()
        })
        .unwrap_or_else(|| {
            meta.data_vars.clone()
        });
    if vars.is_empty() {
        return Err(PyErr::new::<
            pyo3::exceptions::PyValueError,
            _,
        >(
            "no variables found/selected",
        ));
    }

    // Use dataset dims directly
    let dims = meta.dims.clone();

    // Open arrays once (async) for reading.
    let (var_arrays, coord_arrays) =
        open_arrays_async(
            opened_async.store.clone(),
            &meta,
            &vars,
            &dims,
        )
        .await
        .map_err(to_py_err)?;

    // Compile chunk plan using truly async I/O (concurrent coordinate resolution).
    let (grouped_plan, _stats) =
        match compile_expr_to_grouped_chunk_plan_async(
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
                    crate::chunk_plan::PlannerStats {
                        coord_reads: 0,
                    },
                )
            }
        };

    // Build per-variable chunk jobs (array, chunk index).
    let mut chunk_jobs: Vec<(
        Arc<
            Array<
                dyn zarrs::storage::AsyncReadableWritableListableStorageTraits,
            >,
        >,
        Vec<u64>,
    )> = Vec::new();
    for var in &vars {
        let primary = var_arrays
            .iter()
            .find(|(name, _)| name == var)
            .map(|(_, arr)| Arc::clone(arr))
            .ok_or_else(|| {
                PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >("unknown variable")
            })?;

        let mut var_chunk_indices: Vec<Vec<u64>> =
            Vec::new();
        
        // Determine if we should scan all chunks for this variable
        let should_scan_all = if let Some(subsets) =
            grouped_plan.get_plan(var.as_ref())
        {
            // If subsets is empty, it means "select all" (NoSelectionMade case)
            if subsets.subsets_iter().next().is_none() {
                true
            } else {
                // Use the subsets from the plan to select chunks
                for subset in subsets.subsets_iter() {
                    if let Ok(Some(chunks)) = primary
                        .chunks_in_array_subset(subset)
                    {
                        for chunk_idx in
                            chunks.indices().iter()
                        {
                            var_chunk_indices.push(
                                chunk_idx
                                    .iter()
                                    .copied()
                                    .collect(),
                            );
                        }
                    }
                }
                false
            }
        } else {
            // Variable not in plan or plan is empty - scan all
            grouped_plan.is_empty()
        };
        
        if should_scan_all {
            // Scan all chunks for this variable
            let shape = primary.shape().to_vec();
            let full = ArraySubset::new_with_start_shape(
                vec![0; shape.len()],
                shape,
            )
            .map_err(to_py_err)?;
            if let Ok(Some(chunks)) = primary
                .chunks_in_array_subset(&full)
            {
                for chunk_idx in
                    chunks.indices().iter()
                {
                    var_chunk_indices.push(
                        chunk_idx
                            .iter()
                            .copied()
                            .collect(),
                    );
                }
            }
        }

        for idx in var_chunk_indices {
            chunk_jobs.push((
                Arc::clone(&primary),
                idx,
            ));
        }
    }

    let max_conc = max_concurrency
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CONCURRENCY);
    let semaphore = Arc::new(
        tokio::sync::Semaphore::new(max_conc),
    );

    let meta = Arc::new(meta);
    let dims = Arc::new(dims);
    let vars = Arc::new(vars);
    let var_arrays = Arc::new(var_arrays);
    let coord_arrays = Arc::new(coord_arrays);
    // Convert with_columns to IStr
    let with_columns: Arc<
        Option<BTreeSet<IStr>>,
    > = Arc::new(with_columns.map(|s| {
        s.into_iter().map(|c| c.istr()).collect()
    }));

    let mut futs = FuturesUnordered::new();
    for (primary, idx) in chunk_jobs {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();
        let primary = Arc::clone(&primary);
        let meta = Arc::clone(&meta);
        let dims = Arc::clone(&dims);
        let vars = Arc::clone(&vars);
        let var_arrays = Arc::clone(&var_arrays);
        let coord_arrays =
            Arc::clone(&coord_arrays);
        let with_columns =
            Arc::clone(&with_columns);
        futs.push(async move {
            let _permit = permit;
            chunk_to_df(
                idx,
                primary,
                meta,
                dims,
                vars,
                var_arrays,
                coord_arrays,
                with_columns,
            )
            .await
        });
    }

    let mut out: Option<DataFrame> = None;
    while let Some(r) = futs.next().await {
        let df = r?;
        if let Some(acc) = &mut out {
            acc.vstack_mut(&df)
                .map_err(PyPolarsErr::from)?;
        } else {
            out = Some(df);
        }
    }

    Ok(out.unwrap_or_else(|| {
        DataFrame::new(0, vec![]).unwrap()
    }))
}
