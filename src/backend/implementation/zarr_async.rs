use std::sync::Arc;

use pyo3::PyErr;

use crate::scan::async_scan::chunk_to_df_from_grid_with_backend;
use crate::shared::ChunkedExpressionCompilerAsync;
use crate::shared::FullyCachedZarrBackendAsync;
use crate::shared::HasMetadataBackendAsync;
use crate::shared::{
    combine_chunk_dataframes,
    restructure_to_structs,
};
use crate::{IStr, IntoIStr};

/// Internal: Async scan using the backend.
///
/// This uses the backend's cached metadata and chunk reading directly.
pub(crate) async fn scan_zarr_with_backend_async(
    backend: Arc<FullyCachedZarrBackendAsync>,
    expr: polars::prelude::Expr,
    max_concurrency: Option<usize>,
    max_chunks_to_read: Option<usize>,
) -> Result<polars::prelude::DataFrame, PyErr> {
    use futures::stream::{
        FuturesUnordered, StreamExt,
    };
    use std::sync::Arc as StdArc;

    const DEFAULT_MAX_CONCURRENCY: usize = 32;
    let meta = backend.metadata().await?;

    let planning_meta =
        StdArc::new(meta.planning_meta());

    // Compile grouped chunk plan using backend-based resolver
    let (grouped_plan, _stats) = backend
        .clone()
        .compile_expression_async(&expr)
        .await?;

    // Check max_chunks_to_read limit before doing any I/O
    if let Some(max_chunks) = max_chunks_to_read {
        let total_chunks = grouped_plan
            .total_unique_chunks()
            .map_err(|e| {
                PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >(e)
            })?;
        if total_chunks > max_chunks {
            return Err(PyErr::new::<
                pyo3::exceptions::PyRuntimeError,
                _,
            >(format!(
                "max_chunks_to_read exceeded: {} chunks needed, limit is {}",
                total_chunks, max_chunks
            )));
        }
    }

    let max_conc = max_concurrency
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CONCURRENCY);
    let semaphore = StdArc::new(
        tokio::sync::Semaphore::new(max_conc),
    );

    // Read chunks using consolidated (deduplicated) iteration
    let mut futs = FuturesUnordered::new();
    for group in
        grouped_plan.iter_consolidated_chunks()
    {
        let group = group.map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e)
        })?;
        let vars: Vec<IStr> = group
            .vars
            .iter()
            .map(|v| v.istr())
            .collect();

        for idx in group.chunk_indices {
            let sem = semaphore.clone();
            let backend = backend.clone();
            let sig = group.sig.clone();
            let array_shape =
                group.array_shape.clone();
            let vars = vars.clone();

            futs.push(async move {
                // Acquire permit inside the future - this ensures
                // permits are only acquired when the future is polled,
                // enabling proper pipelining instead of batch execution
                let _permit = sem
                    .acquire_owned()
                    .await
                    .expect("semaphore closed");
                chunk_to_df_from_grid_with_backend(
                    backend.as_ref(),
                    idx.into(),
                    &sig,
                    &array_shape,
                    &vars,
                    None,
                )
                .await
            });
        }
    }

    // Collect all chunk DataFrames
    let mut dfs: Vec<polars::prelude::DataFrame> =
        Vec::new();
    while let Some(r) = futs.next().await {
        let df = r?;
        dfs.push(df);
    }

    // Combine all chunk DataFrames
    let result = if dfs.is_empty() {
        let keys: Vec<IStr> = grouped_plan
            .var_to_grid()
            .keys()
            .cloned()
            .collect();
        polars::prelude::DataFrame::empty_with_schema(
            &planning_meta.tidy_schema(Some(
                keys.as_slice(),
            )),
        )
    } else if dfs.len() == 1 {
        dfs.into_iter().next().unwrap()
    } else {
        combine_chunk_dataframes(dfs, &meta)?
    };

    // For hierarchical data, convert flat path columns to struct columns
    if meta.is_hierarchical() {
        restructure_to_structs(&result, &meta)
    } else {
        Ok(result)
    }
}
