use std::sync::Arc;

use snafu::ensure;

use crate::shared::IStr;
use crate::backend::implementation::iterating_common::expr_top_literal_bool;
use crate::chunk_plan::{
    GridGroupExecutionOpts, streaming_grid_chunk_read_count,
};
use crate::errors::BackendError;
use crate::errors::MaxChunksToReadExceededSnafu;
use crate::meta::ZarrMeta;
use crate::scan::async_scan::chunk_to_df_from_grid_with_backend;
use crate::shared::ChunkedDataBackendAsync;
use crate::shared::ChunkedExpressionCompilerAsync;
use crate::shared::HasMetadataBackendAsync;
use crate::shared::{
    combine_chunk_dataframes,
    restructure_to_structs,
};

/// Internal: Async scan using the backend.
///
/// This uses the backend's cached metadata and chunk reading directly.
/// Internal: Async scan using any backend that implements the required traits.
pub(crate) async fn scan_with_backend_async<B>(
    backend: Arc<B>,
    expr: polars::prelude::Expr,
    max_concurrency: Option<usize>,
    max_chunks_to_read: Option<usize>,
) -> Result<
    polars::prelude::DataFrame,
    BackendError,
>
where
    B: ChunkedDataBackendAsync
        + HasMetadataBackendAsync<ZarrMeta>
        + ChunkedExpressionCompilerAsync
        + Send
        + Sync,
{
    use futures::stream::{
        FuturesUnordered, StreamExt,
    };
    use std::sync::Arc as StdArc;

    const DEFAULT_MAX_CONCURRENCY: usize = 32;
    let meta = backend.metadata().await?;

    // Compile grouped chunk plan using backend-based resolver
    let (grouped_plan, _stats) = backend
        .clone()
        .compile_expression_async(&expr)
        .await?;

    let emit_empty_schema_once =
        expr_top_literal_bool(&expr)
            == Some(false);

    let grid_groups = grouped_plan
        .owned_grid_groups_for_io(
            &meta,
            GridGroupExecutionOpts {
                literal_false_clear:
                    emit_empty_schema_once,
                drop_redundant_1d_coords:
                    !emit_empty_schema_once,
                streaming_batch_io_cut: None,
            },
        )?;

    // Check max_chunks_to_read limit before doing any I/O
    if let Some(max_chunks) = max_chunks_to_read {
        let total_chunks =
            streaming_grid_chunk_read_count(
                &grid_groups,
            );
        ensure!(
            total_chunks <= max_chunks,
            MaxChunksToReadExceededSnafu {
                total_chunks,
                max_chunks,
            }
        );
    }

    let max_conc = max_concurrency
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CONCURRENCY);
    let semaphore = StdArc::new(
        tokio::sync::Semaphore::new(max_conc),
    );

    // Read chunks using consolidated (deduplicated) iteration
    let mut futs = FuturesUnordered::new();
    for group in grid_groups {
        let vars: Vec<IStr> = group.vars;

        for (idx, subset) in group
            .chunk_indices
            .into_iter()
            .zip(group.chunk_subsets)
        {
            let sem = semaphore.clone();
            let backend = backend.clone();
            let sig = group.sig.clone();
            let array_shape =
                group.array_shape.clone();
            let vars = vars.clone();
            let meta = meta.clone();

            futs.push(async move {
                let _permit = sem
                    .acquire_owned()
                    .await
                    .expect("semaphore closed");
                chunk_to_df_from_grid_with_backend(
                    backend.as_ref(),
                    idx,
                    sig.as_ref(),
                    &array_shape,
                    &vars,
                    None,
                    subset.as_ref(),
                    &meta,
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
            &meta.tidy_schema(Some(
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
        Ok(restructure_to_structs(
            &result, &meta,
        )?)
    } else {
        Ok(result)
    }
}
