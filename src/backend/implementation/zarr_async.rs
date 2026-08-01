//! Async eager scan using the unified [`GridJoinTree`]-driven reader.

use std::sync::Arc;

use snafu::ensure;

use crate::backend::implementation::iterating_common::{
    build_batches, distinct_chunks_in_batches,
};
use crate::chunk_plan::compile_to_tree_async;
use crate::chunk_plan::indexing::grid_join_reader::{
    assemble_batch_dataframe, flatten_reads,
};
use crate::errors::BackendError;
use crate::errors::MaxChunksToReadExceededSnafu;
use crate::meta::ZarrMeta;
use crate::scan::async_scan::chunk_to_df_from_grid_with_backend;
use crate::shared::ChunkedDataBackendAsync;
use crate::shared::HasMetadataBackendAsync;
use crate::shared::diagonal_concat_batches;
/// Eager async scan: drives the [`GridJoinTree`] reader to exhaustion.
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
        + Send
        + Sync,
{
    use futures::stream::{
        FuturesUnordered, StreamExt,
    };
    use std::sync::Arc as StdArc;

    const DEFAULT_MAX_CONCURRENCY: usize = 32;
    let meta = backend.metadata().await?;

    let (tree, _stats) = compile_to_tree_async(
        &expr,
        &meta,
        backend.as_ref(),
    )
    .await?;

    let batches = match &tree {
        Some(t) => build_batches(t, usize::MAX),
        None => Vec::new(),
    };

    if let Some(max_chunks) = max_chunks_to_read {
        let total_chunks =
            distinct_chunks_in_batches(&batches);
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

    let mut batch_results: Vec<(
        usize,
        polars::prelude::DataFrame,
    )> = Vec::new();
    if let Some(tree_ref) = tree.as_ref() {
        let leaves = tree_ref.leaves();
        let mut batch_work = Vec::new();
        for (batch_idx, plan) in
            batches.iter().enumerate()
        {
            let reads =
                flatten_reads(plan, &leaves);
            if reads.is_empty() {
                continue;
            }
            batch_work.push((
                batch_idx,
                plan.clone(),
                reads,
            ));
        }

        let mut batch_futs = futures::stream::iter(batch_work)
        .map(|(batch_idx, plan, reads)| {
            let backend = Arc::clone(&backend);
            let meta = Arc::clone(&meta);
            let semaphore = Arc::clone(&semaphore);

            async move {
                let mut futs =
                    FuturesUnordered::new();
                for r in reads {
                    let sem =
                        Arc::clone(&semaphore);
                    let backend = Arc::clone(&backend);
                    let meta = Arc::clone(&meta);
                    let leaf_idx = r.leaf_idx;
                    let sig = Arc::clone(&r.sig);
                    let array_shape =
                        Arc::clone(&r.array_shape);
                    let vars = Arc::clone(&r.vars);
                    let idx = Arc::clone(&r.idx);
                    let subset = r.subset.clone();

                    futs.push(async move {
                        let _permit = sem
                            .acquire_owned()
                            .await
                            .expect(
                                "semaphore closed",
                            );
                        let df =
                            chunk_to_df_from_grid_with_backend(
                                backend.as_ref(),
                                idx.as_ref(),
                                sig.as_ref(),
                                array_shape.as_ref(),
                                vars.as_ref(),
                                None,
                                subset.as_deref(),
                                &meta,
                            )
                            .await?;
                        Ok::<_, BackendError>((
                            leaf_idx, df,
                        ))
                    });
                }

                let mut chunk_dfs: Vec<(
                    usize,
                    polars::prelude::DataFrame,
                )> = Vec::new();
                while let Some(r) =
                    futs.next().await
                {
                    chunk_dfs.push(r?);
                }

                let df =
                    assemble_batch_dataframe(
                        &plan, chunk_dfs,
                    )?;
                Ok::<_, BackendError>((
                    batch_idx, df,
                ))
            }
        })
        .buffer_unordered(max_conc);

        while let Some(r) =
            batch_futs.next().await
        {
            let (batch_idx, df) = r?;
            if let Some(df) = df {
                batch_results
                    .push((batch_idx, df));
            }
        }
    }
    batch_results
        .sort_by_key(|(batch_idx, _)| *batch_idx);
    let batch_dfs: Vec<
        polars::prelude::DataFrame,
    > = batch_results
        .into_iter()
        .map(|(_, df)| df)
        .collect();

    let result = if batch_dfs.is_empty() {
        polars::prelude::DataFrame::empty_with_schema(
            &meta.tidy_schema(None),
        )
    } else {
        diagonal_concat_batches(batch_dfs)?
    };

    Ok(result)
}
