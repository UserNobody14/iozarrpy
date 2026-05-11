//! Async eager scan using the unified [`GridJoinTree`]-driven reader.

use std::sync::Arc;

use snafu::ensure;

use crate::backend::implementation::iterating_common::{
    build_batches, distinct_chunks_in_batches, expr_top_literal_bool,
};
use crate::chunk_plan::GridJoinTree;
use crate::chunk_plan::indexing::grid_join_reader::flatten_reads;
use crate::chunk_plan::indexing::joined_assembly::{
    BatchChunkRead, assemble_joined_batch,
};
use crate::errors::BackendError;
use crate::errors::MaxChunksToReadExceededSnafu;
use crate::meta::ZarrMeta;
use crate::scan::async_scan::read_chunk_raw_from_grid_with_backend;
use crate::shared::ChunkedDataBackendAsync;
use crate::shared::ChunkedExpressionCompilerAsync;
use crate::shared::HasMetadataBackendAsync;
use crate::shared::IStr;
use crate::shared::diagonal_concat_batches;
use crate::shared::restructure_to_structs;

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

    let (grouped_plan, _stats) = Arc::clone(&backend)
        .compile_expression_async(&expr)
        .await?;

    let emit_empty_schema_once =
        expr_top_literal_bool(&expr)
            == Some(false);

    let groups = grouped_plan
        .owned_grid_groups_for_io(
            emit_empty_schema_once,
        )?;

    let tree = GridJoinTree::build(groups);
    let batches = tree.as_ref().map_or_else(
        Vec::new,
        |t| build_batches(t, usize::MAX),
    );

    if let Some(max_chunks) = max_chunks_to_read {
        let leaves_for_count = tree
            .as_ref()
            .map_or_else(Vec::new, |t| t.leaves());
        let total_chunks =
            distinct_chunks_in_batches(
                &batches,
                &leaves_for_count,
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

    let mut batch_dfs: Vec<
        polars::prelude::DataFrame,
    > = Vec::new();
    if let Some(tree_ref) = tree.as_ref() {
        let leaves = tree_ref.leaves();
        for plan in &batches {
            let reads =
                flatten_reads(plan, &leaves);
            if reads.is_empty() {
                continue;
            }

            let mut futs =
                FuturesUnordered::new();
            for r in reads {
                let sem = Arc::clone(&semaphore);
                let backend = Arc::clone(&backend);
                let meta = Arc::clone(&meta);
                let leaf_idx = r.leaf_idx;
                let slot = r.slot;
                let sig = Arc::clone(&r.sig);
                let vars = r.vars.clone();
                let idx = r.idx.clone();

                futs.push(async move {
                    let _permit =
                        sem.acquire_owned().await.expect("semaphore closed");
                    let raw = read_chunk_raw_from_grid_with_backend(
                        backend.as_ref(),
                        idx.as_slice(),
                        sig.as_ref(),
                        &vars,
                        None,
                        &meta,
                    )
                    .await?;
                    Ok::<_, BackendError>(BatchChunkRead {
                        leaf_idx,
                        slot,
                        raw,
                    })
                });
            }

            let mut chunk_reads: Vec<BatchChunkRead> =
                Vec::new();
            while let Some(r) = futs.next().await
            {
                chunk_reads.push(r?);
            }

            if let Some(df) = assemble_joined_batch(
                plan,
                &leaves,
                chunk_reads,
                &meta,
            )? {
                batch_dfs.push(df);
            }
        }
    }

    let result = if batch_dfs.is_empty() {
        let keys: Vec<IStr> = grouped_plan
            .var_to_grid()
            .keys()
            .cloned()
            .collect();
        polars::prelude::DataFrame::empty_with_schema(
            &meta.tidy_schema(Some(keys.as_slice())),
        )
    } else {
        diagonal_concat_batches(batch_dfs)?
    };

    if meta.is_hierarchical() {
        Ok(restructure_to_structs(
            &result, &meta,
        )?)
    } else {
        Ok(result)
    }
}
