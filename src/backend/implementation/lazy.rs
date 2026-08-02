//! Synchronous eager scan using the unified [`GridJoinTree`]-driven reader.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use snafu::ensure;

use crate::backend::implementation::iterating_common::{
    build_batches, distinct_chunks_in_batches,
};
use crate::chunk_plan::indexing::grid_join_reader::{
    assemble_batch_dataframe, flatten_reads,
};
use crate::chunk_plan::compile_to_tree_sync;
use crate::errors::BackendError;
use crate::errors::MaxChunksToReadExceededSnafu;
use crate::scan::sync_scan::chunk_to_df_from_grid_with_backend;
use crate::shared::FullyCachedZarrBackendSync;
use crate::shared::IStr;
use crate::shared::MaybeParIter;
use crate::shared::{
    HasMetadataBackendSync, diagonal_concat_batches,
    expand_projection_to_flat_paths,
};

/// Below this many chunk reads per batch the rayon scheduling overhead exceeds
/// the gain from parallel decode. Mirrors the threshold used by the streaming
/// iterator path.
const PARALLEL_CHUNK_READS: usize = 2;

/// Eager sync scan: drives the [`GridJoinTree`] reader to exhaustion and
/// diagonal-concats the per-batch DataFrames.
pub fn scan_zarr_with_backend_sync(
    backend: &Arc<FullyCachedZarrBackendSync>,
    expr: &polars::prelude::Expr,
    with_columns: Option<&BTreeSet<IStr>>,
    max_chunks_to_read: Option<usize>,
) -> Result<
    polars::prelude::DataFrame,
    BackendError,
> {
    let meta = backend.metadata()?;

    let chunk_expanded =
        with_columns.map(|cols| {
            expand_projection_to_flat_paths(
                cols, &meta,
            )
        });

    let (tree, _stats) = compile_to_tree_sync(
        expr, &meta, backend,
    )?;

    // For eager scans we materialize the entire dataset at once, so let each
    // batch be as large as it needs to be. The planner still applies geometry
    // bounds (chunk count cap), but we don't need to cap by row budget.
    let batches = tree
        .as_ref()
        .map_or_else(Vec::new, |t| {
            build_batches(t, usize::MAX)
        });

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

    let mut batch_dfs: Vec<DataFrame> =
        Vec::new();
    if let Some(tree_ref) = tree.as_ref() {
        let leaves = tree_ref.leaves();
        for plan in &batches {
            let reads =
                flatten_reads(plan, &leaves);
            if reads.is_empty() {
                continue;
            }
            let chunk_dfs: Vec<(usize, DataFrame)> = reads
                .maybe_par_iter(PARALLEL_CHUNK_READS)
                .map_collect(|r| {
                    let df = chunk_to_df_from_grid_with_backend(
                        backend,
                        &r.idx,
                        r.sig.as_ref(),
                        &r.array_shape,
                        &r.vars,
                        chunk_expanded.as_ref(),
                        r.subset.as_deref(),
                        &meta,
                    )?;
                    Ok::<_, BackendError>((r.leaf_idx, df))
                })?;

            if let Some(df) =
                assemble_batch_dataframe(
                    plan, chunk_dfs,
                )?
            {
                batch_dfs.push(df);
            }
        }
    }

    let result = if batch_dfs.is_empty() {
        let keys: Option<Vec<IStr>> =
            with_columns.map(|cols| {
                cols.iter().copied().collect()
            });
        let schema =
            meta.tidy_schema(keys.as_deref());
        DataFrame::empty_with_schema(&schema)
    } else {
        diagonal_concat_batches(batch_dfs)?
    };

    Ok(result)
}
