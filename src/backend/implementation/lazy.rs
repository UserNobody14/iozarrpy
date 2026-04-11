//! Synchronous scan using the backend.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;

use crate::IStr;
use crate::backend::implementation::iterating_common::expr_top_literal_bool;
use crate::chunk_plan::{
    GridGroupExecutionOpts, streaming_grid_chunk_read_count,
};
use crate::errors::BackendError;
use crate::errors::MaxChunksToReadExceededSnafu;
use crate::scan::column_policy::ResolvedColumnPolicy;
use crate::scan::sync_scan::chunk_to_df_from_grid_with_backend;
use crate::shared::ChunkedExpressionCompilerSync;
use crate::shared::FullyCachedZarrBackendSync;
use crate::shared::{
    HasMetadataBackendSync,
    combine_chunk_dataframes,
    expand_projection_to_flat_paths,
    restructure_to_structs,
};
use snafu::ensure;

/// Internal: scan using the backend.
///
/// This uses the backend's cached metadata and chunk reading directly.
pub fn scan_zarr_with_backend_sync(
    backend: &Arc<FullyCachedZarrBackendSync>,
    expr: polars::prelude::Expr,
    with_columns: Option<BTreeSet<IStr>>,
    max_chunks_to_read: Option<usize>,
) -> Result<
    polars::prelude::DataFrame,
    BackendError,
> {
    // Get metadata from backend
    let meta = backend.metadata()?;

    let chunk_expanded =
        with_columns.as_ref().map(|cols| {
            expand_projection_to_flat_paths(
                cols, &meta,
            )
        });
    let enrich_policy = ResolvedColumnPolicy::new(
        with_columns.clone().or_else(|| {
            Some(
                meta.tidy_column_order(None)
                    .into_iter()
                    .collect(),
            )
        }),
        &expr,
        &meta,
    );
    let _enrich_wc = enrich_policy
        .physical_superset()
        .cloned();

    // Compile grouped chunk plan
    let (grouped_plan, _stats) =
        backend.compile_expression_sync(&expr)?;

    let emit_empty_schema_once =
        expr_top_literal_bool(&expr) == Some(false);

    let grid_groups = grouped_plan.owned_grid_groups_for_io(
        &meta,
        GridGroupExecutionOpts {
            literal_false_clear: emit_empty_schema_once,
            drop_redundant_1d_coords: !emit_empty_schema_once,
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

    // Read chunks using consolidated (deduplicated) iteration
    let mut dfs = Vec::new();
    for group in &grid_groups {
        let vars: Vec<IStr> = group.vars.clone();

        for (idx, subset) in group
            .chunk_indices
            .iter()
            .cloned()
            .zip(group.chunk_subsets.iter().cloned())
        {
            let df =
                chunk_to_df_from_grid_with_backend(
                    backend,
                    idx,
                    group.sig.as_ref(),
                    &group.array_shape,
                    &vars,
                    chunk_expanded.as_ref(),
                    subset.as_ref(),
                    &meta,
                )?;
            dfs.push(df);
        }
    }

    // Combine all chunk DataFrames
    // For heterogeneous grids, we need to join on coordinate columns
    let result = if dfs.is_empty() {
        let keys: Vec<IStr> = grouped_plan
            .var_to_grid()
            .keys()
            .cloned()
            .collect();
        DataFrame::empty_with_schema(
            &meta.tidy_schema(Some(
                keys.as_slice(),
            )),
        )
    } else if dfs.len() == 1 {
        dfs.into_iter().next().unwrap()
    } else {
        combine_chunk_dataframes(dfs, &meta)?
    };

    // let result = enrich_df_missing_requested_vars(
    //     backend,
    //     result,
    //     &meta,
    //     enrich_wc.as_ref(),
    // )?;

    // For hierarchical data, convert flat path columns to struct columns
    if meta.is_hierarchical() {
        Ok(restructure_to_structs(
            &result, &meta,
        )?)
    } else {
        Ok(result)
    }
}
