//! Synchronous scan using the backend.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;

use crate::IStr;
use crate::errors::BackendError;
use crate::errors::MaxChunksToReadExceededSnafu;
use crate::scan::column_policy::ResolvedColumnPolicy;
use crate::scan::sync_scan::chunk_to_df_from_grid_with_backend;
use crate::shared::ChunkedExpressionCompilerSync;
use crate::shared::FullyCachedZarrBackendSync;
use crate::shared::{
    HasMetadataBackendSync,
    combine_chunk_dataframes,
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
    // Compile grouped chunk plan
    let (mut grouped_plan, _stats) =
        backend.compile_expression_sync(&expr)?;

    // Limit applies to predicate-driven chunk enumeration only. Projection-only
    // arrays merged via [`GroupedChunkPlan::augment_with_physical_vars`] may add
    // further grids (e.g. 1D ``x``/``y`` coordinates) without affecting this check.
    if let Some(max_chunks) = max_chunks_to_read {
        let total_chunks =
            grouped_plan.total_unique_chunks()?;
        ensure!(
            total_chunks <= max_chunks,
            MaxChunksToReadExceededSnafu {
                total_chunks,
                max_chunks,
            }
        );
    }

    if let Some(ref superset) =
        enrich_policy.physical_superset()
    {
        grouped_plan.augment_with_physical_vars(
            superset, &meta,
        )?;
    }

    let chunk_read_superset =
        enrich_policy.physical_superset().cloned();

    // Read chunks using consolidated (deduplicated) iteration
    let mut dfs = Vec::new();
    for group in
        grouped_plan.iter_consolidated_chunks()
    {
        let group = group?;
        let vars: Vec<IStr> = group.vars;

        for (idx, subset) in group
            .chunk_indices
            .into_iter()
            .zip(group.chunk_subsets)
        {
            let df =
                chunk_to_df_from_grid_with_backend(
                    backend,
                    idx.into(),
                    group.sig,
                    &group.array_shape,
                    &vars,
                    chunk_read_superset.as_ref(),
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
