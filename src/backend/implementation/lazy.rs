//! Synchronous scan using the backend.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::PyErr;

use crate::scan::sync_scan::chunk_to_df_from_grid_with_backend;
use crate::shared::ChunkedExpressionCompilerSync;
use crate::shared::FullyCachedZarrBackendSync;
use crate::shared::{
    HasMetadataBackendSync,
    combine_chunk_dataframes,
    expand_projection_to_flat_paths,
    restructure_to_structs,
};
use crate::{IStr, IntoIStr};

/// Internal: scan using the backend.
///
/// This uses the backend's cached metadata and chunk reading directly.
pub fn scan_zarr_with_backend_sync(
    backend: &Arc<FullyCachedZarrBackendSync>,
    expr: polars::prelude::Expr,
    with_columns: Option<BTreeSet<IStr>>,
    max_chunks_to_read: Option<usize>,
) -> Result<polars::prelude::DataFrame, PyErr> {
    use std::sync::Arc as StdArc;

    // Get metadata from backend
    let meta = backend.metadata()?;
    let planning_meta =
        StdArc::new(meta.planning_meta());

    // Expand struct column names to flat paths for chunk reading
    // e.g., "model_a" -> ["model_a/temperature", "model_a/pressure"]
    let expanded_with_columns =
        with_columns.as_ref().map(|cols| {
            expand_projection_to_flat_paths(
                cols, &meta,
            )
        });

    // Compile grouped chunk plan
    let (grouped_plan, _stats) =
        backend.compile_expression_sync(&expr)?;

    // Check max_chunks_to_read limit before doing any I/O
    if let Some(max_chunks) = max_chunks_to_read {
        let total_chunks =
            grouped_plan.total_unique_chunks()?;
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

    // Read chunks using consolidated (deduplicated) iteration
    let mut dfs = Vec::new();
    for group in
        grouped_plan.iter_consolidated_chunks()
    {
        let group = group?;
        let vars: Vec<IStr> = group
            .vars
            .iter()
            .map(|v| v.istr())
            .collect();

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
                    expanded_with_columns.as_ref(),
                    subset.as_ref(),
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
        Ok(restructure_to_structs(
            &result, &meta,
        )?)
    } else {
        Ok(result)
    }
}
