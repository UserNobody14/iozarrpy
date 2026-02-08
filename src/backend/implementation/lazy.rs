//! Synchronous scan using the backend.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::PyErr;

use crate::scan::sync_chunk_to_df::chunk_to_df_from_grid_with_backend;
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
    use std::collections::BTreeSet as StdBTreeSet;
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

    // Count total chunks to read if max_chunks_to_read is set
    if let Some(max_chunks) = max_chunks_to_read {
        let mut total_chunks = 0usize;
        for (_sig, _vars, subsets, chunkgrid) in
            grouped_plan.iter_grids()
        {
            // Deduplicate chunk indices across potentially overlapping subsets.
            let mut uniq: StdBTreeSet<Vec<u64>> =
                StdBTreeSet::new();
            for subset in subsets.subsets_iter() {
                if let Ok(Some(indices)) =
                    chunkgrid
                        .chunks_in_array_subset(
                            subset,
                        )
                {
                    for idx in indices.indices() {
                        uniq.insert(idx.to_vec());
                    }
                }
            }
            total_chunks += uniq.len();
        }
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

    let mut dfs = Vec::new();
    for (sig, vars, subsets, chunkgrid) in
        grouped_plan.iter_grids()
    {
        let vars: Vec<IStr> = vars
            .into_iter()
            .map(|v| v.istr())
            .collect();
        let array_shape =
            chunkgrid.array_shape().to_vec();

        let mut uniq: StdBTreeSet<Vec<u64>> =
            StdBTreeSet::new();
        for subset in subsets.subsets_iter() {
            let chunk_indices = chunkgrid
                .chunks_in_array_subset(subset)
                .map_err(|e| {
                    PyErr::new::<
                        pyo3::exceptions::PyValueError,
                        _,
                    >(e.to_string())
                })?
                .ok_or(PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >("no chunks found"))?;
            for idx in chunk_indices.indices() {
                uniq.insert(idx.to_vec());
            }
        }

        for idx in uniq {
            let df =
                chunk_to_df_from_grid_with_backend(
                    backend,
                    idx.into(),
                    sig,
                    &array_shape,
                    &vars,
                    expanded_with_columns.as_ref(),
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
        restructure_to_structs(&result, &meta)
    } else {
        Ok(result)
    }
}
