//! Synchronous scan using the backend.

use std::collections::BTreeSet;
use std::sync::Arc;

use pyo3::PyErr;

use crate::IStr;
use crate::backend::compile::ChunkedExpressionCompilerSync;
use crate::backend::traits::HasMetadataBackendSync;
use crate::backend::zarr::FullyCachedZarrBackendSync;
use crate::scan::sync_chunk_to_df::chunk_to_df_from_grid_with_backend;

/// Internal: scan using the backend.
///
/// This uses the backend's cached metadata and chunk reading directly.
pub fn scan_zarr_with_backend_sync(
    backend: &Arc<FullyCachedZarrBackendSync>,
    expr: polars::prelude::Expr,
    with_columns: Option<BTreeSet<IStr>>,
) -> Result<polars::prelude::DataFrame, PyErr> {
    use crate::IntoIStr;
    use pyo3_polars::error::PyPolarsErr;
    use std::sync::Arc as StdArc;

    // Get metadata from backend
    let meta = backend.metadata()?;
    let planning_meta =
        StdArc::new(meta.planning_meta());

    // Compile grouped chunk plan
    let (grouped_plan, _stats) =
        backend.compile_expression_sync(&expr)?;

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
                let df =
                    chunk_to_df_from_grid_with_backend(
                        backend,
                        idx.into(),
                        sig,
                        &array_shape,
                        &vars,
                        with_columns.as_ref(),
                    )?;
                dfs.push(df);
            }
        }
    }

    let mut out: Option<
        polars::prelude::DataFrame,
    > = None;
    for df in dfs {
        if let Some(acc) = &mut out {
            acc.vstack_mut(&df)
                .map_err(PyPolarsErr::from)?;
        } else {
            out = Some(df);
        }
    }

    let planning_meta_clone =
        StdArc::clone(&planning_meta);

    Ok(out.unwrap_or_else(|| {
        let keys: Vec<IStr> = grouped_plan
            .var_to_grid()
            .keys()
            .cloned()
            .collect();
        polars::prelude::DataFrame::empty_with_schema(
            &planning_meta_clone
                .tidy_schema(Some(keys.as_slice())),
        )
    }))
}
