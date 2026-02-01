use std::sync::Arc;

use polars::prelude::IntoLazy;
use polars_lazy::prelude::AnonymousScan;
use pyo3::PyErr;

use crate::IStr;
use crate::backend::compile::ChunkedExpressionCompilerSync;
use crate::backend::traits::{
    HasMetadataBackendSync, HasStore,
};
use crate::backend::zarr::FullyCachedZarrBackendSync;
use crate::scan::open_arrays::open_arrays_sync_unified;
use crate::scan::sync_chunk_to_df::chunk_to_df_from_grid;

// impl AnonymousScan
//     for FullyCachedZarrBackendSync
// {
//     fn schema(
//         &self,
//         _infer_schema_length: Option<usize>,
//     ) -> polars::prelude::PolarsResult<
//         polars::prelude::SchemaRef,
//     > {
//         polars::prelude::polars_bail!(ComputeError: "must supply either a schema or a schema function");
//     }

//     fn allows_predicate_pushdown(&self) -> bool {
//         true
//     }

//     fn allows_projection_pushdown(&self) -> bool {
//         true
//     }

//     fn as_any(&self) -> &dyn std::any::Any {
//         todo!()
//     }

//     fn scan(
//         &self,
//         scan_opts: polars::prelude::AnonymousScanArgs,
//     ) -> polars::prelude::PolarsResult<
//         polars::prelude::DataFrame,
//     > {
//         let selfarc = Arc::new(self.clone());
//         let prd = scan_opts.predicate.unwrap();
//         let df = scan_zarr_with_backend_sync(
//             selfarc,
//             prd.clone(),
//         )?;

//         let filtered = df
//             .lazy()
//             .filter(prd.clone())
//             .collect()
//             .map_err(
//                 polars::prelude::PolarsError::from,
//             )?;
//         Ok(filtered)
//     }
// }

/// Internal: scan using the backend.
///
/// This reuses the existing scan infrastructure but uses the backend's cached metadata.
pub fn scan_zarr_with_backend_sync(
    backend: Arc<&FullyCachedZarrBackendSync>,
    expr: polars::prelude::Expr,
) -> Result<polars::prelude::DataFrame, PyErr> {
    use crate::IntoIStr;
    use futures::stream::{
        FuturesUnordered, StreamExt,
    };
    use pyo3_polars::error::PyPolarsErr;
    use std::sync::Arc as StdArc;

    const DEFAULT_MAX_CONCURRENCY: usize = 32;

    // Open arrays for reading
    let meta = backend.metadata()?;
    let planning_meta =
        StdArc::new(meta.planning_meta());
    let (var_arrays, coord_arrays) =
        open_arrays_sync_unified(
            backend.store().clone(),
            meta.as_ref(),
            &meta.all_data_var_paths(),
            &meta.dim_analysis.all_dims,
        )
        .map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e)
        })?;

    // Compile grouped chunk plan
    let (grouped_plan, _stats) =
        backend.compile_expression_sync(&expr)?;

    let var_arrays = StdArc::new(var_arrays);
    let coord_arrays = StdArc::new(coord_arrays);

    let mut dfs = Vec::new();
    for (sig, vars, subsets, chunkgrid) in
        grouped_plan.iter_grids()
    {
        let vars = StdArc::new(
            vars.into_iter()
                .map(|v| v.istr())
                .collect::<Vec<_>>(),
        );
        let dims =
            StdArc::new(sig.dims().to_vec());

        for subset in subsets.subsets_iter() {
            let chunk_indices = chunkgrid
                .chunks_in_array_subset(subset).map_err(
                    |e| PyErr::new::<
                        pyo3::exceptions::PyValueError,
                        _,
                    >(e.to_string())
                )?.ok_or(
                    PyErr::new::<
                        pyo3::exceptions::PyValueError,
                        _,
                    >("no chunks found")
                )?;

            for idx in chunk_indices.indices() {
                let dims = StdArc::clone(&dims);
                let vars = StdArc::clone(&vars);
                let var_arrays =
                    StdArc::clone(&var_arrays);
                let coord_arrays =
                    StdArc::clone(&coord_arrays);
                let meta =
                    StdArc::clone(&planning_meta);

                let df = chunk_to_df_from_grid(
                    idx.into(),
                    sig.clone(),
                    chunkgrid.clone(),
                    meta,
                    dims,
                    vars,
                    var_arrays,
                    coord_arrays,
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
        let keys: Vec<IStr> = grouped_plan.var_to_grid().keys().cloned().collect();
        polars::prelude::DataFrame::empty_with_schema(
            &planning_meta_clone.tidy_schema(Some(keys.as_slice())),
        )
    }))
}
