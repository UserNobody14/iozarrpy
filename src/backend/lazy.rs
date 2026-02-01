//! Synchronous scan using the backend.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::PyErr;

use crate::IStr;
use crate::backend::compile::ChunkedExpressionCompilerSync;
use crate::backend::traits::HasMetadataBackendSync;
use crate::backend::zarr::FullyCachedZarrBackendSync;
use crate::meta::ZarrMeta;
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

    // Combine all chunk DataFrames
    // For heterogeneous grids, we need to use diagonal concat
    // to handle different column sets
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
        // Check if all DataFrames have the same columns
        let first_cols: Vec<_> =
            dfs[0].get_column_names_owned();
        let all_same = dfs.iter().all(|df| {
            df.get_column_names_owned()
                == first_cols
        });

        if all_same {
            // Fast path: simple vstack
            let mut out = dfs.remove(0);
            for df in dfs {
                out.vstack_mut(&df)
                    .map_err(PyPolarsErr::from)?;
            }
            out
        } else {
            // Diagonal concat for heterogeneous grids
            polars::functions::concat_df_diagonal(
                &dfs,
            )
            .map_err(PyPolarsErr::from)?
        }
    };

    // For hierarchical data, convert flat path columns to struct columns
    if meta.is_hierarchical() {
        restructure_to_structs(&result, &meta)
    } else {
        Ok(result)
    }
}

/// Convert flat path columns (e.g., "model_a/temperature") to nested struct columns.
///
/// For hierarchical zarr stores, child groups should become struct columns
/// containing their variables as fields.
fn restructure_to_structs(
    df: &DataFrame,
    meta: &ZarrMeta,
) -> Result<DataFrame, PyErr> {
    use pyo3_polars::error::PyPolarsErr;
    use std::collections::BTreeMap;

    let mut result_columns: Vec<Column> =
        Vec::new();
    let mut processed_paths: BTreeSet<String> =
        BTreeSet::new();

    // First, add dimension columns (non-path columns)
    for dim in &meta.dim_analysis.all_dims {
        let dim_str: &str = dim.as_ref();
        if let Ok(col) = df.column(dim_str) {
            result_columns.push(col.clone());
            processed_paths
                .insert(dim_str.to_string());
        }
    }

    // Add root-level data variables
    for var in &meta.root.data_vars {
        let var_str: &str = var.as_ref();
        if let Ok(col) = df.column(var_str) {
            result_columns.push(col.clone());
            processed_paths
                .insert(var_str.to_string());
        }
    }

    // For each child group, create a struct column
    for (child_name, child_node) in
        &meta.root.children
    {
        let child_name_str: &str =
            child_name.as_ref();
        let struct_col =
            build_struct_column_for_node(
                df,
                child_node,
                child_name_str,
                &mut processed_paths,
            )?;
        if let Some(col) = struct_col {
            result_columns.push(col);
        }
    }

    DataFrame::new(df.height(), result_columns)
        .map_err(PyPolarsErr::from)
        .map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e.to_string())
        })
}

/// Build a struct column for a zarr node (group).
fn build_struct_column_for_node(
    df: &DataFrame,
    node: &crate::meta::ZarrNode,
    prefix: &str,
    processed_paths: &mut BTreeSet<String>,
) -> Result<Option<Column>, PyErr> {
    use pyo3_polars::error::PyPolarsErr;

    let mut fields: Vec<Column> = Vec::new();

    // Add data variable columns as struct fields
    for var in &node.data_vars {
        let var_str: &str = var.as_ref();
        let full_path =
            format!("{}/{}", prefix, var_str);

        if let Ok(col) = df.column(&full_path) {
            // Rename to just the leaf name for struct field
            let renamed = col
                .clone()
                .into_column()
                .with_name(var_str.into());
            fields.push(renamed);
            processed_paths.insert(full_path);
        }
    }

    // Recursively add nested child groups
    for (child_name, child_node) in &node.children
    {
        let child_name_str: &str =
            child_name.as_ref();
        let nested_prefix = format!(
            "{}/{}",
            prefix, child_name_str
        );
        let nested_struct =
            build_struct_column_for_node(
                df,
                child_node,
                &nested_prefix,
                processed_paths,
            )?;
        if let Some(col) = nested_struct {
            // Rename to just the leaf name for the struct field
            let renamed = col
                .with_name(child_name_str.into());
            fields.push(renamed);
        }
    }

    if fields.is_empty() {
        return Ok(None);
    }

    // Create struct column from fields
    let struct_series =
        StructChunked::from_columns(
            prefix.into(),
            df.height(),
            &fields,
        )
        .map_err(PyPolarsErr::from)
        .map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e.to_string())
        })?;

    Ok(Some(struct_series.into_column()))
}
