//! Synchronous scan using the backend.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::PyErr;

use crate::backend::compile::ChunkedExpressionCompilerSync;
use crate::backend::traits::HasMetadataBackendSync;
use crate::backend::zarr::FullyCachedZarrBackendSync;
use crate::meta::ZarrMeta;
use crate::meta::ZarrNode;
use crate::scan::sync_chunk_to_df::chunk_to_df_from_grid_with_backend;
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
    use pyo3_polars::error::PyPolarsErr;
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
            for subset in subsets.subsets_iter() {
                if let Ok(Some(indices)) =
                    chunkgrid
                        .chunks_in_array_subset(
                            subset,
                        )
                {
                    total_chunks += indices
                        .num_elements_usize();
                }
            }
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
                        expanded_with_columns.as_ref(),
                    )?;
                dfs.push(df);
            }
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

/// Combine chunk DataFrames, handling heterogeneous schemas.
///
/// Strategy:
/// 1. Group DataFrames by their column schema (signature)
/// 2. Within each group, use vstack (same schema)
/// 3. Across groups, join on shared coordinate columns
pub fn combine_chunk_dataframes(
    mut dfs: Vec<DataFrame>,
    meta: &ZarrMeta,
) -> Result<DataFrame, PyErr> {
    use pyo3_polars::error::PyPolarsErr;
    use std::collections::BTreeMap;

    // Get dimension column names
    let dim_cols: BTreeSet<&str> = meta
        .dim_analysis
        .all_dims
        .iter()
        .map(|d| d.as_ref())
        .collect();

    // Group DataFrames by schema signature
    // (sorted list of column names)
    let mut schema_groups: BTreeMap<
        Vec<PlSmallStr>,
        Vec<DataFrame>,
    > = BTreeMap::new();

    for df in dfs.drain(..) {
        let mut cols: Vec<PlSmallStr> =
            df.get_column_names_owned();
        cols.sort();
        schema_groups
            .entry(cols)
            .or_default()
            .push(df);
    }

    // Combine within each group (vstack)
    // Need to ensure consistent column order before vstacking
    let mut vstacked: Vec<DataFrame> =
        Vec::with_capacity(schema_groups.len());
    for (_sig, group) in schema_groups {
        let mut iter = group.into_iter();
        let first = iter.next().unwrap();

        // Get the column order from the first DataFrame
        let col_order: Vec<PlSmallStr> =
            first.get_column_names_owned();

        let mut acc = first;
        for df in iter {
            // Reorder columns to match the first DataFrame
            let reordered = df
                .select(col_order.as_slice())
                .map_err(PyPolarsErr::from)?;
            acc.vstack_mut(&reordered)
                .map_err(PyPolarsErr::from)?;
        }
        vstacked.push(acc);
    }

    if vstacked.len() == 1 {
        return Ok(vstacked
            .into_iter()
            .next()
            .unwrap());
    }

    // Multiple schema groups - join on coordinate columns
    // Find shared coordinate columns across all vstacked DFs
    let shared_coords: Vec<PlSmallStr> = {
        let first_cols: BTreeSet<PlSmallStr> =
            vstacked[0]
                .get_column_names_owned()
                .into_iter()
                .filter(|c| {
                    dim_cols.contains(c.as_str())
                })
                .collect();

        vstacked
            .iter()
            .skip(1)
            .fold(first_cols, |acc, df| {
                let df_dims: BTreeSet<
                    PlSmallStr,
                > = df
                    .get_column_names_owned()
                    .into_iter()
                    .filter(|c| {
                        dim_cols
                            .contains(c.as_str())
                    })
                    .collect();
                acc.intersection(&df_dims)
                    .cloned()
                    .collect()
            })
            .into_iter()
            .collect()
    };

    if shared_coords.is_empty() {
        // No shared coordinates - fall back to diagonal concat
        return polars::functions::concat_df_diagonal(
            &vstacked,
        )
        .map_err(PyPolarsErr::from)
        .map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e.to_string())
        });
    }

    // Join on shared coordinates
    // Start with the first DF and successively join others
    let mut result = vstacked.remove(0);
    for df in vstacked {
        result = result
            .join(
                &df,
                shared_coords.as_slice(),
                shared_coords.as_slice(),
                JoinArgs::new(JoinType::Full)
                    .with_coalesce(
                    JoinCoalesce::CoalesceColumns,
                ),
                None,
            )
            .map_err(PyPolarsErr::from)?;
    }

    Ok(result)
}

/// Convert flat path columns (e.g., "model_a/temperature") to nested struct columns.
///
/// For hierarchical zarr stores, child groups should become struct columns
/// containing their variables as fields.
pub fn restructure_to_structs(
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

// =============================================================================
// Projection Expansion (struct columns -> flat paths)
// =============================================================================

/// Expand struct column names to their underlying flat paths.
///
/// When projection pushdown requests "model_a" (a struct column), we need to
/// expand that to all the flat paths under that group:
/// - "model_a" -> ["model_a/temperature", "model_a/pressure", ...]
///
/// This also handles:
/// - Dimension columns (passed through as-is)
/// - Root data variables (passed through as-is)
/// - Nested struct access like "model_a" expanding to all nested paths
pub fn expand_projection_to_flat_paths(
    with_columns: &BTreeSet<IStr>,
    meta: &ZarrMeta,
) -> BTreeSet<IStr> {
    let mut expanded = BTreeSet::new();

    for col in with_columns {
        let col_str: &str = col.as_ref();

        // Check if it's a dimension - pass through
        if meta.dim_analysis.all_dims.iter().any(
            |d| {
                let d_str: &str = d.as_ref();
                d_str == col_str
            },
        ) {
            expanded.insert(col.clone());
            continue;
        }

        // Check if it's a root data variable - pass through
        if meta.root.data_vars.iter().any(|v| {
            let v_str: &str = v.as_ref();
            v_str == col_str
        }) {
            expanded.insert(col.clone());
            continue;
        }

        // Check if it's already a flat path that exists
        if meta.path_to_array.contains_key(col) {
            expanded.insert(col.clone());
            continue;
        }

        // Check if it's a child group name - expand to all paths
        if let Some(child_node) =
            meta.root.children.get(col)
        {
            collect_all_paths_from_node(
                child_node,
                col_str,
                &mut expanded,
            );
            continue;
        }

        // Try to match partial paths like "level_1/level_2"
        // that might reference nested groups
        if let Some(paths) =
            find_paths_matching_prefix(
                col_str, meta,
            )
        {
            for p in paths {
                expanded.insert(p);
            }
            continue;
        }

        // Unknown column - pass through (might be handled elsewhere)
        expanded.insert(col.clone());
    }

    expanded
}

/// Recursively collect all variable paths from a node and its children.
fn collect_all_paths_from_node(
    node: &ZarrNode,
    prefix: &str,
    out: &mut BTreeSet<IStr>,
) {
    // Add all data variables in this node
    for var in &node.data_vars {
        let var_str: &str = var.as_ref();
        let path =
            format!("{}/{}", prefix, var_str);
        out.insert(path.istr());
    }

    // Recursively add from child nodes
    for (child_name, child_node) in &node.children
    {
        let child_name_str: &str =
            child_name.as_ref();
        let child_prefix = format!(
            "{}/{}",
            prefix, child_name_str
        );
        collect_all_paths_from_node(
            child_node,
            &child_prefix,
            out,
        );
    }
}

/// Find all paths that start with the given prefix.
fn find_paths_matching_prefix(
    prefix: &str,
    meta: &ZarrMeta,
) -> Option<Vec<IStr>> {
    let prefix_with_slash =
        format!("{}/", prefix);
    let matching: Vec<IStr> = meta
        .path_to_array
        .keys()
        .filter(|p| {
            let p_str: &str = p.as_ref();
            p_str.starts_with(&prefix_with_slash)
                || p_str == prefix
        })
        .cloned()
        .collect();

    if matching.is_empty() {
        None
    } else {
        Some(matching)
    }
}
