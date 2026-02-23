use polars::prelude::*;
use snafu::ResultExt;
use std::collections::BTreeSet;

use crate::errors::{BackendResult, PolarsSnafu};
use crate::meta::{ZarrMeta, ZarrNode};
use crate::{IStr, IntoIStr};
/// Combine chunk DataFrames, handling heterogeneous schemas.
///
/// Strategy:
/// 1. Group DataFrames by their column schema (signature)
/// 2. Within each group, use vstack (same schema)
/// 3. Across groups, join on shared coordinate columns
pub fn combine_chunk_dataframes(
    mut dfs: Vec<DataFrame>,
    meta: &ZarrMeta,
) -> BackendResult<DataFrame> {
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
            let reordered =
                df.select(col_order.as_slice())
                .context(
                    PolarsSnafu 
                )?;
            acc.vstack_mut(&reordered)
            .context(
                PolarsSnafu 
            )?;
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
        return Ok(polars::functions::concat_df_diagonal(
            &vstacked,
        ).context(
            PolarsSnafu 
        )?);
    }

    // Join on shared coordinates
    // Start with the first DF and successively join others
    let mut result = vstacked.remove(0);
    for df in vstacked {
        result = result.join(
            &df,
            shared_coords.as_slice(),
            shared_coords.as_slice(),
            JoinArgs::new(JoinType::Full)
                .with_coalesce(
                    JoinCoalesce::CoalesceColumns,
                ),
            None,
        ).context(
            PolarsSnafu 
        )?;
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
) -> BackendResult<DataFrame> {
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

    Ok(DataFrame::new(
        df.height(),
        result_columns,
    ).context(
        PolarsSnafu 
    )?)
}

/// Build a struct column for a zarr node (group).
fn build_struct_column_for_node(
    df: &DataFrame,
    node: &crate::meta::ZarrNode,
    prefix: &str,
    processed_paths: &mut BTreeSet<String>,
) -> BackendResult<Option<Column>> {
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
        ).context(
            PolarsSnafu 
        )?;

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
        if meta.array_by_path_contains(col.clone()) {
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
            meta.find_paths_matching_prefix(
                col_str
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

