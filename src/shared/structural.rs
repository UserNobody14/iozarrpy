use polars::prelude::*;
use snafu::ResultExt;
use std::collections::BTreeSet;

use crate::errors::{BackendResult, PolarsSnafu};
use crate::meta::path::ZarrPath;
use crate::meta::{ZarrMeta, ZarrNode};
use crate::IStr;
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
            let reordered = df
                .select(col_order.as_slice())
                .context(PolarsSnafu)?;
            acc.vstack_mut(&reordered)
                .context(PolarsSnafu)?;
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
            .context(PolarsSnafu)?;
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

    for (child_name, child_node) in
        &meta.root.children
    {
        let child_name_str: &str =
            child_name.as_ref();
        let prefix =
            ZarrPath::single(child_name.clone());
        let struct_col =
            build_struct_column_for_node(
                df,
                child_node,
                &prefix,
                &mut processed_paths,
            )?;
        if let Some(col) = struct_col {
            result_columns.push(
                col.with_name(
                    child_name_str.into(),
                ),
            );
        }
    }

    Ok(DataFrame::new(
        df.height(),
        result_columns,
    )
    .context(PolarsSnafu)?)
}

/// Build a struct column for a zarr node (group).
fn build_struct_column_for_node(
    df: &DataFrame,
    node: &ZarrNode,
    prefix: &ZarrPath,
    processed_paths: &mut BTreeSet<String>,
) -> BackendResult<Option<Column>> {
    let mut fields: Vec<Column> = Vec::new();

    for var in &node.data_vars {
        let var_str: &str = var.as_ref();
        let full_zp = prefix.push(var.clone());
        let full_path = full_zp.to_flat_string();

        if let Ok(col) = df.column(&full_path) {
            let renamed = col
                .clone()
                .into_column()
                .with_name(var_str.into());
            fields.push(renamed);
            processed_paths.insert(full_path);
        }
    }

    for (child_name, child_node) in
        &node.children
    {
        let child_name_str: &str =
            child_name.as_ref();
        let nested_prefix =
            prefix.push(child_name.clone());
        let nested_struct =
            build_struct_column_for_node(
                df,
                child_node,
                &nested_prefix,
                processed_paths,
            )?;
        if let Some(col) = nested_struct {
            let renamed = col
                .with_name(child_name_str.into());
            fields.push(renamed);
        }
    }

    if fields.is_empty() {
        return Ok(None);
    }

    let struct_series =
        StructChunked::from_columns(
            prefix.to_flat_string().as_str().into(),
            df.height(),
            &fields,
        )
        .context(PolarsSnafu)?;

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

        if meta.dim_analysis.all_dims.iter().any(
            |d| {
                let d_str: &str = d.as_ref();
                d_str == col_str
            },
        ) {
            expanded.insert(col.clone());
            continue;
        }

        if meta.root.data_vars.iter().any(|v| {
            let v_str: &str = v.as_ref();
            v_str == col_str
        }) {
            expanded.insert(col.clone());
            continue;
        }

        if meta
            .array_by_path_contains(col.clone())
        {
            expanded.insert(col.clone());
            continue;
        }

        // Use tree traversal to expand group names to all child paths
        let zp = ZarrPath::from(col);
        let child_paths =
            meta.root.find_paths_under(&zp);
        if !child_paths.is_empty() {
            for p in child_paths {
                expanded.insert(p.to_istr());
            }
            continue;
        }

        expanded.insert(col.clone());
    }

    expanded
}
