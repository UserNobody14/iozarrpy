use polars::prelude::*;
use snafu::ResultExt;
use std::collections::BTreeSet;

use crate::errors::{BackendResult, PolarsSnafu};
use crate::meta::path::ZarrPath;
use crate::meta::{ZarrMeta, ZarrNode};
use crate::shared::{IStr, IntoIStr};

/// Diagonal-concatenate batch DataFrames produced by the
/// [`crate::chunk_plan::indexing::grid_join_reader`].
///
/// Each batch is already join-closed (joins inside `Join` nodes happen during
/// batch assembly), so combining batches just needs `concat_df_diagonal` to
/// align mismatched schemas with `null` fills.
pub fn diagonal_concat_batches(
    dfs: Vec<DataFrame>,
) -> BackendResult<DataFrame> {
    if dfs.is_empty() {
        return Ok(DataFrame::empty());
    }
    if dfs.len() == 1 {
        return Ok(dfs
            .into_iter()
            .next()
            .unwrap());
    }
    polars::functions::concat_df_diagonal(&dfs).context(PolarsSnafu {
        message: "Error diagonal-concatenating batch DataFrames".to_string(),
    })
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
            ZarrPath::single(*child_name);
        let struct_col =
            build_struct_column_for_node(
                df,
                child_node,
                &prefix,
                &mut processed_paths,
            )?;
        if let Some(col) = struct_col {
            result_columns.push(col.with_name(
                child_name_str.into(),
            ));
        }
    }

    DataFrame::new(df.height(), result_columns)
        .context(PolarsSnafu {
            message: "Error creating DataFrame"
                .to_string(),
        })
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
        let full_zp = prefix.push(*var);
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

    for (child_name, child_node) in &node.children
    {
        let child_name_str: &str =
            child_name.as_ref();
        let nested_prefix =
            prefix.push(*child_name);
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
            prefix
                .to_flat_string()
                .as_str()
                .into(),
            df.height(),
            &fields,
        )
        .context(PolarsSnafu {
            message:
                "Error creating StructChunked"
                    .to_string(),
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

        if meta.dim_analysis.all_dims.iter().any(
            |d| {
                let d_str: &str = d.as_ref();
                d_str == col_str
            },
        ) {
            expanded.insert(*col);
            continue;
        }

        if meta.root.data_vars.iter().any(|v| {
            let v_str: &str = v.as_ref();
            v_str == col_str
        }) {
            expanded.insert(*col);
            continue;
        }

        if meta.array_by_path_contains(*col) {
            expanded.insert(*col);
            continue;
        }

        // Use tree traversal to expand group names to all child paths
        let zp = ZarrPath::from(col);
        let child_paths =
            meta.root.find_paths_under(&zp);
        if !child_paths.is_empty() {
            for p in child_paths {
                expanded.insert(p.istr());
            }
            continue;
        }

        expanded.insert(*col);
    }

    expanded
}
