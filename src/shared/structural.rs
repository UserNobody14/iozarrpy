use polars::prelude::*;
use snafu::ResultExt;
use std::collections::BTreeSet;

use crate::errors::{BackendResult, PolarsSnafu};
use crate::meta::path::ZarrPath;
use crate::meta::ZarrMeta;
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
