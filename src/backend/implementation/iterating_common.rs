//! Shared types and utilities for zarr iterators (sync and async).
//!
//! Both `ZarrIterator` and `IcechunkIterator` use the same chunk-planning and
//! batch-postprocessing logic; this module centralizes the common pieces.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use snafu::ResultExt;

use crate::FromIStr;
use crate::IStr;
use crate::chunk_plan::ChunkSubset;
use crate::chunk_plan::OwnedGridGroup;
use crate::chunk_plan::StreamingBatch;
use crate::errors::BackendError;
use crate::errors::PolarsSnafu;
use crate::meta::ZarrMeta;
use crate::shared::{
    combine_chunk_dataframes,
    restructure_to_structs,
};

/// Maximum chunks to read per batch iteration (avoids unbounded memory growth).
pub(crate) const CHUNKS_PER_BATCH_LIMIT: usize =
    100;

/// Default batch size in rows when not specified.
pub(crate) const DEFAULT_BATCH_SIZE: usize =
    10_000;

/// Top-level boolean literal after stripping [`Expr::Alias`], if any.
pub(crate) fn expr_top_literal_bool(
    expr: &Expr,
) -> Option<bool> {
    match expr {
        Expr::Literal(LiteralValue::Scalar(
            s,
        )) => match s.value() {
            AnyValue::Boolean(b) => Some(*b),
            _ => None,
        },
        Expr::Alias(inner, _) => {
            expr_top_literal_bool(inner)
        }
        _ => None,
    }
}

/// Legacy sequential batching (non-join-closed) state.
pub(crate) struct LegacyBatchState {
    pub current_group_idx: usize,
    pub current_chunk_idx: usize,
    pub current_batch: Vec<DataFrame>,
    pub current_batch_rows: usize,
}

/// Join-closed plan vs sequential streaming.
pub(crate) enum StreamingSchedule {
    Legacy(LegacyBatchState),
    JoinClosed {
        batches: Vec<StreamingBatch>,
        cursor: usize,
    },
}

/// Iterator state shared by both sync and async zarr iterators.
pub(crate) struct IteratorState {
    pub grid_groups: Vec<OwnedGridGroup>,
    pub schedule: StreamingSchedule,

    pub total_rows_yielded: usize,
    pub num_rows_limit: Option<usize>,

    pub meta: Arc<ZarrMeta>,
    /// Polars-requested columns in **pushdown order** (`None` = full schema).
    pub output_columns: Option<Vec<IStr>>,
    /// Superset for chunk reads (predicate cols, all dims, struct expansion, …).
    pub expanded_with_columns:
        Option<BTreeSet<IStr>>,

    pub predicate: Expr,

    /// When set, yield one empty (zero-row) batch with the correct output schema then clear.
    pub emit_empty_schema_once: bool,
}

/// Final column projection for streaming batches.
///
/// When Polars passes a narrowed list, keep that order and append **1D non-dim**
/// extras from the read superset (e.g. CF lat/lon omitted from IO callback).
/// Index dimensions themselves stay in the DataFrame through filter, then are
/// dropped here when not listed. When the callback passes `None`, use full
/// [`ZarrMeta::tidy_column_order`] plus any extra expanded paths.
pub(crate) fn output_columns_for_streaming_batch(
    meta: &ZarrMeta,
    polars_requested: Option<&BTreeSet<IStr>>,
    expanded: Option<&BTreeSet<IStr>>,
) -> Option<Vec<IStr>> {
    let all_dims: BTreeSet<IStr> = meta
        .dim_analysis
        .all_dims
        .iter()
        .cloned()
        .collect();

    let append_from_expanded =
        |out: &mut Vec<IStr>,
         seen: &mut BTreeSet<IStr>| {
            let Some(exp) = expanded else {
                return;
            };
            if polars_requested.is_none() {
                for name in exp {
                    if seen.insert(name.clone()) {
                        out.push(name.clone());
                    }
                }
                return;
            }
            for name in exp {
                if seen.contains(name) {
                    continue;
                }
                if all_dims.contains(name) {
                    continue;
                }
                let Some(vm) =
                    meta.array_by_path(name)
                else {
                    continue;
                };
                if vm.shape.len() != 1 {
                    continue;
                }
                seen.insert(name.clone());
                out.push(name.clone());
            }
        };

    match polars_requested {
        None => {
            let mut out =
                meta.tidy_column_order(None);
            let mut seen: BTreeSet<IStr> =
                out.iter().cloned().collect();
            append_from_expanded(
                &mut out, &mut seen,
            );
            Some(out)
        }
        Some(req) => {
            let mut out: Vec<IStr> =
                req.iter().cloned().collect();
            let mut seen: BTreeSet<IStr> =
                out.iter().cloned().collect();
            append_from_expanded(
                &mut out, &mut seen,
            );
            Some(out)
        }
    }
}

/// Keep only columns Polars asked for; order matches `output_columns`.
fn project_to_polars_output(
    df: DataFrame,
    output_columns: Option<&[IStr]>,
) -> Result<DataFrame, BackendError> {
    let Some(cols) = output_columns else {
        return Ok(df);
    };
    if cols.is_empty() {
        return Ok(df);
    }
    let names: Vec<PlSmallStr> = cols
        .iter()
        .filter(|c| df.column(c.as_ref()).is_ok())
        .map(|c| PlSmallStr::from_istr(*c))
        .collect();
    if names.is_empty() {
        return Ok(df);
    }
    df.select(&names).context(PolarsSnafu {
        message:
            "Error projecting to polars output"
                .to_string(),
    })
}

/// Collect the next batch of chunk indices to read from the current group.
///
/// Advances `current_chunk_idx` and returns up to `CHUNKS_PER_BATCH_LIMIT` chunks,
/// or until `current_batch_rows` would exceed `batch_size`, or the group is exhausted.
///
/// Returns `(chunks_to_read, num_chunks_advanced)`.
#[inline]
pub(crate) fn collect_chunks_for_batch(
    group: &OwnedGridGroup,
    current_chunk_idx: &mut usize,
    current_batch_rows: usize,
    batch_size: usize,
) -> Vec<(Vec<u64>, Option<ChunkSubset>)> {
    let mut chunks_to_read = Vec::new();
    while *current_chunk_idx
        < group.chunk_indices.len()
        && current_batch_rows < batch_size
    {
        let ci = *current_chunk_idx;
        chunks_to_read.push((
            group.chunk_indices[ci].clone(),
            group.chunk_subsets[ci].clone(),
        ));
        *current_chunk_idx += 1;

        if chunks_to_read.len()
            >= CHUNKS_PER_BATCH_LIMIT
        {
            break;
        }
    }
    chunks_to_read
}

pub(crate) fn merge_batch_dataframes(
    batch_dfs: Vec<DataFrame>,
    meta: &ZarrMeta,
) -> Result<DataFrame, BackendError> {
    if batch_dfs.is_empty() {
        let keys: Vec<IStr> =
            meta.all_array_paths();
        Ok(DataFrame::empty_with_schema(
            &meta.tidy_schema(Some(
                keys.as_slice(),
            )),
        ))
    } else if batch_dfs.len() == 1 {
        Ok(batch_dfs.into_iter().next().unwrap())
    } else {
        combine_chunk_dataframes(batch_dfs, meta)
    }
}

/// Combine chunk DataFrames, apply predicate filter, row limit, and restructure.
///
/// Shared post-processing for both sync and async iterators.
pub(crate) fn combine_and_postprocess_batch(
    batch_dfs: Vec<DataFrame>,
    state: &mut IteratorState,
    // enrich: E,
) -> Result<DataFrame, BackendError> {
    let combined = merge_batch_dataframes(
        batch_dfs,
        &state.meta,
    )?;

    // let combined = enrich(combined)?;

    let result = combined
        .lazy()
        .filter(state.predicate.clone())
        .collect()
        .context(PolarsSnafu {
            message: "Error filtering batch"
                .to_string(),
        })?;

    let result = if let Some(limit) =
        state.num_rows_limit
    {
        let remaining = limit.saturating_sub(
            state.total_rows_yielded,
        );
        if remaining < result.height() {
            result.slice(0, remaining)
        } else {
            result
        }
    } else {
        result
    };

    state.total_rows_yielded += result.height();

    let result = if state.meta.is_hierarchical() {
        restructure_to_structs(
            &result,
            &state.meta,
        )?
    } else {
        result
    };

    project_to_polars_output(
        result,
        state.output_columns.as_deref(),
    )
}

/// Zero-row batch matching streaming output schema (e.g. unsatisfiable filter folded to `false`).
pub(crate) fn empty_streaming_schema_batch(
    state: &IteratorState,
) -> Result<DataFrame, BackendError> {
    let keys: Vec<IStr> =
        match &state.output_columns {
            Some(cols) => cols.clone(),
            None => {
                state.meta.tidy_column_order(None)
            }
        };
    let df = DataFrame::empty_with_schema(
        &state
            .meta
            .tidy_schema(Some(keys.as_slice())),
    );
    let df = if state.meta.is_hierarchical() {
        restructure_to_structs(&df, &state.meta)?
    } else {
        df
    };
    project_to_polars_output(
        df,
        state.output_columns.as_deref(),
    )
}
