//! Shared types and utilities for zarr iterators (sync and async).
//!
//! Both `ZarrIterator` and `IcechunkIterator` are thin wrappers around the
//! tree-driven [`crate::chunk_plan::indexing::grid_join_reader::BatchPlanner`].

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use snafu::ResultExt;

use crate::chunk_plan::indexing::grid_join_reader::{BatchPlan, BatchPlanner};
use crate::chunk_plan::GridJoinTree;
use crate::errors::{BackendError, PolarsSnafu};
use crate::meta::ZarrMeta;
use crate::shared::FromManyIstrs;
use crate::shared::IStr;
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

/// Iterator state shared by both sync and async zarr iterators.
pub(crate) struct IteratorState {
    /// The join tree owns its leaves; planners and `flatten_reads` borrow from
    /// `tree.leaves()`.
    pub tree: Option<GridJoinTree>,
    /// Pre-computed batch plans (cheap in-memory metadata describing reads).
    pub batches: Vec<BatchPlan>,
    /// Cursor into `batches`.
    pub cursor: usize,

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

/// Pre-compute every [`BatchPlan`] for `tree` at the configured `batch_size`.
pub(crate) fn build_batches(
    tree: &GridJoinTree,
    batch_size: usize,
) -> Vec<BatchPlan> {
    let mut planner = BatchPlanner::new(tree);
    let mut out = Vec::new();
    while let Some(plan) =
        planner.next_batch(batch_size)
    {
        out.push(plan);
    }
    out
}

/// Final column projection for streaming batches.
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
                    if seen.insert(*name) {
                        out.push(*name);
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
                seen.insert(*name);
                out.push(*name);
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
pub(crate) fn project_to_polars_output(
    df: DataFrame,
    output_columns: Option<&[IStr]>,
) -> Result<DataFrame, BackendError> {
    let Some(cols) = output_columns else {
        return Ok(df);
    };
    if cols.is_empty() {
        return Ok(df);
    }
    let names: Vec<PlSmallStr> =
        Vec::<PlSmallStr>::from_istrs(
            cols.iter().cloned(),
        );
    let filtered_names: Vec<PlSmallStr> = names
        .into_iter()
        .filter(|c| df.column(c.as_ref()).is_ok())
        .collect();

    if filtered_names.is_empty() {
        return Ok(df);
    }
    df.select(&filtered_names)
        .context(PolarsSnafu {
        message:
            "Error projecting to polars output"
                .to_string(),
    })
}

/// Apply predicate filter, row limit, and restructuring to a freshly assembled
/// batch DataFrame.
pub(crate) fn postprocess_batch(
    df: DataFrame,
    state: &mut IteratorState,
) -> Result<DataFrame, BackendError> {
    let result = df
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
    project_to_polars_output(
        df,
        state.output_columns.as_deref(),
    )
}

/// Convenience: count distinct chunks across pre-built batches.
pub(crate) fn distinct_chunks_in_batches(
    batches: &[BatchPlan],
) -> usize {
    let mut seen: BTreeSet<(usize, usize)> =
        BTreeSet::new();
    for b in batches {
        for slab in &b.batch.slabs {
            for &slot in &slab.chunk_slots {
                seen.insert((
                    slab.leaf_idx,
                    slot,
                ));
            }
        }
    }
    seen.len()
}
