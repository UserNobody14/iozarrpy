//! Entry points for chunk planning compilation.
//!
//! This module provides the main entry points for compiling Polars expressions
//! into chunk plans. The compilation uses a lazy approach:
//! 1. Compile expression to `LazyDatasetSelection` (no I/O)
//! 2. Collect resolution requests and batch-resolve value ranges to index ranges
//! 3. Materialize to concrete `DatasetSelection`
//!
//! This enables efficient I/O batching and concurrent resolution for async stores.

use crate::IStr;
use crate::PlannerStats;
use crate::chunk_plan::exprs::CompileError;
use crate::chunk_plan::exprs::LazyCompileCtx;
use crate::chunk_plan::exprs::compile_node_lazy;
use crate::chunk_plan::indexing::SyncCoordResolver;
use crate::chunk_plan::indexing::lazy_materialize::{
    MergedCache, collect_requests_with_meta, materialize,
};
use crate::chunk_plan::indexing::lazy_selection::LazyDatasetSelection;
use crate::chunk_plan::indexing::selection::DatasetSelection;
use crate::chunk_plan::prelude::*;
use crate::meta::ZarrMeta;

pub(crate) fn compute_dims_and_lengths_unified(
    meta: &ZarrMeta,
) -> (Vec<IStr>, Vec<u64>) {
    let dims = meta.dim_analysis.all_dims.clone();
    let dim_lengths: Vec<u64> = dims
        .iter()
        .map(|d| {
            meta.dim_analysis
                .dim_lengths
                .get(d)
                .copied()
                .unwrap_or(1)
        })
        .collect();
    (dims, dim_lengths)
}

// ============================================================================
// Unified ZarrMeta entry points
// ============================================================================

/// Compile an expression to a lazy dataset selection using unified ZarrMeta.
pub(crate) fn compile_expr_to_lazy_selection_unified(
    expr: &Expr,
    meta: &ZarrMeta,
) -> Result<LazyDatasetSelection, CompileError> {
    let legacy_meta = meta.planning_meta();
    let (dims, _dim_lengths) =
        compute_dims_and_lengths_unified(meta);
    let vars = legacy_meta.data_vars.clone();
    let mut ctx = LazyCompileCtx::new(
        &legacy_meta,
        Some(meta),
        &dims,
        &vars,
    );
    compile_node_lazy(expr, &mut ctx)
}

// ============================================================================
// GroupedChunkPlan entry points (heterogeneous chunk grids)
// ============================================================================

use crate::chunk_plan::indexing::GroupedChunkPlan;
use crate::chunk_plan::indexing::selection_to_chunks::{
    selection_to_grouped_chunk_plan_unified
};
