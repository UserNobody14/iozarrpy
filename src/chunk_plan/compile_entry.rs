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
use crate::chunk_plan::exprs::CompileError;
use crate::chunk_plan::exprs::LazyCompileCtx;
use crate::chunk_plan::exprs::compile_node_lazy;
use crate::chunk_plan::indexing::MonotonicCoordResolver;
use crate::chunk_plan::indexing::SyncCoordResolver;
use crate::chunk_plan::indexing::lazy_materialize::{
    MergedCache, collect_requests_with_meta, materialize,
};
use crate::chunk_plan::indexing::lazy_selection::LazyDatasetSelection;
use crate::chunk_plan::indexing::selection::DatasetSelection;
use crate::chunk_plan::prelude::*;
use crate::meta::ZarrMeta;

/// Statistics about the planning process.
pub(crate) struct PlannerStats {
    /// Number of coordinate array reads performed.
    pub(crate) coord_reads: u64,
}

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
// Asynchronous resolution
// ============================================================================

/// Resolve a lazy selection using synchronous I/O and unified metadata.
pub(crate) fn resolve_lazy_selection_sync_unified(
    lazy_selection: &LazyDatasetSelection,
    meta: &ZarrMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
) -> Result<
    (DatasetSelection, PlannerStats),
    CompileError,
> {
    let legacy_meta = meta.planning_meta();
    let (dims, dim_lengths) =
        compute_dims_and_lengths_unified(meta);

    let (requests, immediate_cache) =
        collect_requests_with_meta(
            lazy_selection,
            &legacy_meta,
            &dim_lengths,
            &dims,
        );

    let mut resolver =
        MonotonicCoordResolver::new(
            &legacy_meta,
            store,
        );
    let resolved_cache = resolver
        .resolve_batch(&requests, &legacy_meta);

    let merged = MergedCache::new(
        &*resolved_cache,
        &immediate_cache,
    );
    let selection = materialize(
        lazy_selection,
        &legacy_meta,
        &merged,
    )
    .map_err(|e| {
        CompileError::Unsupported(format!(
            "materialization failed: {}",
            e
        ))
    })?;

    drop(resolved_cache);

    let stats = PlannerStats {
        coord_reads: resolver.coord_read_count(),
    };

    Ok((selection, stats))
}

/// Compile an expression to a dataset selection (sync, unified ZarrMeta).
pub(crate) fn compile_expr_to_dataset_selection_unified(
    expr: &Expr,
    meta: &ZarrMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
) -> Result<
    (DatasetSelection, PlannerStats),
    CompileError,
> {
    let lazy_selection =
        compile_expr_to_lazy_selection_unified(
            expr, meta,
        )?;
    resolve_lazy_selection_sync_unified(
        &lazy_selection,
        meta,
        store,
    )
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
    let (dims, dim_lengths) =
        compute_dims_and_lengths_unified(meta);
    let vars = legacy_meta.data_vars.clone();
    let mut ctx = LazyCompileCtx::new(
        &legacy_meta,
        Some(meta),
        &dims,
        &dim_lengths,
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

/// Compile an expression to a grouped chunk plan (sync, unified ZarrMeta).
pub(crate) fn compile_expr_to_grouped_chunk_plan_unified(
    expr: &Expr,
    meta: &ZarrMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
) -> Result<
    (GroupedChunkPlan, PlannerStats),
    CompileError,
> {
    let (selection, stats) =
        compile_expr_to_dataset_selection_unified(
            expr,
            meta,
            store.clone(),
        )?;
    let grouped_plan =
        selection_to_grouped_chunk_plan_unified(
            &selection, meta, store,
        )?;
    Ok((grouped_plan, stats))
}
