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
use crate::chunk_plan::indexing::AsyncCoordResolver;
use crate::chunk_plan::indexing::AsyncMonotonicResolver;
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

/// Compute dimensions and their lengths from dataset metadata.
///
/// This replaces the old pattern of extracting dims from a "primary variable".
/// Dimensions are taken from `meta.dims` and lengths are derived from
/// coordinate arrays or the first matching data array.
fn compute_dims_and_lengths(
    meta: &ZarrDatasetMeta,
) -> (Vec<IStr>, Vec<u64>) {
    let dims = meta.dims.clone();
    let dim_lengths: Vec<u64> = dims
        .iter()
        .map(|d| {
            // First try: look for a 1D coordinate array with this dimension name
            if let Some(coord_array) =
                meta.arrays.get(d)
            {
                if let Some(&len) =
                    coord_array.shape.first()
                {
                    return len;
                }
            }
            // Second try: look for any array that has this dimension and get its size
            for (_, arr_meta) in &meta.arrays {
                if let Some(pos) = arr_meta
                    .dims
                    .iter()
                    .position(|dim| dim == d)
                {
                    if pos < arr_meta.shape.len()
                    {
                        return arr_meta.shape
                            [pos];
                    }
                }
            }
            // Fallback
            1
        })
        .collect();
    (dims, dim_lengths)
}

fn compute_dims_and_lengths_unified(
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

fn default_vars_for_dataset_selection(
    meta: &ZarrDatasetMeta,
) -> Vec<IStr> {
    let mut out: std::collections::BTreeSet<
        IStr,
    > = std::collections::BTreeSet::new();

    // Include root-level data vars
    out.extend(meta.data_vars.iter().cloned());

    // Include all array paths from meta.arrays (includes hierarchical paths)
    // Filter to only include paths that look like variables (not coordinate arrays)
    for (path, _arr_meta) in &meta.arrays {
        // Skip dimension coordinate arrays (1D arrays where the dim name == array name)
        let is_coord =
            meta.dims.iter().any(|d| d == path);
        if !is_coord {
            // Include hierarchical paths (contain '/')
            let path_str: &str = path.as_ref();
            if path_str.contains('/')
                && !path_str.starts_with('/')
            {
                out.insert(path.clone());
            }
        }
    }

    // Also include 1D dimension coordinate arrays (e.g. `time`, `x`, `y`) if present.
    for d in &meta.dims {
        if meta.arrays.contains_key(d) {
            out.insert(d.clone());
        }
    }
    out.into_iter().collect()
}

// ============================================================================
// Lazy compilation (no I/O)
// ============================================================================

/// Compile an expression to a lazy dataset selection (no I/O during compilation).
///
/// This function traverses the expression tree and produces a `LazyDatasetSelection`
/// containing unresolved `ValueRange` constraints. Use `resolve_lazy_selection_sync`
/// or `resolve_lazy_selection_async` to resolve and materialize the result.
pub(crate) fn compile_expr_to_lazy_selection(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
) -> Result<LazyDatasetSelection, CompileError> {
    let (dims, dim_lengths) =
        compute_dims_and_lengths(meta);

    let vars =
        default_vars_for_dataset_selection(meta);
    let mut ctx = LazyCompileCtx::new(
        meta,
        None,
        &dims,
        &dim_lengths,
        &vars,
    );
    compile_node_lazy(expr, &mut ctx)
}

// ============================================================================
// Synchronous resolution
// ============================================================================

/// Resolve a lazy selection using synchronous I/O and materialize to a concrete selection.
pub(crate) fn resolve_lazy_selection_sync(
    lazy_selection: &LazyDatasetSelection,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
) -> Result<
    (DatasetSelection, PlannerStats),
    CompileError,
> {
    let (dims, dim_lengths) =
        compute_dims_and_lengths(meta);

    // Collect requests, handling index-only dimensions immediately
    let (requests, immediate_cache) =
        collect_requests_with_meta(
            lazy_selection,
            meta,
            &dim_lengths,
            &dims,
        );

    // Resolve remaining requests using the monotonic resolver
    let mut resolver =
        MonotonicCoordResolver::new(meta, store);
    let resolved_cache =
        resolver.resolve_batch(&requests, meta);

    // Merge caches and materialize
    let merged = MergedCache::new(
        &*resolved_cache,
        &immediate_cache,
    );
    let selection = materialize(
        lazy_selection,
        meta,
        &merged,
    )
    .map_err(|e| {
        CompileError::Unsupported(format!(
            "materialization failed: {}",
            e
        ))
    })?;

    // Drop the resolved_cache before borrowing resolver immutably
    drop(resolved_cache);

    let stats = PlannerStats {
        coord_reads: resolver.coord_read_count(),
    };

    Ok((selection, stats))
}

/// Compile an expression to a dataset selection (synchronous).
///
/// This is the main entry point for synchronous chunk planning.
pub(crate) fn compile_expr_to_dataset_selection(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
) -> Result<
    (DatasetSelection, PlannerStats),
    CompileError,
> {
    let lazy_selection =
        compile_expr_to_lazy_selection(
            expr, meta,
        )?;
    resolve_lazy_selection_sync(
        &lazy_selection,
        meta,
        store,
    )
}

// ============================================================================
// Asynchronous resolution
// ============================================================================

/// Resolve a lazy selection using async I/O with concurrent coordinate resolution.
pub(crate) async fn resolve_lazy_selection_async(
    lazy_selection: &LazyDatasetSelection,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
) -> Result<
    (DatasetSelection, PlannerStats),
    CompileError,
> {
    let (dims, dim_lengths) =
        compute_dims_and_lengths(meta);

    // Collect requests, handling index-only dimensions immediately
    let (requests, immediate_cache) =
        collect_requests_with_meta(
            lazy_selection,
            meta,
            &dim_lengths,
            &dims,
        );

    // Resolve remaining requests using the async monotonic resolver
    let resolver =
        AsyncMonotonicResolver::new(store);
    let resolved_cache = resolver
        .resolve_batch(requests, meta)
        .await;

    // Merge caches and materialize
    let merged = MergedCache::new(
        &*resolved_cache,
        &immediate_cache,
    );
    let selection = materialize(
        lazy_selection,
        meta,
        &merged,
    )
    .map_err(|e| {
        CompileError::Unsupported(format!(
            "materialization failed: {}",
            e
        ))
    })?;

    // Async resolver doesn't track read count currently
    let stats = PlannerStats { coord_reads: 0 };

    Ok((selection, stats))
}

/// Resolve a lazy selection using synchronous I/O and unified metadata.
pub(crate) fn resolve_lazy_selection_sync_unified(
    lazy_selection: &LazyDatasetSelection,
    meta: &ZarrMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
) -> Result<(DatasetSelection, PlannerStats), CompileError> {
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
        MonotonicCoordResolver::new(&legacy_meta, store);
    let resolved_cache =
        resolver.resolve_batch(&requests, &legacy_meta);

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
) -> Result<(DatasetSelection, PlannerStats), CompileError> {
    let lazy_selection =
        compile_expr_to_lazy_selection_unified(expr, meta)?;
    resolve_lazy_selection_sync_unified(
        &lazy_selection,
        meta,
        store,
    )
}

/// Compile an expression to a dataset selection (async).
pub(crate) async fn compile_expr_to_dataset_selection_async(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
) -> Result<
    (DatasetSelection, PlannerStats),
    CompileError,
> {
    let lazy_selection =
        compile_expr_to_lazy_selection(
            expr, meta,
        )?;
    resolve_lazy_selection_async(
        &lazy_selection,
        meta,
        store,
    )
    .await
}

/// Resolve a lazy selection using async I/O and unified metadata.
pub(crate) async fn resolve_lazy_selection_async_unified(
    lazy_selection: &LazyDatasetSelection,
    meta: &ZarrMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
) -> Result<(DatasetSelection, PlannerStats), CompileError> {
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

    let resolver =
        AsyncMonotonicResolver::new(store);
    let resolved_cache = resolver
        .resolve_batch(requests, &legacy_meta)
        .await;

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

    let stats = PlannerStats { coord_reads: 0 };

    Ok((selection, stats))
}

/// Compile an expression to a dataset selection (async, unified ZarrMeta).
pub(crate) async fn compile_expr_to_dataset_selection_unified_async(
    expr: &Expr,
    meta: &ZarrMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
) -> Result<(DatasetSelection, PlannerStats), CompileError> {
    let lazy_selection =
        compile_expr_to_lazy_selection_unified(expr, meta)?;
    resolve_lazy_selection_async_unified(
        &lazy_selection,
        meta,
        store,
    )
    .await
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
    selection_to_grouped_chunk_plan,
    selection_to_grouped_chunk_plan_async,
    selection_to_grouped_chunk_plan_unified,
    selection_to_grouped_chunk_plan_unified_async,
};

/// Compile an expression to a grouped chunk plan (sync).
///
/// This handles heterogeneous chunk grids: variables with the same dimensions
/// but different chunk shapes will have separate ChunkPlans.
pub(crate) fn compile_expr_to_grouped_chunk_plan(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
) -> Result<
    (GroupedChunkPlan, PlannerStats),
    CompileError,
> {
    let (selection, stats) =
        compile_expr_to_dataset_selection(
            expr,
            meta,
            store.clone(),
        )?;
    let grouped_plan =
        selection_to_grouped_chunk_plan(
            &selection, meta, store,
        )?;
    Ok((grouped_plan, stats))
}

/// Compile an expression to a grouped chunk plan (async).
///
/// This handles heterogeneous chunk grids: variables with the same dimensions
/// but different chunk shapes will have separate ChunkPlans.
pub(crate) async fn compile_expr_to_grouped_chunk_plan_async(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
) -> Result<
    (GroupedChunkPlan, PlannerStats),
    CompileError,
> {
    let (selection, stats) =
        compile_expr_to_dataset_selection_async(
            expr,
            meta,
            store.clone(),
        )
        .await?;
    let grouped_plan =
        selection_to_grouped_chunk_plan_async(
            &selection, meta, store,
        )
        .await?;
    Ok((grouped_plan, stats))
}

/// Compile an expression to a grouped chunk plan (sync, unified ZarrMeta).
pub(crate) fn compile_expr_to_grouped_chunk_plan_unified(
    expr: &Expr,
    meta: &ZarrMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
) -> Result<(GroupedChunkPlan, PlannerStats), CompileError> {
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

/// Compile an expression to a grouped chunk plan (async, unified ZarrMeta).
pub(crate) async fn compile_expr_to_grouped_chunk_plan_unified_async(
    expr: &Expr,
    meta: &ZarrMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
) -> Result<(GroupedChunkPlan, PlannerStats), CompileError> {
    let (selection, stats) =
        compile_expr_to_dataset_selection_unified_async(
            expr,
            meta,
            store.clone(),
        )
        .await?;
    let grouped_plan =
        selection_to_grouped_chunk_plan_unified_async(
            &selection, meta, store,
        )
        .await?;
    Ok((grouped_plan, stats))
}
