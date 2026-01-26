//! Entry points for chunk planning compilation.
//!
//! This module provides the main entry points for compiling Polars expressions
//! into chunk plans. The compilation uses a lazy approach:
//! 1. Compile expression to `LazyDatasetSelection` (no I/O)
//! 2. Collect resolution requests and batch-resolve value ranges to index ranges
//! 3. Materialize to concrete `DatasetSelection`
//!
//! This enables efficient I/O batching and concurrent resolution for async stores.

use crate::chunk_plan::exprs::compile_node_lazy;
use crate::chunk_plan::exprs::LazyCompileCtx;
use crate::chunk_plan::exprs::CompileError;
use crate::chunk_plan::indexing::DataArraySelection;
use crate::chunk_plan::indexing::monotonic_scalar::MonotonicCoordResolver;
use crate::chunk_plan::indexing::plan::{ChunkPlan, ChunkPlanNode};
use crate::chunk_plan::prelude::*;
use crate::chunk_plan::indexing::selection::DatasetSelection;
use crate::chunk_plan::indexing::selection_to_chunks::plan_data_array_chunk_indices;
use crate::chunk_plan::indexing::lazy_selection::LazyDatasetSelection;
use crate::chunk_plan::indexing::lazy_materialize::{
    collect_requests_with_meta, materialize, MergedCache,
};
use crate::chunk_plan::indexing::resolver_traits::SyncCoordResolver;
use crate::chunk_plan::indexing::monotonic_async::AsyncMonotonicResolver;
use crate::chunk_plan::indexing::resolver_traits::AsyncCoordResolver;

/// Statistics about the planning process.
pub(crate) struct PlannerStats {
    /// Number of coordinate array reads performed.
    pub(crate) coord_reads: u64,
}

fn default_vars_for_dataset_selection(meta: &ZarrDatasetMeta) -> Vec<String> {
    let mut out: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    out.extend(meta.data_vars.iter().cloned());
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
    primary_var: &str,
) -> Result<LazyDatasetSelection, CompileError> {
    let Some(primary_meta) = meta.arrays.get(primary_var) else {
        return Err(CompileError::MissingPrimaryDims(format!(
            "primary variable '{}' not found",
            primary_var
        )));
    };

    let dims = if !primary_meta.dims.is_empty() {
        primary_meta.dims.clone()
    } else {
        meta.dims.clone()
    };

    let dim_lengths = if primary_meta.shape.len() == dims.len() {
        primary_meta.shape.clone()
    } else {
        return Err(CompileError::MissingPrimaryDims(format!(
            "primary variable '{}' has shape {:?} incompatible with dims {:?}",
            primary_var, primary_meta.shape, dims
        )));
    };

    let vars = default_vars_for_dataset_selection(meta);
    let mut ctx = LazyCompileCtx::new(meta, &dims, &dim_lengths, &vars);
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
    primary_var: &str,
) -> Result<(DatasetSelection, PlannerStats), CompileError> {
    let Some(primary_meta) = meta.arrays.get(primary_var) else {
        return Err(CompileError::MissingPrimaryDims(format!(
            "primary variable '{}' not found",
            primary_var
        )));
    };

    let dims = if !primary_meta.dims.is_empty() {
        primary_meta.dims.clone()
    } else {
        meta.dims.clone()
    };

    let dim_lengths = if primary_meta.shape.len() == dims.len() {
        primary_meta.shape.clone()
    } else {
        return Err(CompileError::MissingPrimaryDims(format!(
            "primary variable '{}' has shape {:?} incompatible with dims {:?}",
            primary_var, primary_meta.shape, dims
        )));
    };

    // Collect requests, handling index-only dimensions immediately
    let (requests, immediate_cache) =
        collect_requests_with_meta(lazy_selection, meta, &dim_lengths, &dims);

    // Resolve remaining requests using the monotonic resolver
    let mut resolver = MonotonicCoordResolver::new(meta, store);
    let resolved_cache = resolver.resolve_batch(&requests, meta);

    // Merge caches and materialize
    let merged = MergedCache::new(&*resolved_cache, &immediate_cache);
    let selection = materialize(lazy_selection, &merged).map_err(|e| {
        CompileError::Unsupported(format!("materialization failed: {}", e))
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
    primary_var: &str,
) -> Result<(DatasetSelection, PlannerStats), CompileError> {
    let lazy_selection = compile_expr_to_lazy_selection(expr, meta, primary_var)?;
    resolve_lazy_selection_sync(&lazy_selection, meta, store, primary_var)
}

/// Compile an expression to a chunk plan (synchronous).
///
/// This is the main entry point for synchronous chunk planning.
pub(crate) fn compile_expr_to_chunk_plan(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
    primary_var: &str,
) -> Result<(ChunkPlan, PlannerStats), CompileError> {
    let Some(primary_meta) = meta.arrays.get(primary_var) else {
        return Err(CompileError::MissingPrimaryDims(format!(
            "primary variable '{}' not found",
            primary_var
        )));
    };

    let (selection, stats) =
        compile_expr_to_dataset_selection(expr, meta, store.clone(), primary_var)?;

    let primary = Array::open(store.clone(), &primary_meta.path)
        .map_err(|e| CompileError::Unsupported(format!("failed to open primary array: {:?}", e)))?;
    let grid_shape = primary.chunk_grid().grid_shape().to_vec();
    let zero = vec![0u64; primary.dimensionality()];
    let chunk_shape_nz = primary
        .chunk_shape(&zero)
        .map_err(|e| CompileError::Unsupported(e.to_string()))?;
    let chunk_shape = chunk_shape_nz.iter().map(|nz| nz.get()).collect::<Vec<_>>();

    if let DatasetSelection::Selection(selection) = selection {
        let chunk_set = selection
            .get(primary_var)
            .map(|sel| {
                plan_data_array_chunk_indices(
                    sel,
                    &primary_meta.dims,
                    &primary_meta.shape,
                    &grid_shape,
                    &chunk_shape,
                )
            })
            .unwrap_or_default();

        let indices: Vec<Vec<u64>> = chunk_set.into_iter().collect();
        let plan = ChunkPlan::from_root(grid_shape, ChunkPlanNode::Explicit(indices));
        Ok((plan, stats))
    } else if let DatasetSelection::NoSelectionMade = selection {
        let plan = ChunkPlan::all(grid_shape);
        Ok((plan, stats))
    } else {
        let plan = ChunkPlan::from_root(grid_shape, ChunkPlanNode::Empty);
        Ok((plan, stats))
    }
}

// ============================================================================
// Asynchronous resolution
// ============================================================================

/// Resolve a lazy selection using async I/O with concurrent coordinate resolution.
pub(crate) async fn resolve_lazy_selection_async(
    lazy_selection: &LazyDatasetSelection,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
    primary_var: &str,
) -> Result<(DatasetSelection, PlannerStats), CompileError> {
    let Some(primary_meta) = meta.arrays.get(primary_var) else {
        return Err(CompileError::MissingPrimaryDims(format!(
            "primary variable '{}' not found",
            primary_var
        )));
    };

    let dims = if !primary_meta.dims.is_empty() {
        primary_meta.dims.clone()
    } else {
        meta.dims.clone()
    };

    let dim_lengths = if primary_meta.shape.len() == dims.len() {
        primary_meta.shape.clone()
    } else {
        return Err(CompileError::MissingPrimaryDims(format!(
            "primary variable '{}' has shape {:?} incompatible with dims {:?}",
            primary_var, primary_meta.shape, dims
        )));
    };

    // Collect requests, handling index-only dimensions immediately
    let (requests, immediate_cache) =
        collect_requests_with_meta(lazy_selection, meta, &dim_lengths, &dims);

    // Resolve remaining requests using the async monotonic resolver
    let resolver = AsyncMonotonicResolver::new(store);
    let resolved_cache = resolver.resolve_batch(requests, meta).await;

    // Merge caches and materialize
    let merged = MergedCache::new(&*resolved_cache, &immediate_cache);
    let selection = materialize(lazy_selection, &merged).map_err(|e| {
        CompileError::Unsupported(format!("materialization failed: {}", e))
    })?;

    // Async resolver doesn't track read count currently
    let stats = PlannerStats { coord_reads: 0 };

    Ok((selection, stats))
}

/// Compile an expression to a dataset selection (async).
pub(crate) async fn compile_expr_to_dataset_selection_async(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
    primary_var: &str,
) -> Result<(DatasetSelection, PlannerStats), CompileError> {
    let lazy_selection = compile_expr_to_lazy_selection(expr, meta, primary_var)?;
    resolve_lazy_selection_async(&lazy_selection, meta, store, primary_var).await
}

/// Compile an expression to a chunk plan (async).
///
/// This is the main entry point for async chunk planning with concurrent I/O.
pub(crate) async fn compile_expr_to_chunk_plan_async(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
    primary_var: &str,
) -> Result<(ChunkPlan, PlannerStats), CompileError> {
    let Some(primary_meta) = meta.arrays.get(primary_var) else {
        return Err(CompileError::MissingPrimaryDims(format!(
            "primary variable '{}' not found",
            primary_var
        )));
    };

    let (selection, stats) =
        compile_expr_to_dataset_selection_async(expr, meta, store.clone(), primary_var).await?;

    let primary = Array::async_open(store.clone(), &primary_meta.path)
        .await
        .map_err(|e| CompileError::Unsupported(format!("failed to open primary array: {:?}", e)))?;
    let grid_shape = primary.chunk_grid().grid_shape().to_vec();
    let zero = vec![0u64; primary.dimensionality()];
    let chunk_shape_nz = primary
        .chunk_shape(&zero)
        .map_err(|e| CompileError::Unsupported(e.to_string()))?;
    let chunk_shape = chunk_shape_nz.iter().map(|nz| nz.get()).collect::<Vec<_>>();

    if let DatasetSelection::Selection(selection) = selection {
        let chunk_set = selection
            .get(primary_var)
            .map(|sel| {
                plan_data_array_chunk_indices(
                    sel,
                    &primary_meta.dims,
                    &primary_meta.shape,
                    &grid_shape,
                    &chunk_shape,
                )
            })
            .unwrap_or_default();

        let indices: Vec<Vec<u64>> = chunk_set.into_iter().collect();
        let plan = ChunkPlan::from_root(grid_shape, ChunkPlanNode::Explicit(indices));
        Ok((plan, stats))
    } else if let DatasetSelection::NoSelectionMade = selection {
        let plan = ChunkPlan::all(grid_shape);
        Ok((plan, stats))
    } else {
        let plan = ChunkPlan::from_root(grid_shape, ChunkPlanNode::Empty);
        Ok((plan, stats))
    }
}
