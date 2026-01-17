use crate::chunk_plan::exprs::compile_node::compile_node;
use crate::chunk_plan::exprs::compile_ctx::CompileCtx;
use crate::chunk_plan::exprs::errors::{CompileError, CoordIndexResolver};
use crate::chunk_plan::indexing::monotonic_scalar::MonotonicCoordResolver;
use crate::chunk_plan::indexing::plan::{ChunkPlan, ChunkPlanNode};
use crate::chunk_plan::prelude::*;
use crate::chunk_plan::indexing::selection::DatasetSelection;
use crate::chunk_plan::indexing::selection_to_chunks::plan_data_array_chunk_indices;

pub(crate) struct PlannerStats {
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

pub(crate) fn compile_expr_to_dataset_selection(
    expr: &Expr,
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

    let vars = default_vars_for_dataset_selection(meta);
    let mut resolver: MonotonicCoordResolver<'_> = MonotonicCoordResolver::new(meta, store);
    let mut ctx = CompileCtx {
        meta,
        dims: &dims,
        dim_lengths: &dim_lengths,
        vars: &vars,
        resolver: &mut resolver,
    };
    let selection = compile_node(expr, &mut ctx)?;
    let stats = PlannerStats {
        coord_reads: resolver.coord_read_count(),
    };
    Ok((selection, stats))
}

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

    let (selection, stats) = compile_expr_to_dataset_selection(expr, meta, store.clone(), primary_var)?;

    let primary = Array::open(store.clone(), &primary_meta.path)
        .map_err(|e| CompileError::Unsupported(format!("failed to open primary array: {:?}", e)))?;
    let grid_shape = primary.chunk_grid().grid_shape().to_vec();
    let zero = vec![0u64; primary.dimensionality()];
    let chunk_shape_nz = primary
        .chunk_shape(&zero)
        .map_err(|e| CompileError::Unsupported(e.to_string()))?;
    let chunk_shape = chunk_shape_nz.iter().map(|nz| nz.get()).collect::<Vec<_>>();

    let chunk_set = selection
        .0
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
}

