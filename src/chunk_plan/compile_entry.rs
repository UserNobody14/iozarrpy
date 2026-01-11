use super::errors::CoordIndexResolver;
use super::compile_node::compile_node;
use super::errors::CompileError;
use super::monotonic_scalar::MonotonicCoordResolver;
use super::plan::ChunkPlan;
use super::prelude::*;

pub(crate) struct PlannerStats {
    pub(crate) coord_reads: u64,
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
    let dims = if !primary_meta.dims.is_empty() {
        primary_meta.dims.clone()
    } else {
        meta.dims.clone()
    };

    let primary = Array::open(store.clone(), &primary_meta.path)
        .map_err(|e| CompileError::Unsupported(format!("failed to open primary array: {:?}", e)))?;
    let grid_shape = primary.chunk_grid().grid_shape().to_vec();
    let zero = vec![0u64; primary.dimensionality()];
    let chunk_shape_nz = primary
        .chunk_shape(&zero)
        .map_err(|e| CompileError::Unsupported(e.to_string()))?;
    let regular_chunk_shape = chunk_shape_nz.iter().map(|nz| nz.get()).collect::<Vec<_>>();

    let mut resolver = MonotonicCoordResolver::new(meta, store);
    let root = compile_node(
        expr,
        meta,
        &dims,
        &grid_shape,
        &regular_chunk_shape,
        &mut resolver,
    )?;
    let grid_shape_vec = grid_shape.to_vec();
    let plan = ChunkPlan::from_root(dims, grid_shape_vec, regular_chunk_shape, root);
    let stats = PlannerStats {
        coord_reads: resolver.coord_read_count(),
    };
    Ok((plan, stats))
}

