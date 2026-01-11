use super::compile_cmp::compile_cmp;
use super::errors::{CompileError, CoordIndexResolver};
use super::expr_utils::expr_to_col_name;
use super::literals::{and_nodes, strip_wrappers};
use super::plan::ChunkPlanNode;
use super::prelude::*;

pub(super) fn compile_is_between(
    input: &[Expr],
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    let [expr, low, high] = input else {
        return Err(CompileError::Unsupported(format!(
            "unsupported is_between expression: {:?}",
            input
        )));
    };
    let Some(col) = expr_to_col_name(expr) else {
        return Ok(ChunkPlanNode::AllChunks);
    };
    let Expr::Literal(low_lit) = strip_wrappers(low) else {
        return Ok(ChunkPlanNode::AllChunks);
    };
    let Expr::Literal(high_lit) = strip_wrappers(high) else {
        return Ok(ChunkPlanNode::AllChunks);
    };

    // Conservatively assume a closed interval (inclusive bounds) to avoid false negatives.
    let a = compile_cmp(
        col,
        Operator::GtEq,
        low_lit,
        meta,
        dims,
        grid_shape,
        regular_chunk_shape,
        resolver,
    )
    .unwrap_or(ChunkPlanNode::AllChunks);
    let b = compile_cmp(
        col,
        Operator::LtEq,
        high_lit,
        meta,
        dims,
        grid_shape,
        regular_chunk_shape,
        resolver,
    )
    .unwrap_or(ChunkPlanNode::AllChunks);
    Ok(and_nodes(a, b))
}

