use super::compile_node::compile_node;
use super::errors::{CompileError, CoordIndexResolver};
use super::literals::{and_nodes, or_nodes};
use super::plan::ChunkPlanNode;
use super::prelude::*;

pub(super) fn compile_selector(
    selector: &Selector,
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    match selector {
        Selector::Union(left, right) => {
            let left_node = compile_node(
                left.as_ref().clone().as_expr(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            let right_node = compile_node(
                right.as_ref().clone().as_expr(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            Ok(or_nodes(left_node, right_node))
        }
        Selector::Difference(left, right) => {
            let left_node = compile_node(
                left.as_ref().clone().as_expr(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            let right_node = compile_node(
                right.as_ref().clone().as_expr(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            Ok(and_nodes(left_node, right_node))
        }
        Selector::ExclusiveOr(left, right) => {
            let left_node = compile_node(
                left.as_ref().clone().as_expr(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            let right_node = compile_node(
                right.as_ref().clone().as_expr(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            Ok(or_nodes(left_node, right_node))
        }
        Selector::Intersect(left, right) => {
            let left_node = compile_node(
                left.as_ref().clone().as_expr(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            let right_node = compile_node(
                right.as_ref().clone().as_expr(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            Ok(and_nodes(left_node, right_node))
        }
        Selector::Empty => Ok(ChunkPlanNode::Empty),
        Selector::ByName { .. }
        | Selector::ByIndex { .. }
        | Selector::Matches(_)
        | Selector::ByDType(_)
        | Selector::Wildcard => Ok(ChunkPlanNode::AllChunks),
    }
}
