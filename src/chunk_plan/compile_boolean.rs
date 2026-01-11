use super::compile_node::compile_node;
use super::compile_is_between::compile_is_between;
use super::compile_is_in::compile_is_in;
use super::errors::{CompileError, CoordIndexResolver};
use super::literals::{literal_anyvalue, strip_wrappers};
use super::plan::ChunkPlanNode;
use super::prelude::*;

pub(super) fn compile_boolean_function(
    bf: &BooleanFunction,
    input: &[Expr],
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    match bf {
        BooleanFunction::Not => {
            let [arg] = input else {
                return Err(CompileError::Unsupported(format!(
                    "unsupported boolean function: {:?}",
                    bf
                )));
            };
            // Try constant fold first.
            if let Expr::Literal(lit) = strip_wrappers(arg) {
                return match literal_anyvalue(lit) {
                    Some(AnyValue::Boolean(true)) => Ok(ChunkPlanNode::Empty),
                    Some(AnyValue::Boolean(false)) => Ok(ChunkPlanNode::AllChunks),
                    Some(AnyValue::Null) => Ok(ChunkPlanNode::Empty),
                    _ => Ok(ChunkPlanNode::AllChunks),
                };
            }

            // If the inner predicate is known to match nothing, NOT(...) matches everything.
            // Otherwise we can't represent complements with current plan nodes.
            match compile_node(arg, meta, dims, grid_shape, regular_chunk_shape, resolver)
                .unwrap_or(ChunkPlanNode::AllChunks)
            {
                ChunkPlanNode::Empty => Ok(ChunkPlanNode::AllChunks),
                _ => Ok(ChunkPlanNode::AllChunks),
            }
        }
        BooleanFunction::IsNull | BooleanFunction::IsNotNull => {
            let [arg] = input else {
                return Err(CompileError::Unsupported(format!(
                    "unsupported boolean function: {:?}",
                    bf
                )));
            };

            // Constant fold when possible; otherwise don't constrain.
            if let Expr::Literal(lit) = strip_wrappers(arg) {
                let is_null = matches!(literal_anyvalue(lit), Some(AnyValue::Null));
                let keep = match bf {
                    BooleanFunction::IsNull => is_null,
                    BooleanFunction::IsNotNull => !is_null,
                    _ => unreachable!(),
                };
                return Ok(if keep {
                    ChunkPlanNode::AllChunks
                } else {
                    ChunkPlanNode::Empty
                });
            }
            Ok(ChunkPlanNode::AllChunks)
        }
        _ => {
            // Future-proof handling for optional Polars boolean features without hard-referencing
            // cfg-gated variants (e.g. `is_in`, `is_between`).
            let name = bf.to_string();
            match name.as_str() {
                "is_between" => {
                    compile_is_between(input, meta, dims, grid_shape, regular_chunk_shape, resolver)
                }
                "is_in" => {
                    compile_is_in(input, meta, dims, grid_shape, regular_chunk_shape, resolver)
                }
                _ => Ok(ChunkPlanNode::AllChunks),
            }
        }
    }
}

