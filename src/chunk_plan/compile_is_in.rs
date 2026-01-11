use super::compile_cmp::compile_cmp;
use super::errors::{CompileError, CoordIndexResolver};
use super::expr_utils::expr_to_col_name;
use super::literals::{or_nodes, strip_wrappers};
use super::plan::ChunkPlanNode;
use super::prelude::*;

pub(super) fn compile_is_in(
    input: &[Expr],
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    let [expr, list] = input else {
        return Err(CompileError::Unsupported(format!(
            "unsupported is_in expression: {:?}",
            input
        )));
    };
    let Some(col) = expr_to_col_name(expr) else {
        return Ok(ChunkPlanNode::AllChunks);
    };

    let Expr::Literal(list_lit) = strip_wrappers(list) else {
        return Ok(ChunkPlanNode::AllChunks);
    };

    match list_lit {
        LiteralValue::Series(s) => {
            let series = &**s;
            // Prevent pathological unions for huge lists.
            if series.len() > 4096 {
                return Ok(ChunkPlanNode::AllChunks);
            }

            let mut out: Option<ChunkPlanNode> = None;
            for av in series.iter() {
                let av = av.into_static();
                if matches!(av, AnyValue::Null) {
                    // Null membership semantics depend on `nulls_equal`; we avoid constraining.
                    return Ok(ChunkPlanNode::AllChunks);
                }

                let lit = LiteralValue::Scalar(Scalar::new(series.dtype().clone(), av));
                let node = compile_cmp(
                    col,
                    Operator::Eq,
                    &lit,
                    meta,
                    dims,
                    grid_shape,
                    regular_chunk_shape,
                    resolver,
                )
                .unwrap_or(ChunkPlanNode::AllChunks);

                // If any element falls back to AllChunks, the whole IN predicate becomes unconstrainable.
                if matches!(node, ChunkPlanNode::AllChunks) {
                    return Ok(ChunkPlanNode::AllChunks);
                }
                out = Some(match out.take() {
                    None => node,
                    Some(acc) => or_nodes(acc, node),
                });
            }
            Ok(out.unwrap_or(ChunkPlanNode::Empty))
        }
        _ => Ok(ChunkPlanNode::AllChunks),
    }
}

