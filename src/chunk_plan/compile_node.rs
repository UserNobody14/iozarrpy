use super::compile_boolean::compile_boolean_function;
use super::compile_cmp::compile_cmp;
use super::errors::CompileError;
use super::literals::{col_lit, literal_anyvalue, or_nodes, and_nodes, reverse_operator};
use super::plan::ChunkPlanNode;
use super::prelude::*;
use super::selector::compile_selector;
use super::errors::CoordIndexResolver;

pub(super) fn compile_node(
    // Either a borrowed or owned expression.
    expr: impl std::borrow::Borrow<Expr>,
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    let expr: &Expr = std::borrow::Borrow::borrow(&expr);
    match expr {
        Expr::Alias(inner, _) => compile_node(
            inner.as_ref(),
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        Expr::KeepName(inner) => compile_node(
            inner.as_ref(),
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        Expr::RenameAlias { expr, .. } => compile_node(
            expr.as_ref(),
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        Expr::Cast { expr, .. } => compile_node(
            expr.as_ref(),
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        Expr::Sort { expr, .. } => compile_node(
            expr.as_ref(),
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        Expr::SortBy { expr, .. } => compile_node(
            expr.as_ref(),
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        Expr::Explode { input, .. } => compile_node(
            input.as_ref(),
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        Expr::Slice { input, .. } => compile_node(
            input.as_ref(),
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        // For window expressions, just compile the function expression only for now.
        // TODO: handle partition_by and order_by if needed.
        Expr::Over { function, .. } => compile_node(
            function.as_ref(),
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        // Expr::Rolling { function, .. } => compile_node(
        //     function,
        //     meta,
        //     dims,
        //     grid_shape,
        //     regular_chunk_shape,
        //     resolver,
        // ),
        // Expr::Window { function, .. } => compile_node(function, meta, dims, grid_shape, regular_chunk_shape, resolver),
        // If a filter expression is used where we expect a predicate, focus on the predicate.
        Expr::Filter { by, .. } => compile_node(
            by.as_ref(),
            meta,
            dims,
            grid_shape,
            regular_chunk_shape,
            resolver,
        ),
        Expr::BinaryExpr { left, op, right } => {
            match op {
                Operator::And | Operator::LogicalAnd => {
                    // If one side is unsupported, keep whatever constraints we can from the other.
                    let a = compile_node(
                        left.as_ref(),
                        meta,
                        dims,
                        grid_shape,
                        regular_chunk_shape,
                        resolver,
                    )?;
                    let b = compile_node(
                        right.as_ref(),
                        meta,
                        dims,
                        grid_shape,
                        regular_chunk_shape,
                        resolver,
                    )?;
                    Ok(and_nodes(a, b))
                }
                Operator::Or | Operator::LogicalOr => {
                    let a = compile_node(
                        left.as_ref(),
                        meta,
                        dims,
                        grid_shape,
                        regular_chunk_shape,
                        resolver,
                    )?;
                    let b = compile_node(
                        right.as_ref(),
                        meta,
                        dims,
                        grid_shape,
                        regular_chunk_shape,
                        resolver,
                    )?;
                    Ok(or_nodes(a, b))
                }
                Operator::Eq | Operator::GtEq | Operator::Gt | Operator::LtEq | Operator::Lt => {
                    if let Some((col, lit)) = col_lit(left, right).or_else(|| col_lit(right, left))
                    {
                        let op_eff = if matches!(left.as_ref(), Expr::Literal(_)) {
                            reverse_operator(*op)
                        } else {
                            *op
                        };
                        compile_cmp(
                            &col,
                            op_eff,
                            &lit,
                            meta,
                            dims,
                            grid_shape,
                            regular_chunk_shape,
                            resolver,
                        )
                    } else {
                        Err(CompileError::Unsupported(format!(
                            "unsupported comparison operator: {}",
                            op
                        )))
                    }
                }
                _ => Err(CompileError::Unsupported(format!(
                    "unsupported binary operator: {}",
                    op
                ))),
            }
        }
        Expr::Literal(lit) => {
            // Only boolean-ish literals can be predicates.
            match literal_anyvalue(lit) {
                Some(AnyValue::Boolean(true)) => Ok(ChunkPlanNode::AllChunks),
                Some(AnyValue::Boolean(false)) => Ok(ChunkPlanNode::Empty),
                // In Polars filtering, null predicate behaves like "keep nothing".
                Some(AnyValue::Null) => Ok(ChunkPlanNode::Empty),
                _ => Err(CompileError::Unsupported(format!(
                    "unsupported literal: {:?}",
                    lit
                ))),
            }
        }
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let predicate_node = compile_node(
                predicate.as_ref(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            if predicate_node.is_empty() {
                return Ok(ChunkPlanNode::Empty);
            }
            let truthy_node = compile_node(
                truthy.as_ref(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            if truthy_node.is_empty() {
                return Ok(ChunkPlanNode::Empty);
            }
            let falsy_node = compile_node(
                falsy.as_ref(),
                meta,
                dims,
                grid_shape,
                regular_chunk_shape,
                resolver,
            )?;
            if falsy_node.is_empty() {
                return Ok(ChunkPlanNode::Empty);
            }
            Ok(ChunkPlanNode::Union(vec![
                truthy_node,
                falsy_node,
                predicate_node,
            ]))
        }
        Expr::Function { input, function } => {
            match function {
                FunctionExpr::Boolean(bf) => compile_boolean_function(
                    bf,
                    input,
                    meta,
                    dims,
                    grid_shape,
                    regular_chunk_shape,
                    resolver,
                ),
                FunctionExpr::NullCount => Ok(ChunkPlanNode::AllChunks),
                // Most functions transform values in ways that we can't safely map to chunk-level constraints.
                _ => Err(CompileError::Unsupported(format!(
                    "unsupported function: {:?}",
                    function
                ))),
            }
        }
        Expr::Selector(selector) => {
            compile_selector(selector, meta, dims, grid_shape, regular_chunk_shape, resolver)
        }

        // Variants without a meaningful chunk-planning representation.
        Expr::Element
        | Expr::Column(_)
        | Expr::DataTypeFunction(_)
        | Expr::Gather { .. }
        | Expr::Agg(_)
        | Expr::Len
        | Expr::AnonymousFunction { .. }
        | Expr::Eval { .. }
        | Expr::SubPlan(_, _)
        | Expr::Field(_) => Err(CompileError::Unsupported(format!(
            "unsupported expression: {:?}",
            expr
        ))),
    }
}
