use super::compile_boolean::compile_boolean_function;
use super::compile_cmp::{
    compile_cmp_to_dataset_selection, compile_value_range_to_dataset_selection, try_expr_to_value_range,
};
use super::errors::CompileError;
use super::literals::{col_lit, literal_anyvalue, reverse_operator};
use super::prelude::*;
use super::selection::DatasetSelection;
use super::selector::compile_selector;
use super::errors::CoordIndexResolver;

pub(super) fn compile_node(
    // Either a borrowed or owned expression.
    expr: impl std::borrow::Borrow<Expr>,
    meta: &ZarrDatasetMeta,
    dims: &[String],
    dim_lengths: &[u64],
    vars: &[String],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<DatasetSelection, CompileError> {
    let expr: &Expr = std::borrow::Borrow::borrow(&expr);
    match expr {
        Expr::Alias(inner, _) => compile_node(
            inner.as_ref(),
            meta,
            dims,
            dim_lengths,
            vars,
            resolver,
        ),
        Expr::KeepName(inner) => compile_node(
            inner.as_ref(),
            meta,
            dims,
            dim_lengths,
            vars,
            resolver,
        ),
        Expr::RenameAlias { expr, .. } => compile_node(
            expr.as_ref(),
            meta,
            dims,
            dim_lengths,
            vars,
            resolver,
        ),
        Expr::Cast { expr, .. } => compile_node(
            expr.as_ref(),
            meta,
            dims,
            dim_lengths,
            vars,
            resolver,
        ),
        Expr::Sort { expr, .. } => compile_node(
            expr.as_ref(),
            meta,
            dims,
            dim_lengths,
            vars,
            resolver,
        ),
        Expr::SortBy { expr, .. } => compile_node(
            expr.as_ref(),
            meta,
            dims,
            dim_lengths,
            vars,
            resolver,
        ),
        Expr::Explode { input, .. } => compile_node(
            input.as_ref(),
            meta,
            dims,
            dim_lengths,
            vars,
            resolver,
        ),
        Expr::Slice { input, .. } => compile_node(
            input.as_ref(),
            meta,
            dims,
            dim_lengths,
            vars,
            resolver,
        ),
        // For window expressions, just compile the function expression only for now.
        // TODO: handle partition_by and order_by if needed.
        Expr::Over { function, .. } => compile_node(
            function.as_ref(),
            meta,
            dims,
            dim_lengths,
            vars,
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
            dim_lengths,
            vars,
            resolver,
        ),
        Expr::BinaryExpr { left, op, right } => {
            match op {
                Operator::And | Operator::LogicalAnd => {
                    // Fast path: merge compatible comparisons on the same column into a single
                    // ValueRange. This reduces resolver reads and enables tighter planning.
                    if let (Some((col_a, vr_a)), Some((col_b, vr_b))) = (
                        try_expr_to_value_range(left.as_ref(), meta),
                        try_expr_to_value_range(right.as_ref(), meta),
                    ) {
                        if col_a == col_b {
                            let vr = vr_a.intersect(&vr_b);
                            return compile_value_range_to_dataset_selection(
                                &col_a, &vr, meta, dims, dim_lengths, vars, resolver,
                            );
                        }
                    }

                    // If one side is unsupported, keep whatever constraints we can from the other.
                    let a = compile_node(
                        left.as_ref(),
                        meta,
                        dims,
                        dim_lengths,
                        vars,
                        resolver,
                    )?;
                    let b = compile_node(
                        right.as_ref(),
                        meta,
                        dims,
                        dim_lengths,
                        vars,
                        resolver,
                    )?;
                    Ok(a.intersect(&b))
                }
                Operator::Or | Operator::LogicalOr => {
                    let a = compile_node(
                        left.as_ref(),
                        meta,
                        dims,
                        dim_lengths,
                        vars,
                        resolver,
                    )?;
                    let b = compile_node(
                        right.as_ref(),
                        meta,
                        dims,
                        dim_lengths,
                        vars,
                        resolver,
                    )?;
                    Ok(a.union(&b))
                }
                Operator::Eq | Operator::GtEq | Operator::Gt | Operator::LtEq | Operator::Lt => {
                    if let Some((col, lit)) = col_lit(left, right).or_else(|| col_lit(right, left))
                    {
                        let op_eff = if matches!(left.as_ref(), Expr::Literal(_)) {
                            reverse_operator(*op)
                        } else {
                            *op
                        };
                        compile_cmp_to_dataset_selection(
                            &col, op_eff, &lit, meta, dims, dim_lengths, vars, resolver,
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
                Some(AnyValue::Boolean(true)) => Ok(DatasetSelection::all_for_vars(vars.to_vec())),
                Some(AnyValue::Boolean(false)) => Ok(DatasetSelection::empty()),
                // In Polars filtering, null predicate behaves like "keep nothing".
                Some(AnyValue::Null) => Ok(DatasetSelection::empty()),
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
                dim_lengths,
                vars,
                resolver,
            )?;
            if predicate_node.0.is_empty() {
                return Ok(DatasetSelection::empty());
            }
            let truthy_node = compile_node(
                truthy.as_ref(),
                meta,
                dims,
                dim_lengths,
                vars,
                resolver,
            )?;
            if truthy_node.0.is_empty() {
                return Ok(DatasetSelection::empty());
            }
            let falsy_node = compile_node(
                falsy.as_ref(),
                meta,
                dims,
                dim_lengths,
                vars,
                resolver,
            )?;
            if falsy_node.0.is_empty() {
                return Ok(DatasetSelection::empty());
            }
            Ok(truthy_node.union(&falsy_node).union(&predicate_node))
        }
        Expr::Function { input, function } => {
            match function {
                FunctionExpr::Boolean(bf) => compile_boolean_function(
                    bf,
                    input,
                    meta,
                    dims,
                    dim_lengths,
                    vars,
                    resolver,
                ),
                FunctionExpr::NullCount => Ok(DatasetSelection::all_for_vars(vars.to_vec())),
                // Most functions transform values in ways that we can't safely map to chunk-level constraints.
                _ => Err(CompileError::Unsupported(format!(
                    "unsupported function: {:?}",
                    function
                ))),
            }
        }
        Expr::Selector(selector) => {
            compile_selector(selector, meta, dims, dim_lengths, vars, resolver)
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
