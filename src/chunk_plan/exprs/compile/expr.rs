//! Main expression compilation - compile_expr and match arms.

use super::super::compile_ctx::LazyCompileCtx;
use super::super::compile_node::{
    collect_column_refs,
    extract_struct_field_path,
};
use super::super::expr_plan::{ExprPlan, VarSet};
use super::super::expr_utils::try_expr_to_value_range_lazy;
use super::super::literals::{
    col_lit, literal_anyvalue, reverse_operator,
    strip_wrappers,
};
use super::boolean::compile_boolean_function_lazy;
use super::cmp::{
    compile_cmp_to_plan,
    compile_struct_field_cmp,
    compile_value_range_to_plan,
};
use super::interpolate::interpolate_selection_nd_lazy;
use super::selector::compile_selector_lazy;
use super::utils::{
    collect_refs_from_expr, refs_to_plan,
    refs_to_plan_with_vars,
};
use crate::chunk_plan::prelude::*;
use crate::errors::BackendError;
use crate::{IStr, IntoIStr};

type LazyResult = Result<ExprPlan, BackendError>;

/// Compile an expression to an `ExprPlan`.
///
/// This function traverses the expression tree and produces an `ExprPlan`
/// containing unresolved `ValueRange` constraints and variable references.
/// These are later converted to `LazyDatasetSelection`, batch-resolved,
/// and materialized into a concrete `DatasetSelection`.
pub(crate) fn compile_expr(
    expr: impl std::borrow::Borrow<Expr>,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    let expr: &Expr =
        std::borrow::Borrow::borrow(&expr);
    match expr {
        Expr::Display { .. } => {
            panic!(
                "Display expression not supported"
            );
        }
        Expr::Alias(inner, _) => {
            compile_expr(inner.as_ref(), ctx)
        }
        Expr::KeepName(inner) => {
            compile_expr(inner.as_ref(), ctx)
        }
        Expr::RenameAlias { expr, .. } => {
            compile_expr(expr.as_ref(), ctx)
        }
        Expr::Cast { expr, .. } => {
            compile_expr(expr.as_ref(), ctx)
        }
        Expr::Sort { expr, .. } => {
            compile_expr(expr.as_ref(), ctx)
        }
        Expr::SortBy { expr, .. } => {
            compile_expr(expr.as_ref(), ctx)
        }
        Expr::Explode { input, .. } => {
            compile_expr(input.as_ref(), ctx)
        }
        Expr::Slice { input, .. } => {
            compile_expr(input.as_ref(), ctx)
        }

        Expr::Over {
            function,
            partition_by,
            ..
        } => {
            let func_plan = compile_expr(
                function.as_ref(),
                ctx,
            )?;
            let mut refs = Vec::new();
            for p in partition_by {
                collect_column_refs(p, &mut refs);
            }
            refs.sort();
            refs.dedup();
            if refs.is_empty() {
                Ok(func_plan)
            } else {
                Ok(func_plan.add_vars(
                    VarSet::from_vec(refs),
                ))
            }
        }

        Expr::Rolling { function, .. } => {
            compile_expr(function.as_ref(), ctx)
        }

        Expr::Filter { input, by } => {
            let filter_plan =
                compile_expr(by.as_ref(), ctx)?;
            if filter_plan.is_empty() {
                return Ok(ExprPlan::Empty);
            }
            let input_plan = compile_expr(
                input.as_ref(),
                ctx,
            )?;
            let result = input_plan
                .intersect(&filter_plan);
            let filter_refs: Vec<IStr> =
                collect_refs_from_expr(
                    by.as_ref(),
                )
                .into_iter()
                .filter(|r| {
                    ctx.meta
                        .array_by_path(r.clone())
                        .is_some()
                })
                .collect();
            if filter_refs.is_empty() {
                Ok(result)
            } else {
                Ok(result.add_vars(
                    VarSet::from_vec(filter_refs),
                ))
            }
        }

        Expr::BinaryExpr { left, op, right } => {
            match op {
                Operator::And
                | Operator::LogicalAnd => {
                    // Special-case: A & !B => A \ B
                    // if let Expr::Function {
                    //     input,
                    //     function,
                    // } = strip_wrappers(
                    //     right.as_ref(),
                    // ) {
                    //     if matches!(function, FunctionExpr::Boolean(BooleanFunction::Not))
                    //         && input.len() == 1
                    //     {
                    //         let a = compile_expr(left.as_ref(), ctx)?;
                    //         let b = compile_expr(&input[0], ctx)?;
                    //         return Ok(a.difference(&b));
                    //     }
                    // }
                    // if let Expr::Function {
                    //     input,
                    //     function,
                    // } = strip_wrappers(
                    //     left.as_ref(),
                    // ) {
                    //     if matches!(function, FunctionExpr::Boolean(BooleanFunction::Not))
                    //         && input.len() == 1
                    //     {
                    //         let a = compile_expr(right.as_ref(), ctx)?;
                    //         let b = compile_expr(&input[0], ctx)?;
                    //         return Ok(a.difference(&b));
                    //     }
                    // }

                    // // Fast path: merge compatible comparisons on the same column
                    // if let (Some((col_a, vr_a)), Some((col_b, vr_b))) = (
                    //     try_expr_to_value_range_lazy(left.as_ref()),
                    //     try_expr_to_value_range_lazy(right.as_ref()),
                    // ) {
                    //     if col_a == col_b {
                    //         let Some(vr) = vr_a.intersect(&vr_b) else {
                    //             return Ok(ExprPlan::Empty);
                    //         };
                    //         return compile_value_range_to_plan(&col_a, &vr, ctx);
                    //     }
                    // }

                    let a = compile_expr(
                        left.as_ref(),
                        ctx,
                    )?;
                    let b = compile_expr(
                        right.as_ref(),
                        ctx,
                    )?;
                    Ok(a.intersect(&b))
                }
                Operator::Or
                | Operator::LogicalOr => {
                    let a = compile_expr(
                        left.as_ref(),
                        ctx,
                    )?;
                    let b = compile_expr(
                        right.as_ref(),
                        ctx,
                    )?;
                    Ok(a.union(&b))
                }
                Operator::Xor => {
                    let a = compile_expr(
                        left.as_ref(),
                        ctx,
                    )?;
                    let b = compile_expr(
                        right.as_ref(),
                        ctx,
                    )?;
                    Ok(a.exclusive_or(&b))
                }
                Operator::Eq
                | Operator::GtEq
                | Operator::Gt
                | Operator::LtEq
                | Operator::Lt => {
                    // Check for struct field access
                    if let Some((
                        struct_col,
                        field_name,
                    )) =
                        extract_struct_field_path(
                            strip_wrappers(
                                left.as_ref(),
                            ),
                        )
                    {
                        if let Expr::Literal(
                            lit,
                        ) = strip_wrappers(
                            right.as_ref(),
                        ) {
                            return compile_struct_field_cmp(
                                &struct_col,
                                &field_name,
                                *op,
                                lit,
                                ctx,
                            )
                            .or_else(|_| Ok(ExprPlan::NoConstraint));
                        }
                    }
                    if let Some((
                        struct_col,
                        field_name,
                    )) =
                        extract_struct_field_path(
                            strip_wrappers(
                                right.as_ref(),
                            ),
                        )
                    {
                        if let Expr::Literal(
                            lit,
                        ) = strip_wrappers(
                            left.as_ref(),
                        ) {
                            return compile_struct_field_cmp(
                                &struct_col,
                                &field_name,
                                reverse_operator(*op),
                                lit,
                                ctx,
                            )
                            .or_else(|_| Ok(ExprPlan::NoConstraint));
                        }
                    }

                    // Regular column comparison
                    if let Some((col, lit)) =
                        col_lit(left, right)
                            .or_else(|| {
                                col_lit(
                                    right, left,
                                )
                            })
                    {
                        let op_eff = if matches!(
                            strip_wrappers(
                                left.as_ref()
                            ),
                            Expr::Literal(_)
                        ) {
                            reverse_operator(*op)
                        } else {
                            *op
                        };
                        compile_cmp_to_plan(&col, op_eff, &lit, ctx)
                            .or_else(|_| {
                                Ok(ExprPlan::unconstrained_vars(VarSet::single(col.istr())))
                            })
                    } else {
                        Ok(refs_to_plan(collect_refs_from_expr(expr)))
                    }
                }
                _ => Ok(refs_to_plan(
                    collect_refs_from_expr(expr),
                )),
            }
        }

        Expr::Literal(lit) => {
            match literal_anyvalue(lit) {
                Some(AnyValue::Boolean(true)) => {
                    Ok(ExprPlan::NoConstraint)
                }
                Some(AnyValue::Boolean(
                    false,
                )) => Ok(ExprPlan::Empty),
                Some(AnyValue::Null) => {
                    Ok(ExprPlan::Empty)
                }
                _ => Ok(ExprPlan::NoConstraint),
            }
        }

        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            if let (
                Expr::Literal(t),
                Expr::Literal(f),
            ) = (
                strip_wrappers(truthy.as_ref()),
                strip_wrappers(falsy.as_ref()),
            ) {
                if let (
                    Some(AnyValue::Boolean(t)),
                    Some(AnyValue::Boolean(f)),
                ) = (
                    literal_anyvalue(t),
                    literal_anyvalue(f),
                ) {
                    if t && !f {
                        return compile_expr(
                            predicate.as_ref(),
                            ctx,
                        );
                    }
                    if f {
                        return Ok(ExprPlan::NoConstraint);
                    }
                    return Ok(ExprPlan::Empty);
                }
            }

            let all_refs =
                collect_refs_from_expr(expr);
            let vars = VarSet::from_vec(all_refs);

            let is_falsy_null = matches!(
                strip_wrappers(falsy.as_ref()),
                Expr::Literal(lit) if matches!(literal_anyvalue(lit), Some(AnyValue::Null))
            );

            if is_falsy_null {
                let predicate_plan =
                    compile_expr(
                        predicate.as_ref(),
                        ctx,
                    )?;
                if predicate_plan.is_empty() {
                    return Ok(ExprPlan::Empty);
                }
                Ok(predicate_plan.with_vars(vars))
            } else {
                Ok(refs_to_plan_with_vars(vars))
            }
        }

        Expr::Function { input, function } => {
            match function {
                FunctionExpr::Boolean(bf) => {
                    compile_boolean_function_lazy(
                        bf, input, ctx,
                    )
                }
                FunctionExpr::NullCount => {
                    Ok(ExprPlan::NoConstraint)
                }
                FunctionExpr::FfiPlugin {
                    symbol,
                    ..
                } => {
                    if symbol == "interpolate_nd"
                    {
                        if input.len() < 3 {
                            return Ok(ExprPlan::NoConstraint);
                        }
                        interpolate_selection_nd_lazy(&input[0], &input[1], &input[2], ctx)
                    } else {
                        Ok(refs_to_plan(collect_refs_from_expr(expr)))
                    }
                }
                _ => {
                    let mut refs = Vec::new();
                    for i in input {
                        collect_column_refs(
                            i, &mut refs,
                        );
                    }
                    refs.sort();
                    refs.dedup();
                    Ok(refs_to_plan(refs))
                }
            }
        }

        Expr::Selector(selector) => {
            compile_selector_lazy(selector, ctx)
        }

        Expr::Column(name) => {
            Ok(ExprPlan::unconstrained_vars(
                VarSet::single(name.istr()),
            ))
        }

        Expr::Agg(agg) => {
            let inner: &Expr = agg.as_ref();
            compile_expr(inner, ctx)
        }

        Expr::AnonymousFunction {
            input, ..
        } => {
            let mut refs = Vec::new();
            for i in input {
                collect_column_refs(i, &mut refs);
            }
            refs.sort();
            refs.dedup();
            Ok(refs_to_plan(refs))
        }

        Expr::Gather { expr, idx, .. } => {
            let mut refs = Vec::new();
            collect_column_refs(expr, &mut refs);
            collect_column_refs(idx, &mut refs);
            refs.sort();
            refs.dedup();
            Ok(refs_to_plan(refs))
        }

        Expr::Eval { expr, .. } => {
            compile_expr(expr.as_ref(), ctx)
        }

        Expr::Field(names) => {
            let vars: Vec<IStr> = names
                .iter()
                .map(|n| n.istr())
                .collect();
            if vars.is_empty() {
                Ok(ExprPlan::NoConstraint)
            } else {
                Ok(ExprPlan::unconstrained_vars(
                    VarSet::from_vec(vars),
                ))
            }
        }

        Expr::Element
        | Expr::Len
        | Expr::SubPlan(_, _)
        | Expr::DataTypeFunction(_) => {
            Ok(ExprPlan::NoConstraint)
        }

        Expr::StructEval { .. } => {
            Ok(ExprPlan::NoConstraint)
        }
    }
}

// Helper for compiling a list of (presumably adjacent) columnar expressions
pub(super) fn compile_expr_list(
    exprs: &[Expr],
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    let mut plans = Vec::new();
    for expr in exprs {
        plans.push(compile_expr(expr, ctx)?);
    }
    Ok(plans
        .into_iter()
        .reduce(|a, b| a.intersect(&b))
        .unwrap_or(ExprPlan::NoConstraint))
}
