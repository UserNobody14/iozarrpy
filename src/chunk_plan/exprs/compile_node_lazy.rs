//! Lazy expression compilation - produces ExprPlan without resolving.
//!
//! This module compiles Polars expressions into `ExprPlan`, which separates
//! dimension constraints from variable tracking. The expensive `GroupedSelection`
//! construction is deferred to `ExprPlan::into_lazy_dataset_selection`.

use snafu::{ResultExt, ensure};

use super::compile_ctx::LazyCompileCtx;
use super::compile_node::{
    collect_column_refs, extract_struct_field_path,
};
use super::expr_plan::{ExprPlan, VarSet};
use super::expr_utils::{
    extract_column_names_lazy,
    try_expr_to_value_range_lazy,
    extract_literal_struct_series_lazy,
    series_values_scalar_lazy,
};
use super::literals::{
    col_lit, literal_anyvalue, literal_to_scalar,
    reverse_operator, strip_wrappers,
};
use crate::chunk_plan::indexing::lazy_selection::{
    LazyArraySelection, LazyDimConstraint,
    LazyHyperRectangle,
};
use crate::chunk_plan::indexing::selection::SetOperations;
use crate::chunk_plan::indexing::types::ValueRangePresent;
use crate::chunk_plan::prelude::*;
use crate::{IStr, IntoIStr};
use crate::errors::BackendError;

use super::expr_utils::{
    expr_to_col_name,
    extract_var_from_source_value_expr,
};

type LazyResult = Result<ExprPlan, BackendError>;

fn refs_to_plan(refs: Vec<IStr>) -> ExprPlan {
    if refs.is_empty() {
        ExprPlan::NoConstraint
    } else {
        ExprPlan::unconstrained_vars(
            VarSet::from_vec(refs),
        )
    }
}

fn refs_to_plan_with_vars(
    vars: VarSet,
) -> ExprPlan {
    if vars.is_empty() {
        ExprPlan::NoConstraint
    } else {
        ExprPlan::unconstrained_vars(vars)
    }
}

fn collect_refs_from_expr(
    expr: &Expr,
) -> Vec<IStr> {
    let mut refs = Vec::new();
    collect_column_refs(expr, &mut refs);
    refs.sort();
    refs.dedup();
    refs
}

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
                    if let Expr::Function {
                        input,
                        function,
                    } = strip_wrappers(
                        right.as_ref(),
                    ) {
                        if matches!(function, FunctionExpr::Boolean(BooleanFunction::Not))
                            && input.len() == 1
                        {
                            let a = compile_expr(left.as_ref(), ctx)?;
                            let b = compile_expr(&input[0], ctx)?;
                            return Ok(a.difference(&b));
                        }
                    }
                    if let Expr::Function {
                        input,
                        function,
                    } = strip_wrappers(
                        left.as_ref(),
                    ) {
                        if matches!(function, FunctionExpr::Boolean(BooleanFunction::Not))
                            && input.len() == 1
                        {
                            let a = compile_expr(right.as_ref(), ctx)?;
                            let b = compile_expr(&input[0], ctx)?;
                            return Ok(a.difference(&b));
                        }
                    }

                    // Fast path: merge compatible comparisons on the same column
                    if let (Some((col_a, vr_a)), Some((col_b, vr_b))) = (
                        try_expr_to_value_range_lazy(left.as_ref(), ctx),
                        try_expr_to_value_range_lazy(right.as_ref(), ctx),
                    ) {
                        if col_a == col_b {
                            let Some(vr) = vr_a.intersect(&vr_b) else {
                                return Ok(ExprPlan::Empty);
                            };
                            return compile_value_range_to_plan(&col_a, &vr, ctx);
                        }
                    }

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
                                &struct_col, &field_name, *op, lit, ctx,
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
                                &struct_col, &field_name, reverse_operator(*op), lit, ctx,
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
                            .or_else(|_| Ok(ExprPlan::unconstrained_vars(VarSet::single(col.istr()))))
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

/// Compile a comparison to an ExprPlan with dimension constraint.
fn compile_cmp_to_plan(
    col: &IStr,
    op: Operator,
    lit: &LiteralValue,
    ctx: &LazyCompileCtx<'_>,
) -> LazyResult {
    let time_encoding = ctx
        .meta
        .array_by_path(col.clone())
        .and_then(|a| a.encoding.as_ref())
        .and_then(|e| e.as_time_encoding());
    let scalar =
        literal_to_scalar(lit, time_encoding)?;
    let vr = ValueRangePresent::from_polars_op(
        op, scalar,
    )?;
    compile_value_range_to_plan(col, &vr, ctx)
}

/// Compile a value range to an ExprPlan with dimension constraint.
fn compile_value_range_to_plan(
    col: &str,
    vr: &ValueRangePresent,
    ctx: &LazyCompileCtx<'_>,
) -> LazyResult {
    if ctx.dim_index(col).is_none() {
        return Ok(ExprPlan::NoConstraint);
    }

    let constraint =
        LazyDimConstraint::Unresolved(vr.clone());
    let rect = LazyHyperRectangle::all()
        .with_dim(col.istr(), constraint);
    let sel =
        LazyArraySelection::from_rectangle(rect);

    Ok(ExprPlan::constrained(VarSet::All, sel))
}

/// Compile a struct field comparison to an ExprPlan.
fn compile_struct_field_cmp(
    struct_col: &IStr,
    field_name: &IStr,
    op: Operator,
    lit: &LiteralValue,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    let array_path: IStr =
        format!("{}/{}", struct_col, field_name)
            .istr();

    let arr_meta_opt =
        ctx.meta.array_by_path(&array_path);
    if arr_meta_opt.is_none() {
        return Err(
            BackendError::StructFieldNotFound {
                path: array_path.clone(),
            },
        );
    }

    let time_encoding = arr_meta_opt
        .and_then(|a| a.encoding.as_ref())
        .and_then(|e| e.as_time_encoding());
    let scalar =
        literal_to_scalar(lit, time_encoding)?;
    let vr = ValueRangePresent::from_polars_op(
        op, scalar,
    )?;

    if let Some(arr_meta) = arr_meta_opt {
        if arr_meta.dims.len() == 1 {
            let dim = &arr_meta.dims[0];
            if ctx.dims.contains(dim) {
                let constraint =
                    LazyDimConstraint::Unresolved(
                        vr.clone(),
                    );
                let rect =
                    LazyHyperRectangle::all()
                        .with_dim(
                            dim.clone(),
                            constraint,
                        );
                let sel = LazyArraySelection::from_rectangle(rect);
                return Ok(
                    ExprPlan::constrained(
                        VarSet::All,
                        sel,
                    ),
                );
            }
        }
    }

    Ok(ExprPlan::unconstrained_vars(
        VarSet::single(array_path),
    ))
}

/// Compile a boolean function to an ExprPlan.
fn compile_boolean_function_lazy(
    bf: &BooleanFunction,
    input: &[Expr],
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    match bf {
        BooleanFunction::Not => {
            let [arg] = input else {
                return Err(BackendError::UnsupportedBooleanFunction {
                    function: bf.clone(),
                });
            };
            if let Expr::Literal(lit) =
                strip_wrappers(arg)
            {
                return match literal_anyvalue(lit)
                {
                    Some(AnyValue::Boolean(
                        true,
                    )) => Ok(ExprPlan::Empty),
                    Some(AnyValue::Boolean(
                        false,
                    )) => {
                        Ok(ExprPlan::NoConstraint)
                    }
                    Some(AnyValue::Null) => {
                        Ok(ExprPlan::Empty)
                    }
                    _ => {
                        Ok(ExprPlan::NoConstraint)
                    }
                };
            }
            let inner = compile_expr(arg, ctx)
                .unwrap_or(
                    ExprPlan::NoConstraint,
                );
            if inner.is_empty() {
                Ok(ExprPlan::NoConstraint)
            } else {
                Ok(ExprPlan::NoConstraint)
            }
        }
        BooleanFunction::IsNull
        | BooleanFunction::IsNotNull => {
            let [arg] = input else {
                return Err(BackendError::UnsupportedBooleanFunction {
                    function: bf.clone(),
                });
            };
            if let Expr::Literal(lit) =
                strip_wrappers(arg)
            {
                let is_null = matches!(
                    literal_anyvalue(lit),
                    Some(AnyValue::Null)
                );
                let keep = match bf {
                    BooleanFunction::IsNull => is_null,
                    BooleanFunction::IsNotNull => !is_null,
                    _ => unreachable!(),
                };
                return Ok(if keep {
                    ExprPlan::NoConstraint
                } else {
                    ExprPlan::Empty
                });
            }
            Ok(ExprPlan::NoConstraint)
        }
        BooleanFunction::IsBetween { .. } => {
            compile_is_between_lazy(input, ctx)
        }
        BooleanFunction::IsIn { .. } => {
            compile_is_in_lazy(input, ctx)
        }
        BooleanFunction::AnyHorizontal => {
            let mut acc = ExprPlan::Empty;
            for e in input {
                let plan = compile_expr(e, ctx)
                    .unwrap_or(
                        ExprPlan::NoConstraint,
                    );
                acc = acc.union(&plan);
            }
            Ok(acc)
        }
        BooleanFunction::AllHorizontal => {
            let mut acc = ExprPlan::NoConstraint;
            for e in input {
                let plan = compile_expr(e, ctx)
                    .unwrap_or(
                        ExprPlan::NoConstraint,
                    );
                acc = acc.intersect(&plan);
                if acc.is_empty() {
                    break;
                }
            }
            Ok(acc)
        }
        _ => Ok(ExprPlan::NoConstraint),
    }
}

/// Compile is_between to an ExprPlan.
fn compile_is_between_lazy(
    input: &[Expr],
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    if input.len() < 3 {
        return Err(
            BackendError::compile_polars(
                format!(
                    "unsupported is_between expression: {:?}",
                    input
                ),
            ),
        );
    }
    let expr = &input[0];
    let low = &input[1];
    let high = &input[2];

    let Some(col) = expr_to_col_name(expr) else {
        return Ok(ExprPlan::NoConstraint);
    };
    let Expr::Literal(low_lit) =
        strip_wrappers(low)
    else {
        return Ok(ExprPlan::NoConstraint);
    };
    let Expr::Literal(high_lit) =
        strip_wrappers(high)
    else {
        return Ok(ExprPlan::NoConstraint);
    };

    let a = compile_cmp_to_plan(
        &col,
        Operator::GtEq,
        low_lit,
        ctx,
    )
    .unwrap_or(ExprPlan::NoConstraint);
    let b = compile_cmp_to_plan(
        &col,
        Operator::LtEq,
        high_lit,
        ctx,
    )
    .unwrap_or(ExprPlan::NoConstraint);
    Ok(a.intersect(&b))
}

/// Compile is_in to an ExprPlan.
fn compile_is_in_lazy(
    input: &[Expr],
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    use polars::prelude::Scalar;

    if input.len() < 2 {
        return Err(
            BackendError::compile_polars(
                format!(
                    "unsupported is_in expression: {:?}",
                    input
                ),
            ),
        );
    }
    let expr = &input[0];
    let list = &input[1];

    let Some(col) = expr_to_col_name(expr) else {
        return Ok(ExprPlan::NoConstraint);
    };
    let Expr::Literal(list_lit) =
        strip_wrappers(list)
    else {
        return Ok(ExprPlan::NoConstraint);
    };

    let (dtype, values): (
        &polars::prelude::DataType,
        Vec<AnyValue<'static>>,
    ) = match list_lit {
        LiteralValue::Series(s) => {
            let series = &**s;
            if series.len() > 4096 {
                return Ok(
                    ExprPlan::NoConstraint,
                );
            }
            (
                series.dtype(),
                series
                    .iter()
                    .map(|av| av.into_static())
                    .collect(),
            )
        }
        LiteralValue::Scalar(s) => {
            let ssv = s.clone();
            let av = ssv.into_value();
            match av {
                AnyValue::List(series) => {
                    if series.len() > 4096 {
                        return Err(BackendError::compile_polars(format!(
                            "list literal is too long: {:?}",
                            series
                        )));
                    }
                    (
                        &series.dtype().clone(),
                        series
                            .iter()
                            .map(|av| {
                                av.into_static()
                            })
                            .collect(),
                    )
                }
                AnyValue::Array(series, _) => {
                    if series.len() > 4096 {
                        return Err(BackendError::compile_polars(format!(
                            "array literal is too long: {:?}",
                            series
                        )));
                    }
                    (
                        &series.dtype().clone(),
                        series
                            .iter()
                            .map(|av| {
                                av.into_static()
                            })
                            .collect(),
                    )
                }
                _ => {
                    return Ok(
                        ExprPlan::NoConstraint,
                    );
                }
            }
        }
        _ => return Ok(ExprPlan::NoConstraint),
    };

    let mut out: Option<ExprPlan> = None;
    for av in values {
        if matches!(av, AnyValue::Null) {
            return Ok(ExprPlan::NoConstraint);
        }
        let lit = LiteralValue::Scalar(
            Scalar::new(dtype.clone(), av),
        );
        let node = compile_cmp_to_plan(
            &col,
            Operator::Eq,
            &lit,
            ctx,
        )
        .unwrap_or(ExprPlan::NoConstraint);

        if node == ExprPlan::NoConstraint {
            return Ok(ExprPlan::NoConstraint);
        }
        out = Some(match out.take() {
            None => node,
            Some(acc) => acc.union(&node),
        });
    }
    Ok(out.unwrap_or(ExprPlan::Empty))
}

/// Compile selector to an ExprPlan.
fn compile_selector_lazy(
    selector: &Selector,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    use regex::Regex;

    match selector {
        Selector::Union(left, right) => {
            let l = compile_selector_lazy(
                left.as_ref(),
                ctx,
            )?;
            let r = compile_selector_lazy(
                right.as_ref(),
                ctx,
            )?;
            Ok(l.union(&r))
        }
        Selector::Difference(left, right) => {
            let l = compile_selector_lazy(
                left.as_ref(),
                ctx,
            )?;
            let r = compile_selector_lazy(
                right.as_ref(),
                ctx,
            )?;
            Ok(l.difference(&r))
        }
        Selector::ExclusiveOr(left, right) => {
            let l = compile_selector_lazy(
                left.as_ref(),
                ctx,
            )?;
            let r = compile_selector_lazy(
                right.as_ref(),
                ctx,
            )?;
            Ok(l.exclusive_or(&r))
        }
        Selector::Intersect(left, right) => {
            let l = compile_selector_lazy(
                left.as_ref(),
                ctx,
            )?;
            let r = compile_selector_lazy(
                right.as_ref(),
                ctx,
            )?;
            Ok(l.intersect(&r))
        }
        Selector::Empty => Ok(ExprPlan::Empty),
        Selector::ByName { names, .. } => {
            let vars: Vec<IStr> = names
                .iter()
                .map(|s| s.istr())
                .collect();
            Ok(ExprPlan::unconstrained_vars(
                VarSet::from_vec(vars),
            ))
        }
        Selector::Matches(pattern) => {
            let re = Regex::new(pattern.as_str()).context(
                crate::errors::backend::RegexSnafu {
                    pattern: pattern.clone(),
                },
            )?;
            let matching_vars: Vec<IStr> = ctx
                .meta
                .all_array_paths()
                .iter()
                .filter(|v| {
                    re.is_match(v.as_ref())
                })
                .cloned()
                .collect();
            if matching_vars.is_empty() {
                Ok(ExprPlan::Empty)
            } else {
                Ok(ExprPlan::unconstrained_vars(
                    VarSet::from_vec(
                        matching_vars,
                    ),
                ))
            }
        }
        Selector::ByDType(dtype_selector) => {
            let matching_vars: Vec<IStr> = ctx
                .meta
                .all_array_paths()
                .iter()
                .filter(|v| {
                    if let Some(array_meta) = ctx
                        .meta
                        .array_by_path(v.istr())
                    {
                        dtype_selector.matches(
                            &array_meta
                                .polars_dtype,
                        )
                    } else {
                        true
                    }
                })
                .cloned()
                .collect();
            if matching_vars.is_empty() {
                Ok(ExprPlan::Empty)
            } else {
                Ok(ExprPlan::unconstrained_vars(
                    VarSet::from_vec(
                        matching_vars,
                    ),
                ))
            }
        }
        Selector::ByIndex { .. } => {
            Ok(ExprPlan::NoConstraint)
        }
        Selector::Wildcard => {
            let all_vars = ctx
                .meta
                .all_array_paths()
                .to_vec();
            Ok(ExprPlan::unconstrained_vars(
                VarSet::from_vec(all_vars),
            ))
        }
    }
}

/// Compile interpolation selection (lazy version).
fn interpolate_selection_nd_lazy(
    source_coords: &Expr,
    source_values: &Expr,
    target_values: &Expr,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    use crate::chunk_plan::indexing::types::CoordScalar;

    let coord_names =
        extract_column_names_lazy(source_coords);
    if coord_names.is_empty() {
        return Ok(ExprPlan::NoConstraint);
    }

    let Some(target_struct) =
        extract_literal_struct_series_lazy(
            target_values,
        )
    else {
        return Ok(ExprPlan::NoConstraint);
    };

    let Ok(target_sc) = target_struct.struct_()
    else {
        return Ok(ExprPlan::NoConstraint);
    };
    let target_fields =
        target_sc.fields_as_series();

    let mut dim_values: std::collections::BTreeMap<IStr, Vec<CoordScalar>> =
        std::collections::BTreeMap::new();

    // Include all target dimensions that match ctx.dims, not just coord_names.
    // Extra dims (e.g. time in target when interpolating lat/lon) must be
    // constrained to exact match so we load only the relevant slice.
    for s in target_fields.iter() {
        let name = s.name().as_str().istr();
        if !ctx.dims.iter().any(|d| d == &name) {
            continue;
        }

        let Some(values) =
            series_values_scalar_lazy(s)
        else {
            return Ok(ExprPlan::NoConstraint);
        };

        if !values.is_empty() {
            dim_values.insert(name, values);
        }
    }

    if dim_values.is_empty() {
        return Ok(ExprPlan::NoConstraint);
    }

    let mut constraints: Vec<
        std::collections::BTreeMap<
            IStr,
            LazyDimConstraint,
        >,
    > = vec![];

    // Transform dim_values (dim -> Vec<CoordScalar>) to row-wise constraints.
    // Use InterpolationRange (with expansion) only for coord_names; use Unresolved
    // (no expansion) for filter dimensions so we load only the exact slice.
    let num_rows = dim_values
        .values()
        .next()
        .map(|v| v.len())
        .unwrap_or(0);
    for i in 0..num_rows {
        let mut constraint =
            std::collections::BTreeMap::new();
        for (dim_name, values) in
            dim_values.iter()
        {
            let value = values[i].clone();
            let vr = ValueRangePresent::from_equal_case(value);
            let is_interp_dim = coord_names
                .iter()
                .any(|c| c == dim_name);
            let c = if is_interp_dim {
                LazyDimConstraint::InterpolationRange(vr)
            } else {
                LazyDimConstraint::Unresolved(vr)
            };
            constraint
                .insert(dim_name.clone(), c);
        }
        constraints.push(constraint);
    }

    let rects: Vec<LazyHyperRectangle> =
        constraints
            .into_iter()
            .map(|c| {
                LazyHyperRectangle::with_dims(c)
            })
            .collect();
    let sel = LazyArraySelection::Rectangles(
        rects.into(),
    );

    let (retrieve_vars, filter_plan) =
        match source_values {
            Expr::Function {
                input,
                function,
            } => match function {
                FunctionExpr::AsStruct => {
                    let mut vars =
                        Vec::with_capacity(
                            input.len(),
                        );
                    let mut filter_plan_acc: Option<ExprPlan> = None;
                    for n in input {
                        let Some((name, filter_pred)) =
                        extract_var_from_source_value_expr(n)
                    else {
                        return Err(BackendError::compile_polars(format!(
                            "source_values must be an Expr::Function with FunctionExpr::AsStruct \
                             containing column refs or col(...).filter(predicate): {:?}",
                            source_values
                        )));
                    };
                        vars.push(name);
                        if let Some(pred) =
                            filter_pred
                        {
                            let fp =
                                compile_expr(
                                    pred, ctx,
                                )?;
                            if !fp.is_empty() {
                                filter_plan_acc = Some(match filter_plan_acc.take() {
                                None => fp,
                                Some(acc) => acc.intersect(&fp),
                            });
                            }
                        }
                    }
                    (vars, filter_plan_acc)
                }
                _ => {
                    return Err(BackendError::compile_polars(format!(
                    "source_values must be an Expr::Function with FunctionExpr::AsStruct \
                     containing column refs or col(...).filter(predicate): {:?}",
                    source_values
                )));
                }
            },
            Expr::Field(names) => (
                names
                    .iter()
                    .map(|n| n.istr())
                    .collect::<Vec<_>>(),
                None,
            ),
            _ => {
                return Err(
                    BackendError::compile_polars(
                        format!(
                            "source_values must be an Expr::Field or AsStruct containing variable names: {:?}",
                            source_values
                        ),
                    ),
                );
            }
        };

    let sel = match filter_plan {
        Some(ExprPlan::Active {
            constraints: filter_sel,
            ..
        }) => sel.intersect(filter_sel.as_ref()),
        Some(ExprPlan::Empty) => {
            return Ok(ExprPlan::Empty);
        }
        Some(ExprPlan::NoConstraint) | None => {
            sel
        }
    };

    if sel.is_empty() {
        return Ok(ExprPlan::Empty);
    }

    Ok(ExprPlan::constrained(
        VarSet::from_vec(retrieve_vars),
        sel,
    ))
}
