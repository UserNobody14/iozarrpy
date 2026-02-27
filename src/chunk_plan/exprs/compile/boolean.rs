//! Boolean function compilation (Not, IsNull, IsBetween, IsIn, AnyHorizontal, AllHorizontal).

use super::super::compile_ctx::LazyCompileCtx;
use super::super::expr_plan::ExprPlan;
use super::super::expr_utils::expr_to_col_name;
use super::super::literals::{
    literal_anyvalue, strip_wrappers,
};
use super::cmp::compile_cmp_to_plan;
use crate::chunk_plan::prelude::*;
use crate::errors::BackendError;
use crate::try_extract;

use super::expr::compile_expr;

type LazyResult = Result<ExprPlan, BackendError>;

/// Compile a boolean function to an ExprPlan.
pub(super) fn compile_boolean_function_lazy(
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
                Ok(inner.boolean_not())
            }
        }
        BooleanFunction::IsNull
        | BooleanFunction::IsNotNan
        | BooleanFunction::IsNan
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
                    BooleanFunction::IsNan => is_null,
                    BooleanFunction::IsNotNan => !is_null,
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
        BooleanFunction::IsFinite => {
            todo!()
        }
        BooleanFunction::Any { ignore_nulls } => {
            todo!()
        }
        BooleanFunction::All { ignore_nulls } => {
            todo!()
        }
        BooleanFunction::IsInfinite => todo!(),
        BooleanFunction::IsFirstDistinct => {
            todo!()
        }
        BooleanFunction::IsLastDistinct => {
            todo!()
        }
        BooleanFunction::IsUnique => todo!(),
        BooleanFunction::IsDuplicated => todo!(),
        BooleanFunction::IsClose {
            abs_tol,
            rel_tol,
            nans_equal,
        } => todo!(),
    }
}

/// Compile is_between to an ExprPlan.
pub(super) fn compile_is_between_lazy(
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

    try_extract!(let Some(col) = expr_to_col_name(expr));
    try_extract!(let Expr::Literal(low_lit) = strip_wrappers(low));
    try_extract!(let Expr::Literal(high_lit) = strip_wrappers(high));

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
pub(super) fn compile_is_in_lazy(
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

    try_extract!(let Some(col) = expr_to_col_name(expr));
    try_extract!(let Expr::Literal(list_lit) = strip_wrappers(list));

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
