//! Lazy expression compilation - produces Sel without resolving.
//!
//! This module mirrors `compile_node.rs` but produces lazy selections that store
//! `ValueRange` constraints instead of resolved index ranges.

use snafu::ResultExt;

use super::compile_ctx::LazyCompileCtx;
use super::compile_node::{
    collect_column_refs, extract_struct_field_path,
};
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
use super::{Emptyable, SetOperations};
use crate::chunk_plan::exprs::expr_utils::all_for_referenced_vars_lazy;
use crate::chunk_plan::indexing::lazy_selection::{
    LazyArraySelection, LazyDatasetSelection as Sel,
    LazyDimConstraint, LazyHyperRectangle,
    lazy_dataset_all_for_vars,
    lazy_dataset_for_vars_with_selection,
};
use crate::chunk_plan::indexing::types::{
    ValueRange, ValueRangePresent, HasIntersect,
};
use crate::chunk_plan::prelude::*;
use crate::{IStr, IntoIStr};
use crate::errors::BackendError;

use super::expr_utils::expr_to_col_name;
use std::sync::Arc;

type LazyResult = Result<Sel, BackendError>;

/// Compile an expression to a lazy dataset selection.
///
/// This function traverses the expression tree and produces a `Sel`
/// containing unresolved `ValueRange` constraints. These constraints can later be
/// batch-resolved and materialized into a concrete `DatasetSelection`.
pub(crate) fn compile_expr(
    expr: impl std::borrow::Borrow<Expr>,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    let expr: &Expr =
        std::borrow::Borrow::borrow(&expr);
    match expr {
        Expr::Display { inputs, fmt_str } => {
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
            let func_sel = compile_expr(
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
                Ok(func_sel)
            } else {
                let part_sel =
                    lazy_dataset_all_for_vars(
                        refs, ctx.meta,
                    );
                Ok(func_sel.union(&part_sel))
            }
        }

        Expr::Rolling { function, .. } => {
            compile_expr(function.as_ref(), ctx)
        }

        Expr::Filter { input, by } => {
            let filter_sel =
                compile_expr(by.as_ref(), ctx)?;
            // If the filter predicate is proven always-false, no chunks match
            if filter_sel.is_empty() {
                return Ok(Sel::Empty);
            }
            let input_sel = compile_expr(
                input.as_ref(),
                ctx,
            )?;
            // Use intersect so the filter predicate constrains which chunks to read.
            // Union would take the less-restrictive "all" from the input projection,
            // completely ignoring the filter predicate.
            Ok(input_sel.intersect(&filter_sel))
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
                        if matches!(
                            function,
                            FunctionExpr::Boolean(
                                BooleanFunction::Not
                            )
                        ) && input.len() == 1
                        {
                            let a = compile_expr(
                                left.as_ref(),
                                ctx,
                            )?;
                            let b = compile_expr(
                                &input[0], ctx,
                            )?;
                            return Ok(a.difference(&b));
                        }
                    }
                    if let Expr::Function {
                        input,
                        function,
                    } = strip_wrappers(
                        left.as_ref(),
                    ) {
                        if matches!(
                            function,
                            FunctionExpr::Boolean(
                                BooleanFunction::Not
                            )
                        ) && input.len() == 1
                        {
                            let a = compile_expr(
                                right.as_ref(),
                                ctx,
                            )?;
                            let b = compile_expr(
                                &input[0], ctx,
                            )?;
                            return Ok(a.difference(&b));
                        }
                    }

                    // Fast path: merge compatible comparisons on the same column
                    if let (
                        Some((col_a, vr_a)),
                        Some((col_b, vr_b)),
                    ) = (
                        try_expr_to_value_range_lazy(
                            left.as_ref(),
                            ctx,
                        ),
                        try_expr_to_value_range_lazy(
                            right.as_ref(),
                            ctx,
                        ),
                    ) {
                        if col_a == col_b {
                            let vr = vr_a.intersect(Some(vr_b)).flatten();
                            return compile_value_range_to_lazy_selection(&col_a, &vr, ctx);
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
                    Ok(a.difference(&b)
                        .union(&b.difference(&a)))
                }
                Operator::Eq
                | Operator::GtEq
                | Operator::Gt
                | Operator::LtEq
                | Operator::Lt => {
                    // Check for struct field access (e.g., model_a.struct.field("temp") > 280)
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
                            return compile_struct_field_cmp(&struct_col, &field_name, *op, lit, ctx)
                                .or_else(|_| Ok(all_for_referenced_vars_lazy(expr, ctx)));
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
                            return compile_struct_field_cmp(&struct_col, &field_name, reverse_operator(*op), lit, ctx)
                                .or_else(|_| Ok(all_for_referenced_vars_lazy(expr, ctx)));
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
                        compile_cmp_to_lazy_selection(&col, op_eff, &lit, ctx)
                            .or_else(|_| Ok(lazy_dataset_all_for_vars(vec![col.istr()], ctx.meta)))
                    } else {
                        Ok(all_for_referenced_vars_lazy(
                            expr, ctx,
                        ))
                    }
                }
                _ => Ok(
                    all_for_referenced_vars_lazy(
                        expr, ctx,
                    ),
                ),
            }
        }

        Expr::Literal(lit) => {
            match literal_anyvalue(lit) {
                Some(AnyValue::Boolean(true)) => {
                    Ok(Sel::NoSelectionMade)
                }
                Some(AnyValue::Boolean(
                    false,
                )) => Ok(Sel::Empty),
                Some(AnyValue::Null) => {
                    Ok(Sel::Empty)
                }
                _ => Ok(Sel::NoSelectionMade),
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
                        return Ok(
                            Sel::NoSelectionMade,
                        );
                    }
                    return Ok(Sel::Empty);
                }
            }

            let predicate_node = compile_expr(
                predicate.as_ref(),
                ctx,
            )?;
            let truthy_node = compile_expr(
                truthy.as_ref(),
                ctx,
            )?;
            let falsy_node = compile_expr(
                falsy.as_ref(),
                ctx,
            )?;
            Ok(truthy_node
                .union(&falsy_node)
                .union(&predicate_node))
        }

        Expr::Function { input, function } => {
            match function {
                FunctionExpr::Boolean(bf) => {
                    compile_boolean_function_lazy(
                        bf, input, ctx,
                    )
                }
                FunctionExpr::NullCount => {
                    Ok(Sel::NoSelectionMade)
                }
                FunctionExpr::FfiPlugin {
                    symbol,
                    ..
                } => {
                    if symbol == "interpolate_nd"
                    {
                        if input.len() < 3 {
                            return Ok(Sel::NoSelectionMade);
                        }
                        interpolate_selection_nd_lazy(
                            &input[0], &input[1],
                            &input[2], ctx,
                        )
                    } else {
                        Ok(all_for_referenced_vars_lazy(
                            expr, ctx,
                        ))
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
                    if refs.is_empty() {
                        Ok(Sel::NoSelectionMade)
                    } else {
                        Ok(lazy_dataset_all_for_vars(
                            refs, ctx.meta,
                        ))
                    }
                }
            }
        }

        Expr::Selector(selector) => {
            compile_selector_lazy(selector, ctx)
        }

        Expr::Column(name) => {
            Ok(lazy_dataset_all_for_vars(
                vec![name.istr()],
                ctx.meta,
            ))
        }

        Expr::Agg(_) => {
            Ok(all_for_referenced_vars_lazy(
                expr, ctx,
            ))
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
            if refs.is_empty() {
                Ok(Sel::NoSelectionMade)
            } else {
                Ok(lazy_dataset_all_for_vars(
                    refs, ctx.meta,
                ))
            }
        }

        Expr::Gather { expr, idx, .. } => {
            let mut refs = Vec::new();
            collect_column_refs(expr, &mut refs);
            collect_column_refs(idx, &mut refs);
            refs.sort();
            refs.dedup();
            if refs.is_empty() {
                Ok(Sel::NoSelectionMade)
            } else {
                Ok(lazy_dataset_all_for_vars(
                    refs, ctx.meta,
                ))
            }
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
                Ok(Sel::NoSelectionMade)
            } else {
                Ok(lazy_dataset_all_for_vars(
                    vars, ctx.meta,
                ))
            }
        }

        Expr::Element
        | Expr::Len
        | Expr::SubPlan(_, _)
        | Expr::DataTypeFunction(_) => {
            Ok(Sel::NoSelectionMade)
        }

        Expr::StructEval { expr, .. } => {
            Ok(all_for_referenced_vars_lazy(
                expr.as_ref(),
                ctx,
            ))
        }
    }
}

/// Compile a comparison to a lazy selection.
fn compile_cmp_to_lazy_selection(
    col: &IStr,
    op: Operator,
    lit: &LiteralValue,
    ctx: &LazyCompileCtx<'_>,
) -> LazyResult {
    let time_encoding = ctx
        .meta
        .arrays
        .get(&col.istr())
        .and_then(|a| a.time_encoding.as_ref());
    let scalar =
        literal_to_scalar(lit, time_encoding)?;

    let vr = ValueRangePresent::from_polars_op(
        op, scalar,
    )?;

    compile_value_range_to_lazy_selection(
        col,
        &Some(vr),
        ctx,
    )
}

/// Compile a value range to a lazy selection.
fn compile_value_range_to_lazy_selection(
    col: &str,
    vrr: &ValueRange,
    ctx: &LazyCompileCtx<'_>,
) -> LazyResult {
    if let Some(vr) = vrr {
        // Check if this is a dimension
        let dim_idx = ctx.dim_index(col);
        if dim_idx.is_none() {
            // Not a dimension: skip pushdown and let runtime filtering handle it.
            return Ok(Sel::NoSelectionMade);
        }

        // Create a lazy constraint with the unresolved value range
        let constraint =
            LazyDimConstraint::Unresolved(Some(
                vr.clone(),
            ));
        let rect = LazyHyperRectangle::all()
            .with_dim(col.istr(), constraint);
        let sel =
            LazyArraySelection::from_rectangle(
                rect,
            );

        Ok(lazy_dataset_for_vars_with_selection(
            ctx.vars.iter().cloned(),
            ctx.meta,
            sel,
        ))
    } else {
        Ok(Sel::Empty)
    }
}

/// Compile a struct field comparison to a lazy selection.
/// Maps model_a.struct.field("temperature") > 280 to constraint on "model_a/temperature" array.
fn compile_struct_field_cmp(
    struct_col: &IStr,
    field_name: &IStr,
    op: Operator,
    lit: &LiteralValue,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    // Build the array path (e.g., "model_a/temperature")
    let array_path: IStr =
        format!("{}/{}", struct_col, field_name)
            .istr();

    // Look up array metadata using unified meta when available
    let arr_meta = if let Some(unified) =
        ctx.unified_meta
    {
        let key = unified
            .normalize_array_path(array_path.as_ref())
            .ok_or_else(|| {
                BackendError::StructFieldNotFound {
                    path: array_path.clone()
                }
            })?;
        unified.path_to_array.get(&key)
    } else {
        ctx.meta.arrays.get(&array_path)
    };

    let time_encoding = arr_meta
        .and_then(|a| a.time_encoding.as_ref());
    let scalar =
        literal_to_scalar(lit, time_encoding)?;

    let vr = ValueRangePresent::from_polars_op(
        op, scalar,
    )?;

    // For struct fields, we need to find which dimensions apply
    // If the array is a dimension array, constrain that dimension
    // Otherwise, just return "all" for the referenced vars
    if let Some(meta) = arr_meta {
        if meta.dims.len() == 1 {
            let dim = &meta.dims[0];
            if ctx.dims.contains(dim) {
                // This is a 1D coordinate-like array, apply constraint to its dimension
                let constraint =
                    LazyDimConstraint::Unresolved(
                        Some(vr.clone()),
                    );
                let rect =
                    LazyHyperRectangle::all()
                        .with_dim(
                            dim.clone(),
                            constraint,
                        );
                let sel =
                    LazyArraySelection::from_rectangle(
                        rect,
                    );
                return Ok(
                    lazy_dataset_for_vars_with_selection(
                        ctx.vars.iter().cloned(),
                        ctx.meta,
                        sel,
                    ),
                );
            }
        }
    }

    // Fallback: treat as regular variable reference
    Ok(lazy_dataset_all_for_vars(
        vec![array_path],
        ctx.meta,
    ))
}

/// Compile a boolean function to a lazy selection.
fn compile_boolean_function_lazy(
    bf: &BooleanFunction,
    input: &[Expr],
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    match bf {
        BooleanFunction::Not => {
            let [arg] = input else {
                return Err(
                    BackendError::UnsupportedBooleanFunction {
                        function: bf.clone()
                    }
                );
            };
            if let Expr::Literal(lit) =
                strip_wrappers(arg)
            {
                return match literal_anyvalue(lit)
                {
                    Some(AnyValue::Boolean(
                        true,
                    )) => Ok(Sel::Empty),
                    Some(AnyValue::Boolean(
                        false,
                    )) => {
                        Ok(Sel::NoSelectionMade)
                    }
                    Some(AnyValue::Null) => {
                        Ok(Sel::Empty)
                    }
                    _ => Ok(Sel::NoSelectionMade),
                };
            }
            let inner = compile_expr(arg, ctx)
                .unwrap_or_else(|_| {
                    Sel::NoSelectionMade
                });
            if inner.is_empty() {
                Ok(Sel::NoSelectionMade)
            } else {
                Ok(Sel::NoSelectionMade)
                // return Ok(inner);
            }
        }
        BooleanFunction::IsNull
        | BooleanFunction::IsNotNull => {
            let [arg] = input else {
                return Err(
                    BackendError::UnsupportedBooleanFunction {
                        function: bf.clone()
                    }
                );
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
                    Sel::NoSelectionMade
                } else {
                    Sel::Empty
                });
            }
            Ok(Sel::NoSelectionMade)
        }
        BooleanFunction::IsBetween { .. } => {
            compile_is_between_lazy(input, ctx)
        }
        BooleanFunction::IsIn { .. } => {
            compile_is_in_lazy(input, ctx)
        }
        BooleanFunction::AnyHorizontal => {
            let mut acc = Sel::Empty;
            for e in input {
                let sel = compile_expr(e, ctx)
                    .unwrap_or_else(|_| {
                        Sel::NoSelectionMade
                    });
                acc = acc.union(&sel);
            }
            Ok(acc)
        }
        BooleanFunction::AllHorizontal => {
            let mut acc = Sel::NoSelectionMade;
            for e in input {
                let sel = compile_expr(e, ctx)
                    .unwrap_or_else(|_| {
                        Sel::NoSelectionMade
                    });
                acc = acc.intersect(&sel);
                if acc.is_empty() {
                    break;
                }
            }
            Ok(acc)
        }
        _ => Ok(Sel::NoSelectionMade),
    }
}

/// Compile is_between to lazy selection.
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
        return Ok(Sel::NoSelectionMade);
    };
    let Expr::Literal(low_lit) =
        strip_wrappers(low)
    else {
        return Ok(Sel::NoSelectionMade);
    };
    let Expr::Literal(high_lit) =
        strip_wrappers(high)
    else {
        return Ok(Sel::NoSelectionMade);
    };

    let a = compile_cmp_to_lazy_selection(
        &col,
        Operator::GtEq,
        low_lit,
        ctx,
    )
    .unwrap_or_else(|_| Sel::NoSelectionMade);
    let b = compile_cmp_to_lazy_selection(
        &col,
        Operator::LtEq,
        high_lit,
        ctx,
    )
    .unwrap_or_else(|_| Sel::NoSelectionMade);
    Ok(a.intersect(&b))
}

/// Compile is_in to lazy selection.
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
        return Ok(Sel::NoSelectionMade);
    };
    let Expr::Literal(list_lit) =
        strip_wrappers(list)
    else {
        return Ok(Sel::NoSelectionMade);
    };

    let (dtype, values): (
        &polars::prelude::DataType,
        Vec<AnyValue<'static>>,
    ) = match list_lit {
        LiteralValue::Series(s) => {
            let series = &**s;
            if series.len() > 4096 {
                return Ok(Sel::NoSelectionMade);
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
                        return Err(
                            BackendError::compile_polars(
                                format!(
                                    "list literal is too long: {:?}",
                                    series
                                )
                            )
                        );
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
                        return Err(
                            BackendError::compile_polars(
                                format!(
                                    "array literal is too long: {:?}",
                                    series
                                )
                            )
                        );
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
                        Sel::NoSelectionMade,
                    );
                }
            }
        }
        _ => {
            return Ok(Sel::NoSelectionMade);
        }
    };

    let mut out: Option<Sel> = None;
    for av in values {
        if matches!(av, AnyValue::Null) {
            return Ok(Sel::NoSelectionMade);
        }
        let lit = LiteralValue::Scalar(
            Scalar::new(dtype.clone(), av),
        );
        let node = compile_cmp_to_lazy_selection(
            &col,
            Operator::Eq,
            &lit,
            ctx,
        )
        .unwrap_or_else(|_| Sel::NoSelectionMade);

        if node == Sel::NoSelectionMade {
            return Ok(Sel::NoSelectionMade);
        }
        out = Some(match out.take() {
            None => node,
            Some(acc) => acc.union(&node),
        });
    }
    Ok(out.unwrap_or_else(Sel::empty))
}

/// Compile selector to lazy selection.
fn compile_selector_lazy(
    selector: &Selector,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    use regex::Regex;

    match selector {
        Selector::Union(left, right) => {
            let left_node =
                compile_selector_lazy(
                    left.as_ref(),
                    ctx,
                )?;
            let right_node =
                compile_selector_lazy(
                    right.as_ref(),
                    ctx,
                )?;
            Ok(left_node.union(&right_node))
        }
        Selector::Difference(left, right) => {
            let left_node =
                compile_selector_lazy(
                    left.as_ref(),
                    ctx,
                )?;
            let right_node =
                compile_selector_lazy(
                    right.as_ref(),
                    ctx,
                )?;
            Ok(left_node.difference(&right_node))
        }
        Selector::ExclusiveOr(left, right) => {
            let left_node =
                compile_selector_lazy(
                    left.as_ref(),
                    ctx,
                )?;
            let right_node =
                compile_selector_lazy(
                    right.as_ref(),
                    ctx,
                )?;
            Ok(left_node
                .exclusive_or(&right_node))
        }
        Selector::Intersect(left, right) => {
            let left_node =
                compile_selector_lazy(
                    left.as_ref(),
                    ctx,
                )?;
            let right_node =
                compile_selector_lazy(
                    right.as_ref(),
                    ctx,
                )?;
            Ok(left_node.intersect(&right_node))
        }
        Selector::Empty => Ok(Sel::Empty),
        Selector::ByName { names, .. } => {
            let vars: Vec<IStr> = names
                .iter()
                .map(|s| s.istr())
                .collect();
            Ok(lazy_dataset_all_for_vars(
                vars, ctx.meta,
            ))
        }
        Selector::Matches(pattern) => {
            let re = Regex::new(pattern.as_str())
                .context(crate::errors::backend::RegexSnafu {
                    pattern: pattern.clone(),
                })?;
            let matching_vars: Vec<IStr> = ctx
                .meta
                .data_vars
                .iter()
                .filter(|v| {
                    re.is_match(v.as_ref())
                })
                .cloned()
                .collect();
            if matching_vars.is_empty() {
                Ok(Sel::Empty)
            } else {
                Ok(lazy_dataset_all_for_vars(
                    matching_vars,
                    ctx.meta,
                ))
            }
        }
        Selector::ByDType(dtype_selector) => {
            let matching_vars: Vec<IStr> = ctx
                .meta
                .data_vars
                .iter()
                .filter(|v| {
                    if let Some(array_meta) =
                        ctx.meta.arrays.get(v)
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
                Ok(Sel::Empty)
            } else {
                Ok(lazy_dataset_all_for_vars(
                    matching_vars,
                    ctx.meta,
                ))
            }
        }
        Selector::ByIndex { .. } => {
            Ok(Sel::NoSelectionMade)
        }
        Selector::Wildcard => {
            Ok(lazy_dataset_all_for_vars(
                ctx.meta.data_vars.clone(),
                ctx.meta,
            ))
        }
    }
}

/// Compile interpolation selection (lazy version).
///
/// For interpolation, we extract the target coordinate values and create lazy constraints.
/// The interpolation needs "bracketing" indices - for a value between two indices,
/// we need both indices. This is done by creating per-point constraints.
fn interpolate_selection_nd_lazy(
    source_coords: &Expr,
    source_values: &Expr,
    target_values: &Expr,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    use crate::chunk_plan::indexing::types::CoordScalar;

    // Extract coordinate dimension names from the source coord struct expression.
    let coord_names =
        extract_column_names_lazy(source_coords);
    if coord_names.is_empty() {
        return Ok(Sel::NoSelectionMade);
    }

    // Extract the target values struct (a literal Series containing the target points).
    let Some(target_struct) =
        extract_literal_struct_series_lazy(
            target_values,
        )
    else {
        return Ok(Sel::NoSelectionMade);
    };

    let Ok(target_sc) = target_struct.struct_()
    else {
        return Ok(Sel::NoSelectionMade);
    };
    let target_fields =
        target_sc.fields_as_series();

    // For each dimension, collect all the target values
    let mut dim_values: std::collections::BTreeMap<IStr, Vec<CoordScalar>> = std::collections::BTreeMap::new();

    for name in &coord_names {
        // Only constrain actual dataset dimensions
        if !ctx.dims.iter().any(|d| d == name) {
            continue;
        }

        // If this coord dim isn't present in the target DataFrame, it's a group key
        let Some(s) = target_fields.iter().find(|s| {
            s.name() == <IStr as AsRef<str>>::as_ref(name)
        }) else {
            continue;
        };

        let Some(values) =
            series_values_scalar_lazy(s)
        else {
            return Ok(Sel::NoSelectionMade);
        };

        if !values.is_empty() {
            dim_values
                .insert(name.clone(), values);
        }
    }

    if dim_values.is_empty() {
        return Ok(Sel::NoSelectionMade);
    }

    // For each dimension, create interpolation constraints for each unique target value.
    // Each target value needs bracketing (the indices on both sides).
    let mut constraints: std::collections::BTreeMap<IStr, LazyDimConstraint> = std::collections::BTreeMap::new();

    for (dim_name, values) in dim_values {
        // Create per-point interpolation constraints.
        // We'll store all values and resolve them during materialization.
        let interp_values: Vec<CoordScalar> =
            values;

        // Create a special constraint that holds all the interpolation target values
        constraints.insert(
            dim_name,
            LazyDimConstraint::UnresolvedInterpolationPoints(Arc::new(interp_values)),
        );
    }

    let rect = LazyHyperRectangle::with_dims(
        constraints,
    );
    let sel =
        LazyArraySelection::from_rectangle(rect);

    // Here, use source_values for the list of variables we want to retrieve
    // This is typically an Expr::Field containing variable names
    // May be an as_struct("var_a, var_b, var_c")
    let retrieve_vars = match source_values {
        Expr::Function { input, function } => {
            match function {
                FunctionExpr::AsStruct => input
                    .clone()
                    .iter()
                    .map(|n| -> Result<IStr, BackendError> {
                        let Expr::Column(name) = strip_wrappers(n) else {
                            return Err(
                                BackendError::compile_polars(
                                    format!(
                                        "source_values must be an Expr::Function with FunctionExpr::AsStruct containing variable names: {:?}",
                                        source_values
                                    ),
                                ),
                            );
                        };
                        Ok(name.istr())
                    })
                    .collect::<Result<Vec<IStr>, BackendError>>()?,
                _ => {
                    return Err(
                        BackendError::compile_polars(
                            format!(
                                "source_values must be an Expr::Function with FunctionExpr::AsStruct containing variable names: {:?}",
                                source_values
                            ),
                        ),
                    );
                }
            }
        }
        Expr::Field(names) => names
            .iter()
            .map(|n| n.istr())
            .collect::<Vec<_>>(),
        _ => {
            return Err(
                BackendError::compile_polars(
                    format!(
                        "source_values must be an Expr::Field containing variable names: {:?}",
                        source_values
                    ),
                ),
            );
        }
    };

    Ok(lazy_dataset_for_vars_with_selection(
        retrieve_vars,
        ctx.meta,
        sel,
    ))
}
