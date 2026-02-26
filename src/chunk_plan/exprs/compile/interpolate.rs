//! Interpolation selection compilation (interpolate_nd FfiPlugin).

use super::super::compile_ctx::LazyCompileCtx;
use super::super::expr_plan::{ExprPlan, VarSet};
use super::super::expr_utils::{
    extract_column_names_lazy,
    extract_literal_struct_series_lazy,
    extract_var_from_source_value_expr,
    series_values_scalar_lazy,
};
use super::expr::compile_expr;
use crate::chunk_plan::indexing::lazy_selection::{
    LazyArraySelection, LazyDimConstraint, LazyHyperRectangle,
};
use crate::chunk_plan::indexing::selection::SetOperations;
use crate::chunk_plan::indexing::types::{CoordScalar, ValueRangePresent};
use crate::chunk_plan::prelude::*;
use crate::errors::BackendError;
use crate::{IStr, IntoIStr};

type LazyResult = Result<ExprPlan, BackendError>;

/// Compile interpolation selection (lazy version).
pub(super) fn interpolate_selection_nd_lazy(
    source_coords: &Expr,
    source_values: &Expr,
    target_values: &Expr,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    let coord_names = extract_column_names_lazy(source_coords);
    if coord_names.is_empty() {
        return Ok(ExprPlan::NoConstraint);
    }

    let Some(target_struct) = extract_literal_struct_series_lazy(target_values) else {
        return Ok(ExprPlan::NoConstraint);
    };

    let Ok(target_sc) = target_struct.struct_() else {
        return Ok(ExprPlan::NoConstraint);
    };
    let target_fields = target_sc.fields_as_series();

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

        let Some(values) = series_values_scalar_lazy(s) else {
            return Ok(ExprPlan::NoConstraint);
        };

        if !values.is_empty() {
            dim_values.insert(name, values);
        }
    }

    if dim_values.is_empty() {
        return Ok(ExprPlan::NoConstraint);
    }

    let mut constraints: Vec<std::collections::BTreeMap<IStr, LazyDimConstraint>> = vec![];

    // Transform dim_values (dim -> Vec<CoordScalar>) to row-wise constraints.
    // Use InterpolationRange (with expansion) only for coord_names; use Unresolved
    // (no expansion) for filter dimensions so we load only the exact slice.
    let num_rows = dim_values.values().next().map(|v| v.len()).unwrap_or(0);
    for i in 0..num_rows {
        let mut constraint = std::collections::BTreeMap::new();
        for (dim_name, values) in dim_values.iter() {
            let value = values[i].clone();
            let vr = ValueRangePresent::from_equal_case(value);
            let is_interp_dim = coord_names.iter().any(|c| c == dim_name);
            let c = if is_interp_dim {
                LazyDimConstraint::InterpolationRange(vr)
            } else {
                LazyDimConstraint::Unresolved(vr)
            };
            constraint.insert(dim_name.clone(), c);
        }
        constraints.push(constraint);
    }

    let rects: Vec<LazyHyperRectangle> = constraints
        .into_iter()
        .map(|c| LazyHyperRectangle::with_dims(c))
        .collect();
    let sel = LazyArraySelection::Rectangles(rects.into());

    let (retrieve_vars, filter_plan) = match source_values {
        Expr::Function { input, function } => match function {
            FunctionExpr::AsStruct => {
                let mut vars = Vec::with_capacity(input.len());
                let mut filter_plan_acc: Option<ExprPlan> = None;
                for n in input {
                    let Some((names, filter_pred)) =
                        extract_var_from_source_value_expr(n)
                    else {
                        return Err(BackendError::compile_polars(format!(
                            "source_values must be an Expr::Function with FunctionExpr::AsStruct \
                             containing column refs or col(...).filter(predicate): {:?}",
                            source_values
                        )));
                    };
                    vars.extend(names);
                    if let Some(pred) = filter_pred {
                        let fp = compile_expr(pred, ctx)?;
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
            names.iter().map(|n| n.istr()).collect::<Vec<_>>(),
            None,
        ),
        _ => {
            return Err(BackendError::compile_polars(format!(
                "source_values must be an Expr::Field or AsStruct containing variable names: {:?}",
                source_values
            )));
        }
    };

    let sel = match filter_plan {
        Some(ExprPlan::Active {
            constraints: filter_sel,
            ..
        }) => sel.intersect(filter_sel.as_ref()),
        Some(ExprPlan::Empty) => return Ok(ExprPlan::Empty),
        Some(ExprPlan::NoConstraint) | None => sel,
    };

    if sel.is_empty() {
        return Ok(ExprPlan::Empty);
    }

    Ok(ExprPlan::constrained(VarSet::from_vec(retrieve_vars), sel))
}
