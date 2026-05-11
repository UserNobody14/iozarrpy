//! Interpolation selection compilation (interpolate_nd / interpolate_geospatial FfiPlugin).

use super::super::compile_ctx::LazyCompileCtx;
use super::super::expr_plan::{ExprPlan, VarSet};
use super::super::expr_utils::{
    extract_column_names_lazy,
    extract_literal_struct_series_lazy,
    series_values_scalar_lazy,
};
use super::super::literals::strip_wrappers;
use crate::chunk_plan::coord_resolve::Expansion;
use crate::chunk_plan::exprs::compile::expr::compile_expr_list;
use crate::chunk_plan::exprs::compile::utils::collect_refs_from_expr_list;
use crate::chunk_plan::indexing::index_set::RectangleSet;
use crate::chunk_plan::indexing::types::{
    CoordScalar, ValueRangePresent,
};
use crate::chunk_plan::prelude::*;
use crate::errors::BackendError;
use crate::shared::{IStr, IntoIStr, IntoManyIstrs};
use crate::try_extract;

type LazyResult = Result<ExprPlan, BackendError>;

/// Resolve a per-row constraint map into a single rectangle set (the AND of
/// per-dim resolved index ranges).
fn resolve_row_constraints(
    constraints: &[(IStr, ValueRangePresent, Expansion)],
    ctx: &LazyCompileCtx<'_>,
) -> Result<RectangleSet, BackendError> {
    let mut acc: Option<RectangleSet> = None;
    for (dim, vr, exp) in constraints {
        let ranges = ctx.resolve(*dim, vr, *exp)?;
        let one = RectangleSet::from_dim_constraint(
            ctx.universe.dims.clone(),
            ctx.universe.shape.clone(),
            *dim,
            &ranges,
        );
        acc = Some(match acc {
            Some(prev) => prev.intersect(&one),
            None => one,
        });
    }
    Ok(acc.unwrap_or_else(|| {
        RectangleSet::full(
            ctx.universe.dims.clone(),
            ctx.universe.shape.clone(),
        )
    }))
}

/// Compile interpolation selection.
pub(super) fn interpolate_selection_nd_lazy(
    source_coords: &Expr,
    source_values: &Expr,
    target_values: &Expr,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    try_extract!(let Some(coord_names) = extract_column_names_lazy(source_coords));
    try_extract!(let Some(target_struct) =
        extract_literal_struct_series_lazy(
            target_values,
        )
    );
    try_extract!(let Ok(target_sc) = target_struct.struct_());
    let target_fields =
        target_sc.fields_as_series();

    let mut dim_values: std::collections::BTreeMap<IStr, Vec<CoordScalar>> =
        std::collections::BTreeMap::new();

    for s in target_fields.iter() {
        let name = s.name().as_str().istr();
        if !ctx.dims().iter().any(|d| d == &name) {
            continue;
        }
        try_extract!(let Some(values) = series_values_scalar_lazy(s));
        if !values.is_empty() {
            dim_values.insert(name, values);
        }
    }

    if dim_values.is_empty() {
        return Ok(ExprPlan::NoConstraint);
    }

    let num_rows = dim_values
        .values()
        .next()
        .map(|v| v.len())
        .unwrap_or(0);
    let mut row_rects: Vec<RectangleSet> =
        Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let mut row: Vec<(
            IStr,
            ValueRangePresent,
            Expansion,
        )> = Vec::with_capacity(dim_values.len());
        for (dim_name, values) in dim_values.iter()
        {
            let value = values[i].clone();
            let vr = ValueRangePresent::from_equal_case(value);
            let is_interp_dim = coord_names
                .iter()
                .any(|c| c == dim_name);
            let exp = if is_interp_dim {
                Expansion::InterpolationNeighbor
            } else {
                Expansion::Exact
            };
            row.push((*dim_name, vr, exp));
        }
        row_rects.push(resolve_row_constraints(
            &row, ctx,
        )?);
    }

    let combined =
        union_rows(row_rects, ctx);

    let (retrieve_vars, filter_plan) =
        match source_values {
            Expr::Function {
                input,
                function,
            } => match function {
                FunctionExpr::AsStruct => {
                    let filter_initial =
                        compile_expr_list(
                            input, ctx,
                        )?;
                    let vars =
                        collect_refs_from_expr_list(input);
                    (vars, Some(filter_initial))
                }
                _ => {
                    return Err(BackendError::compile_polars(format!(
                        "source_values must be an Expr::Function with FunctionExpr::AsStruct \
                         containing column refs or col(...).filter(predicate): {:?}",
                        source_values
                    )));
                }
            },
            Expr::Field(names) => {
                (names.into_istrs(), None)
            }
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

    let plan = ExprPlan::active(
        VarSet::from_vec(retrieve_vars),
        combined,
    );
    let plan = match filter_plan {
        Some(ExprPlan::Empty) => {
            return Ok(ExprPlan::Empty);
        }
        Some(p) => plan.intersect(&p),
        None => plan,
    };
    Ok(plan)
}

fn union_rows(
    rows: Vec<RectangleSet>,
    ctx: &LazyCompileCtx<'_>,
) -> RectangleSet {
    rows.into_iter()
        .reduce(|a, b| a.union(&b))
        .unwrap_or_else(|| {
            RectangleSet::empty(
                ctx.universe.dims.clone(),
                ctx.universe.shape.clone(),
            )
        })
}

/// Extract coordinate column names from an AsStruct expression, preserving
/// input order. For `pl.struct([source_lat, source_lon])` this yields
/// `["lat", "lon"]` — the last element is always longitude.
fn extract_coord_names_ordered(
    expr: &Expr,
) -> Option<Vec<IStr>> {
    let expr = strip_wrappers(expr);
    if let Expr::Function { input, function } =
        expr
        && matches!(
            function,
            FunctionExpr::AsStruct
        )
    {
        let mut names =
            Vec::with_capacity(input.len());
        for e in input {
            let mut found = None;
            walk_for_first_column(
                strip_wrappers(e),
                &mut found,
            );
            names.push(found?);
        }
        return if names.is_empty() {
            None
        } else {
            Some(names)
        };
    }
    None
}

fn walk_for_first_column(
    expr: &Expr,
    out: &mut Option<IStr>,
) {
    if out.is_some() {
        return;
    }
    match expr {
        Expr::Column(name) => {
            *out = Some(name.istr());
        }
        Expr::Alias(inner, _)
        | Expr::KeepName(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::Sort { expr: inner, .. } => {
            walk_for_first_column(inner, out);
        }
        _ => {}
    }
}

/// Compile geospatial interpolation selection.
///
/// Like `interpolate_selection_nd_lazy` but uses `WrappingGhost` expansion
/// for the longitude dimension (the last coordinate) to handle ghost-point
/// expansion for periodic grids.
pub(super) fn interpolate_selection_geospatial_lazy(
    source_coords: &Expr,
    source_values: &Expr,
    target_values: &Expr,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    let coord_names_ordered =
        extract_coord_names_ordered(source_coords);

    let coord_names: Vec<IStr> =
        match coord_names_ordered.as_deref() {
            Some(names) => names.to_vec(),
            None => {
                try_extract!(let Some(names) = extract_column_names_lazy(source_coords));
                names
            }
        };

    let lon_dim: Option<&IStr> =
        coord_names_ordered
            .as_ref()
            .and_then(|names| names.last());

    try_extract!(let Some(target_struct) =
        extract_literal_struct_series_lazy(target_values)
    );
    try_extract!(let Ok(target_sc) = target_struct.struct_());
    let target_fields =
        target_sc.fields_as_series();

    let mut dim_values: std::collections::BTreeMap<IStr, Vec<CoordScalar>> =
        std::collections::BTreeMap::new();

    for s in target_fields.iter() {
        let name = s.name().as_str().istr();
        if !ctx.dims().iter().any(|d| d == &name) {
            continue;
        }
        try_extract!(let Some(values) = series_values_scalar_lazy(s));
        if !values.is_empty() {
            dim_values.insert(name, values);
        }
    }

    if dim_values.is_empty() {
        return Ok(ExprPlan::NoConstraint);
    }

    let num_rows = dim_values
        .values()
        .next()
        .map(|v| v.len())
        .unwrap_or(0);
    let mut row_rects: Vec<RectangleSet> =
        Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let mut row: Vec<(
            IStr,
            ValueRangePresent,
            Expansion,
        )> = Vec::with_capacity(dim_values.len());
        for (dim_name, values) in dim_values.iter()
        {
            let value = values[i].clone();
            let vr = ValueRangePresent::from_equal_case(value);
            let is_interp_dim = coord_names
                .iter()
                .any(|c| c == dim_name);
            let is_lon =
                lon_dim == Some(dim_name);
            let exp = if is_lon {
                Expansion::WrappingGhost
            } else if is_interp_dim {
                Expansion::InterpolationNeighbor
            } else {
                Expansion::Exact
            };
            row.push((*dim_name, vr, exp));
        }
        row_rects.push(resolve_row_constraints(
            &row, ctx,
        )?);
    }

    let combined = union_rows(row_rects, ctx);

    let (retrieve_vars, filter_plan) =
        match source_values {
            Expr::Function {
                input,
                function,
            } => match function {
                FunctionExpr::AsStruct => {
                    let filter_initial =
                        compile_expr_list(
                            input, ctx,
                        )?;
                    let vars =
                        collect_refs_from_expr_list(input);
                    (vars, Some(filter_initial))
                }
                _ => {
                    return Err(BackendError::compile_polars(format!(
                        "source_values must be an Expr::Function with FunctionExpr::AsStruct \
                         containing column refs or col(...).filter(predicate): {:?}",
                        source_values
                    )));
                }
            },
            Expr::Field(names) => {
                (names.into_istrs(), None)
            }
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

    let plan = ExprPlan::active(
        VarSet::from_vec(retrieve_vars),
        combined,
    );
    let plan = match filter_plan {
        Some(ExprPlan::Empty) => {
            return Ok(ExprPlan::Empty);
        }
        Some(p) => plan.intersect(&p),
        None => plan,
    };
    Ok(plan)
}
