//! Comparison and value range compilation.

use super::super::compile_ctx::LazyCompileCtx;
use super::super::expr_plan::{ExprPlan, VarSet};
use super::super::literals::literal_to_scalar;
use crate::ensure_some;
use crate::chunk_plan::indexing::lazy_selection::{
    LazyArraySelection, LazyDimConstraint, LazyHyperRectangle,
};
use crate::chunk_plan::indexing::types::ValueRangePresent;
use crate::chunk_plan::prelude::*;
use crate::errors::BackendError;
use crate::{IStr, IntoIStr};

type LazyResult = Result<ExprPlan, BackendError>;

/// Compile a comparison to an ExprPlan with dimension constraint.
pub(super) fn compile_cmp_to_plan(
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
pub(super) fn compile_value_range_to_plan(
    col: &str,
    vr: &ValueRangePresent,
    ctx: &LazyCompileCtx<'_>,
) -> LazyResult {
    ensure_some!(ctx.dim_index(col));

    let constraint =
        LazyDimConstraint::Unresolved(vr.clone());
    let rect = LazyHyperRectangle::all()
        .with_dim(col.istr(), constraint);
    let sel =
        LazyArraySelection::from_rectangle(rect);

    Ok(ExprPlan::constrained(VarSet::All, sel))
}

/// Compile a struct field comparison to an ExprPlan.
pub(super) fn compile_struct_field_cmp(
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
