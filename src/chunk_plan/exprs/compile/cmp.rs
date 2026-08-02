//! Comparison and value range compilation.

use super::super::compile_ctx::LazyCompileCtx;
use super::super::expr_plan::{ExprPlan, VarSet};
use super::super::literals::literal_to_scalar;
use crate::chunk_plan::coord_resolve::Expansion;
use crate::chunk_plan::indexing::index_set::RectangleSet;
use crate::chunk_plan::indexing::types::ValueRangePresent;
use crate::chunk_plan::prelude::*;
use crate::ensure_some;
use crate::errors::BackendError;
use crate::meta::path::ZarrPath;
use crate::shared::{IStr, IntoIStr};

type LazyResult = Result<ExprPlan, BackendError>;

/// Compile a comparison to an ExprPlan with a resolved rectangle.
pub(super) fn compile_cmp_to_plan(
    col: &IStr,
    op: Operator,
    lit: &LiteralValue,
    ctx: &LazyCompileCtx<'_>,
) -> LazyResult {
    let scalar = literal_to_scalar(lit)?;
    let vr = ValueRangePresent::from_polars_op(
        op, scalar,
    )?;
    compile_value_range_to_plan(col, &vr, ctx)
}

/// Resolve a single-dim value range to a `RectangleSet` and wrap as an
/// `Active` plan.
pub(super) fn compile_value_range_to_plan(
    col: &str,
    vr: &ValueRangePresent,
    ctx: &LazyCompileCtx<'_>,
) -> LazyResult {
    let dim = col.istr();
    ensure_some!(ctx.dim_index(dim));
    let ranges =
        ctx.resolve(dim, vr, Expansion::Exact)?;
    let rects = RectangleSet::from_dim_constraint(
        ctx.universe.dims.clone(),
        ctx.universe.shape.clone(),
        dim,
        &ranges,
    );
    Ok(ExprPlan::active(VarSet::All, rects))
}

/// Compile a struct field comparison to an ExprPlan.
pub(super) fn compile_struct_field_cmp(
    struct_col: &IStr,
    field_path: &ZarrPath,
    op: Operator,
    lit: &LiteralValue,
    ctx: &LazyCompileCtx<'_>,
) -> LazyResult {
    let full_path = ZarrPath::single(*struct_col);
    let array_zp = field_path
        .components()
        .iter()
        .fold(full_path, |acc, c| acc.push(*c));
    let array_path = array_zp.istr();

    let arr_meta_opt =
        ctx.meta.array_by_path(array_path);
    if arr_meta_opt.is_none() {
        return Err(
            BackendError::StructFieldNotFound {
                path: array_path,
            },
        );
    }

    let scalar = literal_to_scalar(lit)?;
    let vr = ValueRangePresent::from_polars_op(
        op, scalar,
    )?;

    if let Some(arr_meta) = arr_meta_opt {
        let dims = arr_meta.dims();
        if dims.len() == 1 {
            let dim = dims[0];
            if ctx.dims().contains(&dim) {
                let ranges = ctx.resolve(
                    dim,
                    &vr,
                    Expansion::Exact,
                )?;
                let rects =
                    RectangleSet::from_dim_constraint(
                        ctx.universe.dims.clone(),
                        ctx.universe.shape.clone(),
                        dim,
                        &ranges,
                    );
                return Ok(ExprPlan::active(
                    VarSet::All,
                    rects,
                ));
            }
        }
    }

    Ok(ExprPlan::unconstrained_vars(
        VarSet::single(array_path),
    ))
}
