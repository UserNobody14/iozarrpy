use super::errors::CompileError;
use super::compile_ctx::CompileCtx;
use crate::chunk_plan::indexing::index_ranges;
use super::literals::{literal_to_scalar, reverse_operator, strip_wrappers};
use crate::chunk_plan::prelude::*;
use crate::chunk_plan::indexing::selection::{DataArraySelection, DatasetSelection, HyperRectangleSelection, RangeList};
use crate::chunk_plan::indexing::types::{BoundKind, ValueRange};
use crate::chunk_plan::indexing::selection::dataset_for_vars_with_selection;

pub(super) fn compile_value_range_to_dataset_selection(
    col: &str,
    vr: &ValueRange,
    ctx: &mut CompileCtx<'_>,
) -> Result<DatasetSelection, CompileError> {
    if vr.empty {
        return Ok(DatasetSelection::empty());
    }

    let dim_idx = ctx
        .dims
        .iter()
        .position(|d| d == col)
        .ok_or(CompileError::Unsupported(format!(
            "column '{}' not found in dimensions",
            col
        )))?;

    let idx_range = match ctx.resolver.index_range_for_value_range(col, vr) {
        Ok(Some(r)) => r,
        Ok(None) => {
            // If there's no 1D coordinate array for this dimension, treat it as a pure index dim.
            // This is common for grid dims like (y, x) where the user predicates on integer indices.
            if ctx.meta.arrays.get(col).is_none() {
                let dim_len = ctx
                    .dim_lengths
                    .get(dim_idx)
                    .copied()
                    .ok_or_else(|| CompileError::Unsupported("dimension length unavailable".to_owned()))?;
                index_ranges::index_range_for_index_dim(vr, dim_len).ok_or_else(|| {
                    CompileError::Unsupported("failed to plan index-only dimension".to_owned())
                })?
            } else {
                // We have a coord array, but it may be non-monotonic / non-1D. Don't constrain.
                return Ok(dataset_for_vars_with_selection(
                    ctx.vars.to_vec(),
                    DataArraySelection::all(),
                ));
            }
        }
        Err(e) => {
            return Err(CompileError::Unsupported(format!(
                "failed to get index range for value range: {e}"
            )));
        }
    };

    if idx_range.is_empty() {
        return Ok(DatasetSelection::empty());
    }

    let rect = HyperRectangleSelection::all().with_dim(col.to_string(), RangeList::from_index_range(idx_range));
    let sel = DataArraySelection(vec![rect]);
    Ok(dataset_for_vars_with_selection(
        ctx.vars.to_vec(),
        sel,
    ))
}

pub(super) fn compile_cmp_to_dataset_selection(
    col: &str,
    op: Operator,
    lit: &LiteralValue,
    ctx: &mut CompileCtx<'_>,
) -> Result<DatasetSelection, CompileError> {
    let time_encoding = ctx.meta.arrays.get(col).and_then(|a| a.time_encoding.as_ref());
    let Some(scalar) = literal_to_scalar(lit, time_encoding) else {
        return Err(CompileError::Unsupported(format!(
            "unsupported literal: {:?}",
            lit
        )));
    };

    let mut vr = ValueRange::default();
    match op {
        Operator::Eq => vr.eq = Some(scalar),
        Operator::Gt => vr.min = Some((scalar, BoundKind::Exclusive)),
        Operator::GtEq => vr.min = Some((scalar, BoundKind::Inclusive)),
        Operator::Lt => vr.max = Some((scalar, BoundKind::Exclusive)),
        Operator::LtEq => vr.max = Some((scalar, BoundKind::Inclusive)),
        _ => {
            return Err(CompileError::Unsupported(format!(
                "unsupported operator: {:?}",
                op
            )));
        }
    }

    compile_value_range_to_dataset_selection(
        col,
        &vr,
        ctx,
    )
}

pub(super) fn try_expr_to_value_range(
    expr: &Expr,
    meta: &ZarrDatasetMeta,
) -> Option<(String, ValueRange)> {
    let expr = strip_wrappers(expr);
    let Expr::BinaryExpr { left, op, right } = expr else {
        return None;
    };
    if !matches!(
        op,
        Operator::Eq | Operator::GtEq | Operator::Gt | Operator::LtEq | Operator::Lt
    ) {
        return None;
    }

    let (col, lit, op_eff) = if let (Expr::Column(name), Expr::Literal(lit)) =
        (strip_wrappers(left), strip_wrappers(right))
    {
        (name.to_string(), lit.clone(), *op)
    } else if let (Expr::Literal(lit), Expr::Column(name)) = (strip_wrappers(left), strip_wrappers(right)) {
        (name.to_string(), lit.clone(), reverse_operator(*op))
    } else {
        return None;
    };

    let time_encoding = meta.arrays.get(col.as_str()).and_then(|a| a.time_encoding.as_ref());
    let scalar = literal_to_scalar(&lit, time_encoding)?;

    let mut vr = ValueRange::default();
    match op_eff {
        Operator::Eq => vr.eq = Some(scalar),
        Operator::Gt => vr.min = Some((scalar, BoundKind::Exclusive)),
        Operator::GtEq => vr.min = Some((scalar, BoundKind::Inclusive)),
        Operator::Lt => vr.max = Some((scalar, BoundKind::Exclusive)),
        Operator::LtEq => vr.max = Some((scalar, BoundKind::Inclusive)),
        other => {
            debug_assert!(
                false,
                "unexpected operator after pre-check in try_expr_to_value_range: {other:?}"
            );
            return None;
        }
    }

    Some((col, vr))
}
