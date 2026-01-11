use super::compile_cmp::compile_cmp_to_dataset_selection;
use super::compile_ctx::CompileCtx;
use super::errors::CompileError;
use super::expr_utils::expr_to_col_name;
use super::literals::strip_wrappers;
use super::prelude::*;
use super::selection::DatasetSelection;

pub(super) fn compile_is_between(
    input: &[Expr],
    ctx: &mut CompileCtx<'_>,
) -> Result<DatasetSelection, CompileError> {
    if input.len() < 3 {
        return Err(CompileError::Unsupported(format!(
            "unsupported is_between expression: {:?}",
            input
        )));
    };
    let expr = &input[0];
    let low = &input[1];
    let high = &input[2];
    let Some(col) = expr_to_col_name(expr) else {
        return Ok(ctx.all());
    };
    let Expr::Literal(low_lit) = strip_wrappers(low) else {
        return Ok(ctx.all());
    };
    let Expr::Literal(high_lit) = strip_wrappers(high) else {
        return Ok(ctx.all());
    };

    // Conservatively assume a closed interval (inclusive bounds) to avoid false negatives.
    let a = compile_cmp_to_dataset_selection(
        col,
        Operator::GtEq,
        low_lit,
        ctx,
    )
    .unwrap_or_else(|_| ctx.all());
    let b = compile_cmp_to_dataset_selection(
        col,
        Operator::LtEq,
        high_lit,
        ctx,
    )
    .unwrap_or_else(|_| ctx.all());
    Ok(a.intersect(&b))
}

