use super::compile_cmp::compile_cmp_to_dataset_selection;
use super::errors::{CompileError, CoordIndexResolver};
use super::expr_utils::expr_to_col_name;
use super::literals::strip_wrappers;
use super::prelude::*;
use super::selection::DatasetSelection;

pub(super) fn compile_is_between(
    input: &[Expr],
    meta: &ZarrDatasetMeta,
    dims: &[String],
    dim_lengths: &[u64],
    vars: &[String],
    resolver: &mut dyn CoordIndexResolver,
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
        return Ok(DatasetSelection::all_for_vars(vars.to_vec()));
    };
    let Expr::Literal(low_lit) = strip_wrappers(low) else {
        return Ok(DatasetSelection::all_for_vars(vars.to_vec()));
    };
    let Expr::Literal(high_lit) = strip_wrappers(high) else {
        return Ok(DatasetSelection::all_for_vars(vars.to_vec()));
    };

    // Conservatively assume a closed interval (inclusive bounds) to avoid false negatives.
    let a = compile_cmp_to_dataset_selection(
        col,
        Operator::GtEq,
        low_lit,
        meta,
        dims,
        dim_lengths,
        vars,
        resolver,
    )
    .unwrap_or_else(|_| DatasetSelection::all_for_vars(vars.to_vec()));
    let b = compile_cmp_to_dataset_selection(
        col,
        Operator::LtEq,
        high_lit,
        meta,
        dims,
        dim_lengths,
        vars,
        resolver,
    )
    .unwrap_or_else(|_| DatasetSelection::all_for_vars(vars.to_vec()));
    Ok(a.intersect(&b))
}

