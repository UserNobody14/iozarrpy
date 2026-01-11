use super::compile_cmp::compile_cmp_to_dataset_selection;
use super::errors::{CompileError, CoordIndexResolver};
use super::expr_utils::expr_to_col_name;
use super::literals::strip_wrappers;
use super::prelude::*;
use super::selection::DatasetSelection;

pub(super) fn compile_is_in(
    input: &[Expr],
    meta: &ZarrDatasetMeta,
    dims: &[String],
    dim_lengths: &[u64],
    vars: &[String],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<DatasetSelection, CompileError> {
    let [expr, list] = input else {
        return Err(CompileError::Unsupported(format!(
            "unsupported is_in expression: {:?}",
            input
        )));
    };
    let Some(col) = expr_to_col_name(expr) else {
        return Ok(DatasetSelection::all_for_vars(vars.to_vec()));
    };

    let Expr::Literal(list_lit) = strip_wrappers(list) else {
        return Ok(DatasetSelection::all_for_vars(vars.to_vec()));
    };

    match list_lit {
        LiteralValue::Series(s) => {
            let series = &**s;
            // Prevent pathological unions for huge lists.
            if series.len() > 4096 {
                return Ok(DatasetSelection::all_for_vars(vars.to_vec()));
            }

            let mut out: Option<DatasetSelection> = None;
            for av in series.iter() {
                let av = av.into_static();
                if matches!(av, AnyValue::Null) {
                    // Null membership semantics depend on `nulls_equal`; we avoid constraining.
                    return Ok(DatasetSelection::all_for_vars(vars.to_vec()));
                }

                let lit = LiteralValue::Scalar(Scalar::new(series.dtype().clone(), av));
                let node = compile_cmp_to_dataset_selection(
                    col,
                    Operator::Eq,
                    &lit,
                    meta,
                    dims,
                    dim_lengths,
                    vars,
                    resolver,
                )
                .unwrap_or_else(|_| DatasetSelection::all_for_vars(vars.to_vec()));

                // If any element falls back to AllChunks, the whole IN predicate becomes unconstrainable.
                // We model AllChunks as “all vars present and unconstrained”. If we ever hit that,
                // stop trying to refine.
                if node == DatasetSelection::all_for_vars(vars.to_vec()) {
                    return Ok(DatasetSelection::all_for_vars(vars.to_vec()));
                }
                out = Some(match out.take() {
                    None => node,
                    Some(acc) => acc.union(&node),
                });
            }
            Ok(out.unwrap_or_else(DatasetSelection::empty))
        }
        _ => Ok(DatasetSelection::all_for_vars(vars.to_vec())),
    }
}

