use super::compile_cmp::compile_cmp_to_dataset_selection;
use super::compile_ctx::CompileCtx;
use super::errors::CompileError;
use super::expr_utils::expr_to_col_name;
use super::literals::strip_wrappers;
use crate::chunk_plan::prelude::*;
use crate::chunk_plan::indexing::selection::DatasetSelection;

pub(super) fn compile_is_in(
    input: &[Expr],
    ctx: &mut CompileCtx<'_>,
) -> Result<DatasetSelection, CompileError> {
    if input.len() < 2 {
        return Err(CompileError::Unsupported(format!(
            "unsupported is_in expression: {:?}",
            input
        )));
    };
    let expr = &input[0];
    let list = &input[1];
    let Some(col) = expr_to_col_name(expr) else {
        return Ok(ctx.all());
    };

    let Expr::Literal(list_lit) = strip_wrappers(list) else {
        return Ok(ctx.all());
    };

    let (dtype, values): (polars::prelude::DataType, Vec<AnyValue<'static>>) =
        match list_lit {
            LiteralValue::Series(s) => {
                let series = &**s;
                if series.len() > 4096 {
                    return Ok(ctx.all());
                }
                (
                    series.dtype().clone(),
                    series.iter().map(|av| av.into_static()).collect(),
                )
            }
            LiteralValue::Scalar(s) => {
                let av = s.clone().into_value().into_static();
                match av {
                    AnyValue::List(series) => {
                        if series.len() > 4096 {
                            return Ok(ctx.all());
                        }
                        (
                            series.dtype().clone(),
                            series.iter().map(|av| av.into_static()).collect(),
                        )
                    }
                    AnyValue::Array(series, _size) => {
                        if series.len() > 4096 {
                            return Ok(ctx.all());
                        }
                        (
                            series.dtype().clone(),
                            series.iter().map(|av| av.into_static()).collect(),
                        )
                    }
                    _ => return Ok(ctx.all()),
                }
            }
            _ => return Ok(ctx.all()),
        };

    let mut out: Option<DatasetSelection> = None;
    for av in values {
        if matches!(av, AnyValue::Null) {
            // Null membership semantics depend on `nulls_equal`; we avoid constraining.
            return Ok(ctx.all());
        }

        let lit = LiteralValue::Scalar(Scalar::new(dtype.clone(), av));
        let node = compile_cmp_to_dataset_selection(
            col,
            Operator::Eq,
            &lit,
            ctx,
        )
        .unwrap_or_else(|_| ctx.all());

        if node == ctx.all() {
            return Ok(ctx.all());
        }
        out = Some(match out.take() {
            None => node,
            Some(acc) => acc.union(&node),
        });
    }
    Ok(out.unwrap_or_else(DatasetSelection::empty))
}

