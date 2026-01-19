use super::SetOperations;
use super::compile_cmp::compile_cmp_to_dataset_selection;
use super::compile_ctx::CompileCtx;
use super::errors::CompileError;
use super::expr_utils::expr_to_col_name;
use super::literals::strip_wrappers;
use crate::chunk_plan::indexing::selection::DatasetSelection;
use crate::chunk_plan::prelude::*;
use polars::prelude::Scalar;

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
        return Ok(DatasetSelection::NoSelectionMade);
    };

    let Expr::Literal(list_lit) = strip_wrappers(list) else {
        return Ok(DatasetSelection::NoSelectionMade);
    };

    let (dtype, values): (&polars::prelude::DataType, Vec<AnyValue<'static>>) = match list_lit {
        LiteralValue::Series(s) => {
            let series = &**s;
            if series.len() > 4096 {
                return Ok(DatasetSelection::NoSelectionMade);
            }
            (
                series.dtype(),
                series.iter().map(|av| av.into_static()).collect(),
            )
        }
        LiteralValue::Scalar(s) => {
            let ssv = s.clone();
            let av = ssv.into_value();
            match av {
                AnyValue::List(series) => {
                    if series.len() > 4096 {
                        return Ok(DatasetSelection::NoSelectionMade);
                    }
                    (
                        &series.dtype().clone(),
                        series.iter().map(|av| av.into_static()).collect(),
                    )
                }
                AnyValue::Array(series, _size) => {
                    if series.len() > 4096 {
                        return Ok(DatasetSelection::NoSelectionMade);
                    }
                    (
                        &series.dtype().clone(),
                        series.iter().map(|av| av.into_static()).collect(),
                    )
                }
                AnyValue::Int8(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Int16(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Int32(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Int64(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::UInt8(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::UInt16(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::UInt32(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::UInt64(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Float16(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Float32(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Float64(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Int128(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::UInt128(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::DatetimeOwned(_, _, _) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Categorical(_, _) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::CategoricalOwned(_, _) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Time(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Duration(_, _) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Struct(_, _, _) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::StructOwned(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Null => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Boolean(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::String(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Binary(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Date(_) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Datetime(_, _, _) => return Ok(DatasetSelection::NoSelectionMade),
                AnyValue::Enum(_, _categorical_mapping) => todo!(),
                AnyValue::EnumOwned(_, _categorical_mapping) => todo!(),
                AnyValue::Object(_polars_object_safe) => todo!(),
                AnyValue::ObjectOwned(_owned_object) => todo!(),
                AnyValue::StringOwned(_pl_small_str) => todo!(),
                AnyValue::BinaryOwned(_items) => todo!(),
                AnyValue::Decimal(_, _, _) => todo!(),
            }
        }
        _ => return Ok(DatasetSelection::NoSelectionMade),
    };

    let mut out: Option<DatasetSelection> = None;
    for av in values {
        if matches!(av, AnyValue::Null) {
            // Null membership semantics depend on `nulls_equal`; we avoid constraining.
            return Ok(DatasetSelection::NoSelectionMade);
        }

        let lit = LiteralValue::Scalar(Scalar::new(dtype.clone(), av));
        let node = compile_cmp_to_dataset_selection(col, Operator::Eq, &lit, ctx)
            .unwrap_or_else(|_| DatasetSelection::NoSelectionMade);

        if node == DatasetSelection::NoSelectionMade {
            return Ok(DatasetSelection::NoSelectionMade);
        }
        out = Some(match out.take() {
            None => node,
            Some(acc) => acc.union(&node),
        });
    }
    Ok(out.unwrap_or_else(DatasetSelection::empty))
}
