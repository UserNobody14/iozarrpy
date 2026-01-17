use super::compile_node::compile_node;
use super::compile_is_between::compile_is_between;
use super::compile_is_in::compile_is_in;
use super::compile_ctx::CompileCtx;
use super::errors::CompileError;
use super::literals::{literal_anyvalue, strip_wrappers};
use crate::chunk_plan::prelude::*;
use crate::chunk_plan::indexing::selection::DatasetSelection;

pub(super) fn compile_boolean_function(
    bf: &BooleanFunction,
    input: &[Expr],
    ctx: &mut CompileCtx<'_>,
) -> Result<DatasetSelection, CompileError> {
    match bf {
        BooleanFunction::Not => {
            let [arg] = input else {
                return Err(CompileError::Unsupported(format!(
                    "unsupported boolean function: {:?}",
                    bf
                )));
            };
            // Try constant fold first.
            if let Expr::Literal(lit) = strip_wrappers(arg) {
                return match literal_anyvalue(lit) {
                    Some(AnyValue::Boolean(true)) => Ok(DatasetSelection::empty()),
                    Some(AnyValue::Boolean(false)) => Ok(ctx.all()),
                    Some(AnyValue::Null) => Ok(DatasetSelection::empty()),
                    _ => Ok(ctx.all()),
                };
            }

            // If the inner predicate is known to match nothing, NOT(...) matches everything.
            // Otherwise we can't represent complements with current plan nodes.
            let inner =
                compile_node(arg, ctx)
                    .unwrap_or_else(|_| ctx.all());
            if inner.0.is_empty() {
                Ok(ctx.all())
            } else {
                Ok(ctx.all())
            }
        }
        BooleanFunction::IsNull | BooleanFunction::IsNotNull => {
            let [arg] = input else {
                return Err(CompileError::Unsupported(format!(
                    "unsupported boolean function: {:?}",
                    bf
                )));
            };

            // Constant fold when possible; otherwise don't constrain.
            if let Expr::Literal(lit) = strip_wrappers(arg) {
                let is_null = matches!(literal_anyvalue(lit), Some(AnyValue::Null));
                let keep = match bf {
                    BooleanFunction::IsNull => is_null,
                    BooleanFunction::IsNotNull => !is_null,
                    _ => unreachable!(),
                };
                return Ok(if keep {
                    ctx.all()
                } else {
                    DatasetSelection::empty()
                });
            }
            Ok(ctx.all())
        }
        BooleanFunction::IsBetween { .. } => {
            compile_is_between(input, ctx)
        }
        BooleanFunction::IsIn { .. } => compile_is_in(input, ctx),
        BooleanFunction::AnyHorizontal => {
            // OR across all input expressions.
            let mut acc = DatasetSelection::empty();
            for e in input {
                let sel = compile_node(e, ctx)
                    .unwrap_or_else(|_| ctx.all());
                // If any side is unconstrainable, OR becomes unconstrainable.
                if sel == ctx.all() {
                    return Ok(ctx.all());
                }
                acc = acc.union(&sel);
            }
            Ok(acc)
        }
        BooleanFunction::AllHorizontal => {
            // AND across all input expressions.
            let mut acc = ctx.all();
            for e in input {
                let sel = compile_node(e, ctx)
                    .unwrap_or_else(|_| ctx.all());
                acc = acc.intersect(&sel);
                if acc.0.is_empty() {
                    break;
                }
            }
            Ok(acc)
        }
        _ => {
            Ok(ctx.all())
        }
    }
}

