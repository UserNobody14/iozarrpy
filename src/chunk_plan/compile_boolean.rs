use super::compile_node::compile_node;
use super::compile_is_between::compile_is_between;
use super::compile_is_in::compile_is_in;
use super::errors::{CompileError, CoordIndexResolver};
use super::literals::{literal_anyvalue, strip_wrappers};
use super::prelude::*;
use super::selection::DatasetSelection;

pub(super) fn compile_boolean_function(
    bf: &BooleanFunction,
    input: &[Expr],
    meta: &ZarrDatasetMeta,
    dims: &[String],
    dim_lengths: &[u64],
    vars: &[String],
    resolver: &mut dyn CoordIndexResolver,
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
                    Some(AnyValue::Boolean(false)) => Ok(DatasetSelection::all_for_vars(vars.to_vec())),
                    Some(AnyValue::Null) => Ok(DatasetSelection::empty()),
                    _ => Ok(DatasetSelection::all_for_vars(vars.to_vec())),
                };
            }

            // If the inner predicate is known to match nothing, NOT(...) matches everything.
            // Otherwise we can't represent complements with current plan nodes.
            let inner =
                compile_node(arg, meta, dims, dim_lengths, vars, resolver)
                    .unwrap_or_else(|_| DatasetSelection::all_for_vars(vars.to_vec()));
            if inner.0.is_empty() {
                Ok(DatasetSelection::all_for_vars(vars.to_vec()))
            } else {
                Ok(DatasetSelection::all_for_vars(vars.to_vec()))
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
                    DatasetSelection::all_for_vars(vars.to_vec())
                } else {
                    DatasetSelection::empty()
                });
            }
            Ok(DatasetSelection::all_for_vars(vars.to_vec()))
        }
        _ => {
            // Future-proof handling for optional Polars boolean features without hard-referencing
            // cfg-gated variants (e.g. `is_in`, `is_between`).
            let name = bf.to_string();
            match name.as_str() {
                "is_between" => {
                    compile_is_between(input, meta, dims, dim_lengths, vars, resolver)
                }
                "is_in" => {
                    compile_is_in(input, meta, dims, dim_lengths, vars, resolver)
                }
                _ => Ok(DatasetSelection::all_for_vars(vars.to_vec())),
            }
        }
    }
}

