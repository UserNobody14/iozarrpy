use regex::Regex;

use super::compile_ctx::CompileCtx;
use super::errors::CompileError;
use crate::chunk_plan::prelude::*;
use super::SetOperations;
use crate::chunk_plan::indexing::selection::DatasetSelection;
use crate::chunk_plan::indexing::selection::dataset_all_for_vars;

pub(super) fn compile_selector(
    selector: &Selector,
    ctx: &mut CompileCtx<'_>,
) -> Result<DatasetSelection, CompileError> {
    match selector {
        Selector::Union(left, right) => {
            let left_node = compile_selector(
                left.as_ref(),
                ctx,
            )?;
            let right_node = compile_selector(
                right.as_ref(),
                ctx,
            )?;
            Ok(left_node.union(&right_node))
        }
        Selector::Difference(left, right) => {
            let left_node = compile_selector(
                left.as_ref(),
                ctx,
            )?;
            let right_node = compile_selector(
                right.as_ref(),
                ctx,
            )?;
            Ok(left_node.difference(&right_node))
        }
        Selector::ExclusiveOr(left, right) => {
            let left_node = compile_selector(
                left.as_ref(),
                ctx,
            )?;
            let right_node = compile_selector(
                right.as_ref(),
                ctx,
            )?;
            Ok(left_node.exclusive_or(&right_node))
        }
        Selector::Intersect(left, right) => {
            let left_node = compile_selector(
                left.as_ref(),
                ctx,
            )?;
            let right_node = compile_selector(
                right.as_ref(),
                ctx,
            )?;
            Ok(left_node.intersect(&right_node))
        }
        Selector::Empty => Ok(DatasetSelection::empty()),
        Selector::ByName { names, .. } => {
            let vars = names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            Ok(dataset_all_for_vars(vars))
        }
        Selector::Matches(pattern) => {
            // Filter data variables that match the regex pattern
            // Use meta.data_vars to exclude dimension coordinates
            let re = Regex::new(pattern.as_str()).map_err(|e| {
                CompileError::Unsupported(format!("invalid regex pattern '{}': {}", pattern, e))
            })?;
            let matching_vars: Vec<String> = ctx.meta.data_vars.iter()
                .filter(|v| re.is_match(v))
                .cloned()
                .collect();
            if matching_vars.is_empty() {
                Ok(DatasetSelection::empty())
            } else {
                Ok(dataset_all_for_vars(matching_vars))
            }
        }
        Selector::ByDType(dtype_selector) => {
            // Filter data variables by their Polars dtype from zarr array metadata.
            // Use meta.data_vars (not ctx.vars) to exclude dimension coordinates.
            // The DataTypeSelector has a matches() method that checks if a dtype matches.
            let matching_vars: Vec<String> = ctx.meta.data_vars.iter()
                .filter(|v| {
                    // Look up the array metadata for this variable
                    if let Some(array_meta) = ctx.meta.arrays.get(*v) {
                        dtype_selector.matches(&array_meta.polars_dtype)
                    } else {
                        // If we can't find the array, conservatively include it
                        true
                    }
                })
                .cloned()
                .collect();
            if matching_vars.is_empty() {
                Ok(DatasetSelection::empty())
            } else {
                Ok(dataset_all_for_vars(matching_vars))
            }
        }
        Selector::ByIndex { .. } => Ok(DatasetSelection::NoSelectionMade),
        Selector::Wildcard => {
            // Wildcard selects all data variables
            Ok(dataset_all_for_vars(ctx.meta.data_vars.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chunk_plan::exprs::errors::CoordIndexResolver;
    use crate::chunk_plan::indexing::types::{IndexRange, ValueRange};
    use crate::chunk_plan::exprs::errors::ResolveError;

    struct DummyResolver;
    impl CoordIndexResolver for DummyResolver {
        fn index_range_for_value_range(
            &mut self,
            _dim: &str,
            _range: &ValueRange,
        ) -> Result<Option<IndexRange>, ResolveError> {
            Ok(None)
        }
    }

}
