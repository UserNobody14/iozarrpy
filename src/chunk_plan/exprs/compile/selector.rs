//! Selector expression compilation.

use super::super::compile_ctx::LazyCompileCtx;
use super::super::expr_plan::{ExprPlan, VarSet};
use crate::chunk_plan::prelude::*;
use crate::errors::BackendError;
use crate::{IStr, IntoIStr};
use snafu::ResultExt;

type LazyResult = Result<ExprPlan, BackendError>;

/// Compile selector to an ExprPlan.
pub(super) fn compile_selector_lazy(
    selector: &Selector,
    ctx: &mut LazyCompileCtx<'_>,
) -> LazyResult {
    use regex::Regex;

    match selector {
        Selector::Union(left, right) => {
            let l = compile_selector_lazy(
                left.as_ref(),
                ctx,
            )?;
            let r = compile_selector_lazy(
                right.as_ref(),
                ctx,
            )?;
            Ok(l.union(&r))
        }
        Selector::Difference(left, right) => {
            let l = compile_selector_lazy(
                left.as_ref(),
                ctx,
            )?;
            let r = compile_selector_lazy(
                right.as_ref(),
                ctx,
            )?;
            Ok(l.difference(&r))
        }
        Selector::ExclusiveOr(left, right) => {
            let l = compile_selector_lazy(
                left.as_ref(),
                ctx,
            )?;
            let r = compile_selector_lazy(
                right.as_ref(),
                ctx,
            )?;
            Ok(l.exclusive_or(&r))
        }
        Selector::Intersect(left, right) => {
            let l = compile_selector_lazy(
                left.as_ref(),
                ctx,
            )?;
            let r = compile_selector_lazy(
                right.as_ref(),
                ctx,
            )?;
            Ok(l.intersect(&r))
        }
        Selector::Empty => Ok(ExprPlan::Empty),
        Selector::ByName { names, .. } => {
            let vars: Vec<IStr> = names
                .iter()
                .map(|s| s.istr())
                .collect();
            Ok(ExprPlan::unconstrained_vars(
                VarSet::from_vec(vars),
            ))
        }
        Selector::Matches(pattern) => {
            let re = Regex::new(pattern.as_str())
                .context(crate::errors::backend::RegexSnafu {
                    pattern: pattern.clone(),
                })?;
            let matching_vars: Vec<IStr> = ctx
                .meta
                .all_array_paths()
                .iter()
                .filter(|v| {
                    re.is_match(v.as_ref())
                })
                .cloned()
                .collect();
            if matching_vars.is_empty() {
                Ok(ExprPlan::Empty)
            } else {
                Ok(ExprPlan::unconstrained_vars(
                    VarSet::from_vec(
                        matching_vars,
                    ),
                ))
            }
        }
        Selector::ByDType(dtype_selector) => {
            let matching_vars: Vec<IStr> = ctx
                .meta
                .all_array_paths()
                .iter()
                .filter(|v| {
                    if let Some(array_meta) = ctx
                        .meta
                        .array_by_path(v.istr())
                    {
                        dtype_selector.matches(
                            &array_meta
                                .polars_dtype,
                        )
                    } else {
                        true
                    }
                })
                .cloned()
                .collect();
            if matching_vars.is_empty() {
                Ok(ExprPlan::Empty)
            } else {
                Ok(ExprPlan::unconstrained_vars(
                    VarSet::from_vec(
                        matching_vars,
                    ),
                ))
            }
        }
        Selector::ByIndex { .. } => {
            Ok(ExprPlan::NoConstraint)
        }
        Selector::Wildcard => {
            let all_vars = ctx
                .meta
                .all_array_paths()
                .to_vec();
            Ok(ExprPlan::unconstrained_vars(
                VarSet::from_vec(all_vars),
            ))
        }
    }
}
