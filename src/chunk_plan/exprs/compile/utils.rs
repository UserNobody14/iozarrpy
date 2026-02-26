//! Small utilities for lazy expression compilation.

use super::super::compile_node::collect_column_refs;
use super::super::expr_plan::{ExprPlan, VarSet};
use crate::IStr;
use crate::chunk_plan::prelude::*;

/// Wrap `let pat = expr` with an else that returns `NoConstraint` on mismatch.
/// Use any pattern (e.g. `Some(x)`, `Expr::Literal(lit)`) — no helper functions needed.
///
/// # Example
/// ```ignore
/// try_extract!(let Some(col) = expr_to_col_name(expr));
/// try_extract!(let Expr::Literal(low_lit) = strip_wrappers(low));
/// ```
#[macro_export]
macro_rules! try_extract {
    (let $pat:pat = $expr:expr) => {
        let $pat = $expr else {
            return Ok($crate::chunk_plan::exprs::expr_plan::ExprPlan::NoConstraint);
        };
    };
}

/// If the expression is `None`, return `NoConstraint`. Use when you need to ensure
/// an `Option` is `Some` but don't need the value.
///
/// # Example
/// ```ignore
/// ensure_some!(ctx.dim_index(col));
/// ```
#[macro_export]
macro_rules! ensure_some {
    ($expr:expr) => {
        if $expr.is_none() {
            return Ok($crate::chunk_plan::exprs::expr_plan::ExprPlan::NoConstraint);
        }
    };
}

pub(super) fn refs_to_plan(
    refs: Vec<IStr>,
) -> ExprPlan {
    if refs.is_empty() {
        ExprPlan::NoConstraint
    } else {
        ExprPlan::unconstrained_vars(
            VarSet::from_vec(refs),
        )
    }
}

pub(super) fn refs_to_plan_with_vars(
    vars: VarSet,
) -> ExprPlan {
    if vars.is_empty() {
        ExprPlan::NoConstraint
    } else {
        ExprPlan::unconstrained_vars(vars)
    }
}

pub(super) fn collect_refs_from_expr(
    expr: &Expr,
) -> Vec<IStr> {
    let mut refs = Vec::new();
    collect_column_refs(expr, &mut refs);
    refs.sort();
    refs.dedup();
    refs
}
