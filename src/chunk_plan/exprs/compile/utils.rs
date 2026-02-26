//! Small utilities for lazy expression compilation.

use super::super::compile_node::collect_column_refs;
use super::super::expr_plan::{ExprPlan, VarSet};
use crate::chunk_plan::prelude::*;
use crate::IStr;

pub(super) fn refs_to_plan(refs: Vec<IStr>) -> ExprPlan {
    if refs.is_empty() {
        ExprPlan::NoConstraint
    } else {
        ExprPlan::unconstrained_vars(VarSet::from_vec(refs))
    }
}

pub(super) fn refs_to_plan_with_vars(vars: VarSet) -> ExprPlan {
    if vars.is_empty() {
        ExprPlan::NoConstraint
    } else {
        ExprPlan::unconstrained_vars(vars)
    }
}

pub(super) fn collect_refs_from_expr(expr: &Expr) -> Vec<IStr> {
    let mut refs = Vec::new();
    collect_column_refs(expr, &mut refs);
    refs.sort();
    refs.dedup();
    refs
}
