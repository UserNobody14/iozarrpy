//! Generic expression tree walking.
//!
//! Provides a single traversal over `Expr` (and `Selector`) trees so that
//! different passes can apply mutable updates (e.g. collect column refs,
//! visit every node) without duplicating the recursion structure.

use super::literals::strip_wrappers;
use crate::chunk_plan::prelude::*;

/// Walks an expression tree in pre-order and calls `visitor` on each node.
///
/// The visitor receives a mutable reference (e.g. `&mut impl FnMut(&Expr)`) so it can
/// accumulate state (e.g. push column names to a `Vec`). Recursion is driven by
/// the structure of the expression after stripping alias/cast wrappers so every
/// sub-expression is visited exactly once.
pub fn walk_expr(
    expr: &Expr,
    visitor: &mut impl FnMut(&Expr),
) {
    visitor(expr);
    let expr = strip_wrappers(expr);
    match expr {
        Expr::Display { .. } => {
            panic!(
                "Display expression not supported"
            );
        }
        Expr::Alias(inner, _)
        | Expr::KeepName(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::Sort { expr: inner, .. }
        | Expr::Explode {
            input: inner, ..
        }
        | Expr::Slice { input: inner, .. }
        | Expr::RenameAlias {
            expr: inner, ..
        } => {
            walk_expr(inner.as_ref(), visitor);
        }
        Expr::SortBy { expr, by, .. } => {
            walk_expr(expr.as_ref(), visitor);
            for b in by {
                walk_expr(b, visitor);
            }
        }
        Expr::Over {
            function,
            partition_by,
            ..
        } => {
            walk_expr(function.as_ref(), visitor);
            for p in partition_by {
                walk_expr(p, visitor);
            }
        }
        Expr::Rolling { function, .. } => {
            walk_expr(function.as_ref(), visitor);
        }
        Expr::Filter { input, by } => {
            walk_expr(input.as_ref(), visitor);
            walk_expr(by.as_ref(), visitor);
        }
        Expr::BinaryExpr {
            left, right, ..
        } => {
            walk_expr(left.as_ref(), visitor);
            walk_expr(right.as_ref(), visitor);
        }
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            walk_expr(
                predicate.as_ref(),
                visitor,
            );
            walk_expr(truthy.as_ref(), visitor);
            walk_expr(falsy.as_ref(), visitor);
        }
        Expr::Function { input, .. }
        | Expr::AnonymousFunction {
            input, ..
        } => {
            for i in input {
                walk_expr(i, visitor);
            }
        }
        Expr::Agg(agg) => {
            let inner: &Expr = agg.as_ref();
            walk_expr(inner, visitor);
        }
        Expr::Gather { expr, idx, .. } => {
            walk_expr(expr.as_ref(), visitor);
            walk_expr(idx.as_ref(), visitor);
        }
        Expr::Selector(_) => {
            // No Expr children; visitor can call walk_selector if needed.
        }
        Expr::Eval { expr, .. } => {
            walk_expr(expr.as_ref(), visitor);
        }
        Expr::Column(_)
        | Expr::Literal(_)
        | Expr::Field(_)
        | Expr::Len
        | Expr::Element
        | Expr::SubPlan(_, _)
        | Expr::DataTypeFunction(_)
        | Expr::StructEval { .. } => {}
    }
}

/// Walks a selector tree in pre-order and calls `visitor` on each node.
pub fn walk_selector(
    sel: &Selector,
    visitor: &mut impl FnMut(&Selector),
) {
    visitor(sel);
    match sel {
        Selector::Union(a, b)
        | Selector::Intersect(a, b)
        | Selector::Difference(a, b)
        | Selector::ExclusiveOr(a, b) => {
            walk_selector(a.as_ref(), visitor);
            walk_selector(b.as_ref(), visitor);
        }
        _ => {}
    }
}
