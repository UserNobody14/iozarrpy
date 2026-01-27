//! Helper functions for expression analysis.
//!
//! These utilities are used by the lazy compilation path to collect
//! column references from expressions.

use crate::chunk_plan::prelude::*;
use crate::{IStr, IntoIStr};

/// Collects all column names referenced in an expression using a simple recursive approach.
/// This is used for variable inference - determining which columns/variables are explicitly
/// referenced by an expression.
pub(crate) fn collect_column_refs(expr: &Expr, out: &mut Vec<IStr>) {
    match expr {
        Expr::Column(name) => out.push(name.istr()),
        Expr::Alias(inner, _)
        | Expr::KeepName(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::Sort { expr: inner, .. }
        | Expr::Explode { input: inner, .. }
        | Expr::Slice { input: inner, .. } => collect_column_refs(inner, out),
        Expr::RenameAlias { expr: inner, .. } => collect_column_refs(inner, out),
        Expr::SortBy { expr, by, .. } => {
            collect_column_refs(expr, out);
            for b in by {
                collect_column_refs(b, out);
            }
        }
        Expr::Over { function, partition_by, .. } => {
            collect_column_refs(function, out);
            for p in partition_by {
                collect_column_refs(p, out);
            }
        }
        Expr::Rolling { function, .. } => {
            collect_column_refs(function, out);
        }
        Expr::Filter { input, by } => {
            collect_column_refs(input, out);
            collect_column_refs(by, out);
        }
        Expr::BinaryExpr { left, right, .. } => {
            collect_column_refs(left, out);
            collect_column_refs(right, out);
        }
        Expr::Ternary { predicate, truthy, falsy } => {
            collect_column_refs(predicate, out);
            collect_column_refs(truthy, out);
            collect_column_refs(falsy, out);
        }
        Expr::Function { input, .. } | Expr::AnonymousFunction { input, .. } => {
            for i in input {
                collect_column_refs(i, out);
            }
        }
        Expr::Agg(agg) => {
            // AggExpr implements AsRef<Expr> to get the inner expression.
            use polars::prelude::AggExpr;
            let agg_ref: &AggExpr = agg;
            let inner: &Expr = agg_ref.as_ref();
            collect_column_refs(inner, out);
        }
        Expr::Gather { expr, idx, .. } => {
            collect_column_refs(expr, out);
            collect_column_refs(idx, out);
        }
        Expr::Selector(sel) => {
            collect_selector_refs(sel, out);
        }
        Expr::Eval { expr, .. } => collect_column_refs(expr, out),
        Expr::Field(names) => {
            for n in names.iter() {
                out.push(n.istr());
            }
        }
        // Literals, Len, etc. don't reference columns
        Expr::Literal(_) | Expr::Len | Expr::Element | Expr::SubPlan(_, _) 
        | Expr::DataTypeFunction(_) | Expr::StructEval { .. } => {}
    }
}

/// Collects column names from a selector expression.
pub(crate) fn collect_selector_refs(sel: &Selector, out: &mut Vec<IStr>) {
    match sel {
        Selector::ByName { names, .. } => {
            for n in names.iter() {
                out.push(n.istr());
            }
        }
        Selector::Union(a, b) | Selector::Intersect(a, b) | Selector::Difference(a, b) | Selector::ExclusiveOr(a, b) => {
            collect_selector_refs(&a.as_ref(), out);
            collect_selector_refs(&b.as_ref(), out);
        }
        // For Wildcard, ByDType, ByIndex, Matches - we can't determine columns statically
        _ => {}
    }
}

// =============================================================================
// Struct Field Path Extraction (for DataTree predicate pushdown)
// =============================================================================

/// Extract struct field path from an expression.
/// Returns (struct_column, field_name) or None if not a struct field access.
///
/// This handles expressions like:
/// - `pl.col("model_a").struct.field("temperature")`
/// - Which creates: `Expr::Function { input: [Column("model_a")], function: StructExpr(FieldByName(name))}`
pub(crate) fn extract_struct_field_path(expr: &Expr) -> Option<(IStr, IStr)> {
    match expr {
        Expr::Function { input, function, .. } => {
            // Check for struct field access function
            use polars::prelude::FunctionExpr;
            if let FunctionExpr::StructExpr(struct_fn) = function {
                use polars::prelude::StructFunction;
                match struct_fn {
                    StructFunction::FieldByName(field_name) => {
                        // Get the struct column from input
                        if let Some(Expr::Column(col_name)) = input.first() {
                            let col_name_str: &str = col_name.as_str();
                            let field_name_str: &str = field_name.as_str();
                            return Some((col_name_str.istr(), field_name_str.istr()));
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        // Handle nested expressions - unwrap wrappers
        Expr::Alias(inner, _) | Expr::KeepName(inner) | Expr::Cast { expr: inner, .. } => {
            extract_struct_field_path(inner)
        }
        _ => None,
    }
}

/// Collects column references including struct field paths.
/// Returns tuples of (struct_column, field_name) for struct field accesses.
pub(crate) fn collect_struct_field_refs(expr: &Expr, out: &mut Vec<(IStr, IStr)>) {
    if let Some(path) = extract_struct_field_path(expr) {
        out.push(path);
        return;
    }

    match expr {
        Expr::Alias(inner, _)
        | Expr::KeepName(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::Sort { expr: inner, .. } => collect_struct_field_refs(inner, out),
        Expr::BinaryExpr { left, right, .. } => {
            collect_struct_field_refs(left, out);
            collect_struct_field_refs(right, out);
        }
        Expr::Function { input, .. } | Expr::AnonymousFunction { input, .. } => {
            for i in input {
                collect_struct_field_refs(i, out);
            }
        }
        Expr::Ternary { predicate, truthy, falsy } => {
            collect_struct_field_refs(predicate, out);
            collect_struct_field_refs(truthy, out);
            collect_struct_field_refs(falsy, out);
        }
        _ => {}
    }
}
