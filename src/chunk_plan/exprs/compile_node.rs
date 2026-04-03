//! Helper functions for expression analysis.
//!
//! These utilities are used by the lazy compilation path to collect
//! column references from expressions.

use super::expr_walk::{
    walk_expr, walk_selector,
};
use crate::chunk_plan::prelude::*;
use crate::meta::path::ZarrPath;
use crate::{IStr, IntoIStr};

/// Collects all column names referenced in an expression using a simple recursive approach.
/// This is used for variable inference - determining which columns/variables are explicitly
/// referenced by an expression.
pub(crate) fn collect_column_refs(
    expr: &Expr,
    out: &mut Vec<IStr>,
) {
    walk_expr(expr, &mut |e| match e {
        Expr::Display { .. } => panic!(
            "Display expression not supported"
        ),
        Expr::Column(name) => {
            out.push(name.istr())
        }
        Expr::Field(names) => {
            for n in names.iter() {
                out.push(n.istr());
            }
        }
        Expr::Selector(sel) => {
            collect_selector_refs(sel, out)
        }
        _ => {}
    });
}

/// Collects column names from a selector expression.
pub(crate) fn collect_selector_refs(
    sel: &Selector,
    out: &mut Vec<IStr>,
) {
    walk_selector(sel, &mut |s| {
        if let Selector::ByName {
            names, ..
        } = s
        {
            for n in names.iter() {
                out.push(n.istr());
            }
        }
    });
}

// =============================================================================
// Struct Field Path Extraction (for DataTree predicate pushdown)
// =============================================================================

/// Extract struct field path from an expression.
/// Returns `(root_column, ZarrPath)` or None if not a struct field access.
///
/// Handles arbitrarily nested struct field access, e.g.:
/// - `col("a").struct.field("b")` → `("a", ZarrPath["b"])`
/// - `col("a").struct.field("b").struct.field("c")` → `("a", ZarrPath["b", "c"])`
pub(crate) fn extract_struct_field_path(
    expr: &Expr,
) -> Option<(IStr, ZarrPath)> {
    use polars::prelude::{
        FunctionExpr, StructFunction,
    };
    match expr {
        Expr::Function {
            input,
            function:
                FunctionExpr::StructExpr(
                    StructFunction::FieldByName(
                        field_name,
                    ),
                ),
            ..
        } => {
            let inner = input.first()?;
            if let Expr::Column(col_name) = inner
            {
                Some((
                    col_name.as_str().istr(),
                    ZarrPath::single(
                        field_name
                            .as_str()
                            .istr(),
                    ),
                ))
            } else if let Some((
                root,
                parent_path,
            )) =
                extract_struct_field_path(inner)
            {
                Some((
                    root,
                    parent_path.push(
                        field_name
                            .as_str()
                            .istr(),
                    ),
                ))
            } else {
                None
            }
        }
        Expr::Alias(inner, _)
        | Expr::KeepName(inner)
        | Expr::Cast { expr: inner, .. } => {
            extract_struct_field_path(inner)
        }
        _ => None,
    }
}
