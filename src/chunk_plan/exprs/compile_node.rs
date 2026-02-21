//! Helper functions for expression analysis.
//!
//! These utilities are used by the lazy compilation path to collect
//! column references from expressions.

use super::expr_walk::{
    walk_expr, walk_selector,
};
use crate::chunk_plan::prelude::*;
use crate::{IStr, IntoIStr};

/// Collects all column names referenced in an expression using a simple recursive approach.
/// This is used for variable inference - determining which columns/variables are explicitly
/// referenced by an expression.
pub(crate) fn collect_column_refs(
    expr: &Expr,
    out: &mut Vec<IStr>,
) {
    walk_expr(expr, &mut |e| match e {
        Expr::AnonymousAgg { .. } => panic!(
            "AnonymousAgg is not supported"
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
/// Returns (struct_column, field_name) or None if not a struct field access.
///
/// This handles expressions like:
/// - `pl.col("model_a").struct.field("temperature")`
/// - Which creates: `Expr::Function { input: [Column("model_a")], function: StructExpr(FieldByName(name))}`
pub(crate) fn extract_struct_field_path(
    expr: &Expr,
) -> Option<(IStr, IStr)> {
    match expr {
        Expr::Function {
            input,
            function,
            ..
        } => {
            // Check for struct field access function
            use polars::prelude::FunctionExpr;
            if let FunctionExpr::StructExpr(
                struct_fn,
            ) = function
            {
                use polars::prelude::StructFunction;
                match struct_fn {
                    StructFunction::FieldByName(
                        field_name,
                    ) => {
                        // Get the struct column from input
                        if let Some(Expr::Column(
                            col_name,
                        )) = input.first()
                        {
                            let col_name_str: &str =
                                col_name.as_str();
                            let field_name_str: &str =
                                field_name.as_str();
                            return Some((
                                col_name_str.istr(),
                                field_name_str.istr(),
                            ));
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        // Handle nested expressions - unwrap wrappers
        Expr::Alias(inner, _)
        | Expr::KeepName(inner)
        | Expr::Cast { expr: inner, .. } => {
            extract_struct_field_path(inner)
        }
        _ => None,
    }
}
