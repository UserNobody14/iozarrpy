
use super::compile_boolean::compile_boolean_function;
use super::compile_cmp::{
    compile_cmp_to_dataset_selection, compile_value_range_to_dataset_selection, try_expr_to_value_range,
};
use super::interpolate_selection_nd::interpolate_selection_nd;
use super::compile_ctx::CompileCtx;
use super::errors::CompileError;
use super::literals::{col_lit, literal_anyvalue, reverse_operator, strip_wrappers};
use crate::chunk_plan::prelude::*;
use crate::chunk_plan::indexing::selection::DatasetSelection;
use super::selector::compile_selector;

/// Collects all column names referenced in an expression using a simple recursive approach.
fn collect_column_refs(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Column(name) => out.push(name.to_string()),
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
            // Arc<T>: Deref<Target=T> so we can get &AggExpr via implicit deref.
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
                out.push(n.to_string());
            }
        }
        // Literals, Len, etc. don't reference columns
        Expr::Literal(_) | Expr::Len | Expr::Element | Expr::SubPlan(_, _) 
        | Expr::DataTypeFunction(_) | Expr::StructEval { .. } => {}
    }
}

/// Collects column names from a selector expression.
fn collect_selector_refs(sel: &Selector, out: &mut Vec<String>) {
    match sel {
        Selector::ByName { names, .. } => {
            for n in names.iter() {
                out.push(n.to_string());
            }
        }
        Selector::Union(a, b) | Selector::Intersect(a, b) | Selector::Difference(a, b) | Selector::ExclusiveOr(a, b) => {
            collect_column_refs(&a.as_ref().clone().as_expr(), out);
            collect_column_refs(&b.as_ref().clone().as_expr(), out);
        }
        // For Wildcard, ByDType, ByIndex, Matches - we can't determine columns statically
        _ => {}
    }
}

/// Returns a DatasetSelection for the variables referenced in an expression, with all chunks.
fn all_for_referenced_vars(expr: &Expr, ctx: &CompileCtx<'_>) -> DatasetSelection {
    let mut refs = Vec::new();
    collect_column_refs(expr, &mut refs);
    refs.sort();
    refs.dedup();
    
    if refs.is_empty() {
        ctx.all()
    } else {
        DatasetSelection::all_for_vars(refs)
    }
}

pub(crate) fn compile_node(
    // Either a borrowed or owned expression.
    expr: impl std::borrow::Borrow<Expr>,
    ctx: &mut CompileCtx<'_>,
) -> Result<DatasetSelection, CompileError> {
    let expr: &Expr = std::borrow::Borrow::borrow(&expr);
    match expr {
        Expr::Alias(inner, _) => compile_node(
            inner.as_ref(),
            ctx,
        ),
        Expr::KeepName(inner) => compile_node(
            inner.as_ref(),
            ctx,
        ),
        Expr::RenameAlias { expr, .. } => compile_node(
            expr.as_ref(),
            ctx,
        ),
        Expr::Cast { expr, .. } => compile_node(
            expr.as_ref(),
            ctx,
        ),
        Expr::Sort { expr, .. } => compile_node(
            expr.as_ref(),
            ctx,
        ),
        Expr::SortBy { expr, .. } => compile_node(
            expr.as_ref(),
            ctx,
        ),
        Expr::Explode { input, .. } => compile_node(
            input.as_ref(),
            ctx,
        ),
        Expr::Slice { input, .. } => compile_node(
            input.as_ref(),
            ctx,
        ),
        // Window expressions: track both the function and partition_by columns.
        Expr::Over { function, partition_by, .. } => {
            // Get variables from the function expression
            let func_sel = compile_node(function.as_ref(), ctx)?;
            // Collect variables from partition_by columns
            let mut refs = Vec::new();
            for p in partition_by {
                collect_column_refs(p, &mut refs);
            }
            refs.sort();
            refs.dedup();
            
            // Union the function vars with partition_by vars
            if refs.is_empty() {
                Ok(func_sel)
            } else {
                let part_sel = DatasetSelection::all_for_vars(refs);
                Ok(func_sel.union(&part_sel))
            }
        }
        Expr::Rolling { function, .. } => compile_node(
            function.as_ref(),
            ctx,
        ),
        // Filter expression: need both the input data variable and the filter predicate.
        Expr::Filter { input, by } => {
            // Get variables from the input expression
            let input_sel = compile_node(input.as_ref(), ctx)?;
            // Get variables from the filter predicate
            let filter_sel = compile_node(by.as_ref(), ctx)?;
            // Union both - we need the input data and all vars referenced in the filter
            Ok(input_sel.union(&filter_sel))
        }
        Expr::BinaryExpr { left, op, right } => {
            match op {
                Operator::And | Operator::LogicalAnd => {
                    // Special-case: A & !B => A \ B (can cut holes).
                    if let Expr::Function { input, function } = super::literals::strip_wrappers(right.as_ref()) {
                        if matches!(function, FunctionExpr::Boolean(BooleanFunction::Not)) && input.len() == 1 {
                            let a = compile_node(
                                left.as_ref(),
                                ctx,
                            )?;
                            let b = compile_node(
                                input[0].clone(),
                                ctx,
                            )?;
                            return Ok(a.difference(&b));
                        }
                    }
                    if let Expr::Function { input, function } = super::literals::strip_wrappers(left.as_ref()) {
                        if matches!(function, FunctionExpr::Boolean(BooleanFunction::Not)) && input.len() == 1 {
                            let a = compile_node(
                                right.as_ref(),
                                ctx,
                            )?;
                            let b = compile_node(
                                input[0].clone(),
                                ctx,
                            )?;
                            return Ok(a.difference(&b));
                        }
                    }

                    // Fast path: merge compatible comparisons on the same column into a single
                    // ValueRange. This reduces resolver reads and enables tighter planning.
                    if let (Some((col_a, vr_a)), Some((col_b, vr_b))) = (
                        try_expr_to_value_range(left.as_ref(), ctx.meta),
                        try_expr_to_value_range(right.as_ref(), ctx.meta),
                    ) {
                        if col_a == col_b {
                            let vr = vr_a.intersect(&vr_b);
                            return compile_value_range_to_dataset_selection(
                                &col_a, &vr, ctx,
                            );
                        }
                    }

                    // If one side is unsupported, keep whatever constraints we can from the other.
                    let a = compile_node(
                        left.as_ref(),
                        ctx,
                    )?;
                    let b = compile_node(
                        right.as_ref(),
                        ctx,
                    )?;
                    Ok(a.intersect(&b))
                }
                Operator::Or | Operator::LogicalOr => {
                    let a = compile_node(
                        left.as_ref(),
                        ctx,
                    )?;
                    let b = compile_node(
                        right.as_ref(),
                        ctx,
                    )?;
                    Ok(a.union(&b))
                }
                Operator::Xor => {
                    let a = compile_node(
                        left.as_ref(),
                        ctx,
                    )?;
                    let b = compile_node(
                        right.as_ref(),
                        ctx,
                    )?;
                    Ok(a.difference(&b).union(&b.difference(&a)))
                }
                Operator::Eq | Operator::GtEq | Operator::Gt | Operator::LtEq | Operator::Lt => {
                    if let Some((col, lit)) = col_lit(left, right).or_else(|| col_lit(right, left))
                    {
                        let op_eff = if matches!(strip_wrappers(left.as_ref()), Expr::Literal(_)) {
                            reverse_operator(*op)
                        } else {
                            *op
                        };
                        // Try to compile as dimension constraint; if not a dimension, return
                        // the referenced variable with all chunks (can't narrow).
                        compile_cmp_to_dataset_selection(&col, op_eff, &lit, ctx)
                            .or_else(|_| Ok(DatasetSelection::all_for_vars(vec![col])))
                    } else {
                        // Complex comparison (e.g., col1 < col2) - return referenced vars with all chunks.
                        Ok(all_for_referenced_vars(expr, ctx))
                    }
                }
                // Arithmetic/other operators: return referenced vars with all chunks.
                _ => Ok(all_for_referenced_vars(expr, ctx)),
            }
        }
        Expr::Literal(lit) => {
            // Only boolean-ish literals can be predicates.
            match literal_anyvalue(lit) {
                Some(AnyValue::Boolean(true)) => Ok(ctx.all()),
                Some(AnyValue::Boolean(false)) => Ok(DatasetSelection::empty()),
                // In Polars filtering, null predicate behaves like "keep nothing".
                Some(AnyValue::Null) => Ok(DatasetSelection::empty()),
                // Non-boolean literals don't reference any variables, return all conservatively.
                _ => Ok(ctx.all()),
            }
        }
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            // Fast paths for common boolean ternaries produced by `when/then/otherwise`.
            // These show up frequently and we can often preserve pushdown.
            if let (Expr::Literal(t), Expr::Literal(f)) = (
                super::literals::strip_wrappers(truthy.as_ref()),
                super::literals::strip_wrappers(falsy.as_ref()),
            ) {
                if let (Some(AnyValue::Boolean(t)), Some(AnyValue::Boolean(f))) =
                    (literal_anyvalue(t), literal_anyvalue(f))
                {
                    if t && !f {
                        // when(predicate).then(true).otherwise(false) == predicate
                        return compile_node(
                            predicate.as_ref(),
                            ctx,
                        );
                    }
                    if !t && f {
                        // when(predicate).then(false).otherwise(true) == !predicate, which we
                        // can't represent precisely: return conservative All.
                        return Ok(ctx.all());
                    }
                    if t && f {
                        return Ok(ctx.all());
                    }
                    if !t && !f {
                        return Ok(DatasetSelection::empty());
                    }
                }
            }

            let predicate_node = compile_node(
                predicate.as_ref(),
                ctx,
            )?;
            let truthy_node = compile_node(
                truthy.as_ref(),
                ctx,
            )?;
            let falsy_node = compile_node(
                falsy.as_ref(),
                ctx,
            )?;
            Ok(truthy_node.union(&falsy_node).union(&predicate_node))
        }
        Expr::Function { input, function } => {
            match function {
                FunctionExpr::Boolean(bf) => compile_boolean_function(
                    bf,
                    input,
                    ctx,
                ),
                FunctionExpr::NullCount => Ok(ctx.all()),
                FunctionExpr::FfiPlugin {
                    symbol,
                    ..
                } => {
                    println!("symbol: {:?}", symbol);
                    // If the symbol is "interpolate_nd", perform the proper interpolation.
                    if symbol == "interpolate_nd" {
                        if input.len() < 3 {
                            return Ok(ctx.all());
                        }
                        interpolate_selection_nd(&input[0], &input[1], &input[2], ctx)
                    } else {
                        // Unknown FFI plugin - return referenced vars with all chunks
                        Ok(all_for_referenced_vars(expr, ctx))
                    }
                },
                // Most functions transform values in ways that we can't safely map to chunk-level constraints.
                // Return the referenced variables with all chunks (conservative but correct for variable inference).
                _ => {
                    // Collect variables from input expressions
                    let mut refs = Vec::new();
                    for i in input {
                        collect_column_refs(i, &mut refs);
                    }
                    refs.sort();
                    refs.dedup();
                    if refs.is_empty() {
                        Ok(ctx.all())
                    } else {
                        Ok(DatasetSelection::all_for_vars(refs))
                    }
                }
            }
        }
        Expr::Selector(selector) => {
            compile_selector(selector, ctx)
        }

        // Column reference: return just that variable with all chunks.
        Expr::Column(name) => {
            Ok(DatasetSelection::all_for_vars(vec![name.to_string()]))
        }

        // Aggregations: collect referenced variables, return with all chunks.
        // Aggregations need all rows to compute correctly, so no chunk narrowing.
        Expr::Agg(_agg) => {
            // For aggregations, we can't narrow chunks. Return referenced vars with all chunks.
            Ok(all_for_referenced_vars(expr, ctx))
        }

        // Anonymous functions: return referenced vars with all chunks.
        Expr::AnonymousFunction { input, .. } => {
            let mut refs = Vec::new();
            for i in input {
                collect_column_refs(i, &mut refs);
            }
            refs.sort();
            refs.dedup();
            if refs.is_empty() {
                Ok(ctx.all())
            } else {
                Ok(DatasetSelection::all_for_vars(refs))
            }
        }

        // Gather/index access: return referenced vars with all chunks.
        Expr::Gather { expr, idx, .. } => {
            let mut refs = Vec::new();
            collect_column_refs(expr, &mut refs);
            collect_column_refs(idx, &mut refs);
            refs.sort();
            refs.dedup();
            if refs.is_empty() {
                Ok(ctx.all())
            } else {
                Ok(DatasetSelection::all_for_vars(refs))
            }
        }

        // Eval expressions: compile the inner expression.
        Expr::Eval { expr, .. } => compile_node(expr.as_ref(), ctx),

        // Field access: return referenced fields as vars with all chunks.
        Expr::Field(names) => {
            let vars: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            if vars.is_empty() {
                Ok(ctx.all())
            } else {
                Ok(DatasetSelection::all_for_vars(vars))
            }
        }

        // These don't reference specific columns, return all conservatively.
        Expr::Element | Expr::Len | Expr::SubPlan(_, _) | Expr::DataTypeFunction(_) => {
            Ok(ctx.all())
        }
        
        // StructEval is complex, return all conservatively with referenced vars.
        Expr::StructEval { expr, .. } => Ok(all_for_referenced_vars(expr.as_ref(), ctx)),
    }
}
