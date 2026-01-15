use super::compile_ctx::CompileCtx;
use super::errors::CompileError;
use super::index_ranges::index_range_for_index_dim;
use super::literals::strip_wrappers;
use super::selection::{DataArraySelection, DatasetSelection, HyperRectangleSelection, RangeList};
use super::types::{BoundKind, CoordScalar, IndexRange, ValueRange};
use super::prelude::*;
use polars::prelude::Series;

/// Take an nd interpolation and extract the necessary selection to allow it to work.
/// For an interpolation, if the value is (e.g.) 8.5 and the chunk is from 5 to 10, we only need chunk 5->10.
/// If the value is 11.5, we need both chunks 5->10 and 10->15.
pub(super) fn interpolate_selection_nd(
    source_coords: &Expr,
    _source_values: &Expr,
    target_values: &Expr,
    ctx: &mut CompileCtx<'_>,
) -> Result<DatasetSelection, CompileError> {
    // Extract coordinate names from the source coord struct expression.
    let coord_names = extract_column_names(source_coords);
    if coord_names.is_empty() {
        return Ok(ctx.all());
    }

    let Some(target_struct) = extract_literal_struct_series(target_values) else {
        return Ok(ctx.all());
    };

    let Ok(target_sc) = target_struct.struct_() else {
        return Ok(ctx.all());
    };
    let target_fields = target_sc.fields_as_series();

    let mut rect = HyperRectangleSelection::all();

    for name in coord_names {
        // Only constrain actual dataset dimensions; otherwise we can't safely prune chunks.
        if !ctx.dims.iter().any(|d| d == &name) {
            continue;
        }

        let Some(s) = target_fields.iter().find(|s| s.name() == name.as_str()) else {
            return Ok(ctx.all());
        };

        let Ok((min_v, max_v)) = series_min_max_scalar(s) else {
            return Ok(ctx.all());
        };

        let mut vr = ValueRange::default();
        vr.min = Some((min_v, BoundKind::Inclusive));
        vr.max = Some((max_v, BoundKind::Inclusive));

        let dim_idx = ctx
            .dims
            .iter()
            .position(|d| d == &name)
            .ok_or_else(|| CompileError::Unsupported("dimension not found".to_owned()))?;
        let dim_len = ctx
            .dim_lengths
            .get(dim_idx)
            .copied()
            .ok_or_else(|| CompileError::Unsupported("dimension length unavailable".to_owned()))?;

        let mut idx_range = match ctx.resolver.index_range_for_value_range(&name, &vr) {
            Ok(Some(r)) => r,
            Ok(None) => {
                // Index-only dimension fallback (only works for integer scalar ranges).
                if ctx.meta.arrays.get(&name).is_none() {
                    index_range_for_index_dim(&vr, dim_len)
                        .ok_or_else(|| CompileError::Unsupported("failed to plan index-only dimension".to_owned()))?
                } else {
                    return Ok(ctx.all());
                }
            }
            Err(_) => return Ok(ctx.all()),
        };

        if idx_range.is_empty() {
            return Ok(DatasetSelection::empty());
        }

        // Expand by one in index space to ensure we include neighbors for interpolation.
        idx_range = expand_index_range(idx_range, dim_len);

        rect = rect.with_dim(name, RangeList::from_index_range(idx_range));
        if rect.is_empty() {
            return Ok(DatasetSelection::empty());
        }
    }

    // If we couldn't constrain anything, fall back to conservative All.
    if rect.dims().next().is_none() {
        return Ok(ctx.all());
    }

    let sel = DataArraySelection(vec![rect]);
    Ok(DatasetSelection::for_vars_with_selection(
        ctx.vars.to_vec(),
        sel,
    ))
}

fn expand_index_range(r: IndexRange, len: u64) -> IndexRange {
    let start = r.start.saturating_sub(1);
    let end_exclusive = r.end_exclusive.saturating_add(1).min(len);
    IndexRange { start, end_exclusive }
}

fn extract_literal_struct_series(expr: &Expr) -> Option<Series> {
    let expr = strip_wrappers(expr);
    let Expr::Literal(lit) = expr else {
        return None;
    };
    match lit {
        LiteralValue::Series(s) => Some((**s).clone()),
        _ => None,
    }
}

fn extract_column_names(expr: &Expr) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    walk_expr(expr, &mut |e| {
        if let Expr::Column(name) = e {
            out.push(name.to_string());
        }
    });
    out.sort();
    out.dedup();
    out
}

fn walk_expr(expr: &Expr, f: &mut impl FnMut(&Expr)) {
    f(expr);
    match expr {
        Expr::Alias(inner, _)
        | Expr::KeepName(inner)
        | Expr::RenameAlias { expr: inner, .. } => walk_expr(inner.as_ref(), f),
        Expr::Cast { expr: inner, .. }
        | Expr::Sort { expr: inner, .. }
        | Expr::SortBy { expr: inner, .. }
        | Expr::Explode { input: inner, .. }
        | Expr::Slice { input: inner, .. } => walk_expr(inner.as_ref(), f),
        Expr::Over { function, .. } => walk_expr(function.as_ref(), f),
        Expr::Rolling { function, .. } => walk_expr(function.as_ref(), f),
        Expr::Filter { by, .. } => walk_expr(by.as_ref(), f),
        Expr::BinaryExpr { left, right, .. } => {
            walk_expr(left.as_ref(), f);
            walk_expr(right.as_ref(), f);
        }
        Expr::Ternary { predicate, truthy, falsy } => {
            walk_expr(predicate.as_ref(), f);
            walk_expr(truthy.as_ref(), f);
            walk_expr(falsy.as_ref(), f);
        }
        Expr::Function { input, .. } => {
            for e in input {
                walk_expr(e, f);
            }
        }
        Expr::Selector(sel) => walk_expr(&sel.clone().as_expr(), f),
        _ => {}
    }
}

fn series_min_max_scalar(s: &Series) -> Result<(CoordScalar, CoordScalar), ()> {
    use polars::prelude::DataType as PlDataType;
    match s.dtype() {
        PlDataType::Int64 => {
            let ca = s.i64().map_err(|_| ())?;
            let mut min: Option<i64> = None;
            let mut max: Option<i64> = None;
            for v in ca.into_iter() {
                let v = v.ok_or(())?;
                min = Some(min.map(|m| m.min(v)).unwrap_or(v));
                max = Some(max.map(|m| m.max(v)).unwrap_or(v));
            }
            Ok((CoordScalar::I64(min.ok_or(())?), CoordScalar::I64(max.ok_or(())?)))
        }
        PlDataType::UInt64 => {
            let ca = s.u64().map_err(|_| ())?;
            let mut min: Option<u64> = None;
            let mut max: Option<u64> = None;
            for v in ca.into_iter() {
                let v = v.ok_or(())?;
                min = Some(min.map(|m| m.min(v)).unwrap_or(v));
                max = Some(max.map(|m| m.max(v)).unwrap_or(v));
            }
            Ok((CoordScalar::U64(min.ok_or(())?), CoordScalar::U64(max.ok_or(())?)))
        }
        PlDataType::Float64 => {
            let ca = s.f64().map_err(|_| ())?;
            let mut min: Option<f64> = None;
            let mut max: Option<f64> = None;
            for v in ca.into_iter() {
                let v = v.ok_or(())?;
                if v.is_nan() {
                    return Err(());
                }
                min = Some(min.map(|m| m.min(v)).unwrap_or(v));
                max = Some(max.map(|m| m.max(v)).unwrap_or(v));
            }
            Ok((CoordScalar::F64(min.ok_or(())?), CoordScalar::F64(max.ok_or(())?)))
        }
        _ => {
            // Try a last-ditch cast to Float64.
            let s2 = s.cast(&PlDataType::Float64).map_err(|_| ())?;
            series_min_max_scalar(&s2)
        }
    }
}