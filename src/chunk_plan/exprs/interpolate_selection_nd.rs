use crate::chunk_plan::exprs::compile_ctx::CompileCtx;
use crate::chunk_plan::exprs::errors::CompileError;
use crate::chunk_plan::indexing::index_ranges::index_range_for_index_dim;
use crate::chunk_plan::exprs::literals;
use crate::chunk_plan::indexing::selection::{DataArraySelection, DatasetSelection, HyperRectangleSelection, RangeList, dataset_for_vars_with_selection};
use crate::chunk_plan::indexing::types::{BoundKind, CoordScalar, IndexRange, ValueRange};
use crate::chunk_plan::prelude::*;
use crate::chunk_plan::indexing::selection::SetOperations;
use polars::prelude::Series;

/// Take an nd interpolation and extract the necessary selection to allow it to work.
/// For an interpolation, if the value is (e.g.) 8.5 and the chunk is from 5 to 10, we only need chunk 5->10.
/// If the value is 11.5, we need both chunks 5->10 and 10->15.
pub(crate) fn interpolate_selection_nd(
    source_coords: &Expr,
    _source_values: &Expr,
    target_values: &Expr,
    ctx: &mut CompileCtx<'_>,
) -> Result<DatasetSelection, CompileError> {
    // Extract coordinate names from the source coord struct expression.
    let coord_names = extract_column_names(source_coords);
    if coord_names.is_empty() {
        return Ok(DatasetSelection::NoSelectionMade);
    }

    let Some(target_struct) = extract_literal_struct_series(target_values) else {
        return Ok(DatasetSelection::NoSelectionMade);
    };

    let Ok(target_sc) = target_struct.struct_() else {
        return Ok(DatasetSelection::NoSelectionMade);
    };
    let target_fields = target_sc.fields_as_series();

    let mut rect = HyperRectangleSelection::all();

    for name in coord_names {
        // Only constrain actual dataset dimensions; otherwise we can't safely prune chunks.
        if !ctx.dims.iter().any(|d| d == &name) {
            continue;
        }

        // If this coord dim isn't present in the target DataFrame, interpolars treats it as a
        // group key (README: "Grouped interpolation over extra coordinate dims").
        // For planning: we can't constrain that dim, but we *can* still constrain others.
        let Some(s) = target_fields.iter().find(|s| s.name() == name.as_str()) else {
            continue;
        };

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

        // New interpolars behavior can group by "extra" columns in the target coords.
        // For chunk planning we must take the union of all coordinate points (across all groups),
        // *without* collapsing them into a single [min,max] range (which can over-select).
        let Ok(values) = series_values_scalar(s) else {
            return Ok(DatasetSelection::NoSelectionMade);
        };

        let mut idx_ranges: Vec<IndexRange> = Vec::new();
        for v in values {
            let Some(idx_range) = index_range_for_interp_value(&name, &v, dim_len, ctx) else {
                return Ok(DatasetSelection::NoSelectionMade);
            };
            idx_ranges.push(idx_range);
        }

        if idx_ranges.is_empty() {
            return Ok(DatasetSelection::empty());
        }

        let mut rl = RangeList::empty();
        for r in idx_ranges {
            rl = rl.union(&RangeList::from_index_range(r));
        }

        rect = rect.with_dim(name, rl);
        if rect.is_empty() {
            return Ok(DatasetSelection::empty());
        }
    }

    // If we couldn't constrain anything, fall back to conservative All.
    if rect.dims().next().is_none() {
        return Ok(DatasetSelection::NoSelectionMade);
    }

    let sel = DataArraySelection(vec![rect]);
    Ok(dataset_for_vars_with_selection(
        ctx.vars.to_vec(),
        sel,
    ))
}

fn expand_index_range(r: IndexRange, len: u64) -> IndexRange {
    let start = r.start.saturating_sub(1);
    let end_exclusive = r.end_exclusive.saturating_add(1).min(len);
    IndexRange { start, end_exclusive }
}

fn index_range_for_interp_value(
    dim: &str,
    v: &CoordScalar,
    dim_len: u64,
    ctx: &mut CompileCtx<'_>,
) -> Option<IndexRange> {
    if dim_len == 0 {
        return Some(IndexRange {
            start: 0,
            end_exclusive: 0,
        });
    }

    // For interpolation we need the bracketing indices around `v` (clamping at bounds).
    //
    // We derive those using two monotonic lookups:
    // - `<= v` (max bound) gives us the last index on the left
    // - `>= v` (min bound) gives us the first index on the right
    //
    // This works even when `v` is between grid points (no exact match) and for out-of-bounds
    // values (it naturally clamps to 0 / n-1).
    let mut vr_max = ValueRange::default();
    vr_max.max = Some((v.clone(), BoundKind::Inclusive));
    let mut vr_min = ValueRange::default();
    vr_min.min = Some((v.clone(), BoundKind::Inclusive));

    let (left_end, right_start) = match (
        ctx.resolver.index_range_for_value_range(dim, &vr_max),
        ctx.resolver.index_range_for_value_range(dim, &vr_min),
    ) {
        (Ok(Some(r_max)), Ok(Some(r_min))) => (r_max.end_exclusive, r_min.start),
        (Ok(None), Ok(None)) => {
            // Index-only dimension fallback (only works for integer scalar ranges).
            if ctx.meta.arrays.get(dim).is_some() {
                return None;
            }
            let r_max = index_range_for_index_dim(&vr_max, dim_len)?;
            let r_min = index_range_for_index_dim(&vr_min, dim_len)?;
            (r_max.end_exclusive, r_min.start)
        }
        _ => return None,
    };

    let last = dim_len - 1;
    let left_idx = if left_end == 0 {
        0
    } else {
        (left_end - 1).min(last)
    };
    let right_idx = right_start.min(last);

    let start = left_idx.min(right_idx);
    let end_exclusive = left_idx.max(right_idx).saturating_add(1).min(dim_len);

    // Optionally expand by one to be extra-safe near chunk boundaries.
    Some(expand_index_range(
        IndexRange { start, end_exclusive },
        dim_len,
    ))
}

fn extract_literal_struct_series(expr: &Expr) -> Option<Series> {
    let expr = literals::strip_wrappers(expr);
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

fn series_values_scalar(s: &Series) -> Result<Vec<CoordScalar>, ()> {
    use polars::prelude::DataType as PlDataType;
    use polars::prelude::TimeUnit;
    match s.dtype() {
        PlDataType::Date => {
            // Polars Date is i32 days since unix epoch.
            let phys = s.to_physical_repr();
            let ca = phys.i32().map_err(|_| ())?;
            let mut out: Vec<CoordScalar> = Vec::with_capacity(ca.len());
            for v in ca.into_iter() {
                let days = v.ok_or(())? as i64;
                let ns = days.saturating_mul(86_400_000_000_000);
                out.push(CoordScalar::DatetimeNs(ns));
            }
            Ok(out)
        }
        PlDataType::Datetime(tu, _tz) => {
            // Stored as i64 in the given time unit, relative to unix epoch.
            let phys = s.to_physical_repr();
            let ca = phys.i64().map_err(|_| ())?;
            let mut out: Vec<CoordScalar> = Vec::with_capacity(ca.len());
            for v in ca.into_iter() {
                let raw = v.ok_or(())?;
                let ns = match tu {
                    TimeUnit::Nanoseconds => raw,
                    TimeUnit::Microseconds => raw.saturating_mul(1_000),
                    TimeUnit::Milliseconds => raw.saturating_mul(1_000_000),
                };
                out.push(CoordScalar::DatetimeNs(ns));
            }
            Ok(out)
        }
        PlDataType::Duration(tu) => {
            // Stored as i64 in the given time unit.
            let phys = s.to_physical_repr();
            let ca = phys.i64().map_err(|_| ())?;
            let mut out: Vec<CoordScalar> = Vec::with_capacity(ca.len());
            for v in ca.into_iter() {
                let raw = v.ok_or(())?;
                let ns = match tu {
                    TimeUnit::Nanoseconds => raw,
                    TimeUnit::Microseconds => raw.saturating_mul(1_000),
                    TimeUnit::Milliseconds => raw.saturating_mul(1_000_000),
                };
                out.push(CoordScalar::DurationNs(ns));
            }
            Ok(out)
        }
        PlDataType::Int64 => {
            let ca = s.i64().map_err(|_| ())?;
            let mut out: Vec<CoordScalar> = Vec::with_capacity(ca.len());
            for v in ca.into_iter() {
                let v = v.ok_or(())?;
                out.push(CoordScalar::I64(v));
            }
            Ok(out)
        }
        PlDataType::UInt64 => {
            let ca = s.u64().map_err(|_| ())?;
            let mut out: Vec<CoordScalar> = Vec::with_capacity(ca.len());
            for v in ca.into_iter() {
                let v = v.ok_or(())?;
                out.push(CoordScalar::U64(v));
            }
            Ok(out)
        }
        PlDataType::Float64 => {
            let ca = s.f64().map_err(|_| ())?;
            let mut out: Vec<CoordScalar> = Vec::with_capacity(ca.len());
            for v in ca.into_iter() {
                let v = v.ok_or(())?;
                if v.is_nan() {
                    return Err(());
                }
                out.push(CoordScalar::F64(v));
            }
            Ok(out)
        }
        _ => {
            // Try a last-ditch cast to Float64.
            let s2 = s.cast(&PlDataType::Float64).map_err(|_| ())?;
            series_values_scalar(&s2)
        }
    }
}