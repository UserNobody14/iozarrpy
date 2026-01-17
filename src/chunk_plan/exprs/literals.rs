use crate::chunk_plan::prelude::*;
use crate::chunk_plan::indexing::plan::ChunkPlanNode;
use crate::chunk_plan::indexing::types::{CoordScalar, DimChunkRange, IndexRange};

pub(crate) fn apply_time_encoding(raw: i64, te: Option<&TimeEncoding>) -> CoordScalar {
    if let Some(enc) = te {
        let ns = enc.decode(raw);
        if enc.is_duration {
            CoordScalar::DurationNs(ns)
        } else {
            CoordScalar::DatetimeNs(ns)
        }
    } else {
        CoordScalar::I64(raw)
    }
}

pub(super) fn literal_anyvalue(lit: &LiteralValue) -> Option<AnyValue<'static>> {
    match lit {
        LiteralValue::Scalar(s) => Some(s.clone().into_value().into_static()),
        LiteralValue::Series(s) => {
            // Polars may represent some Python literals (notably datetimes) as a length-1 Series literal.
            if s.len() != 1 {
                return None;
            }
            let v = s.get(0).ok()?;
            Some(v.into_static())
        }
        LiteralValue::Dyn(d) => {
            // Polars (via pyo3-polars) commonly serializes Python literals as dyn literals
            // (e.g. "dyn int: 20"). Polars doesn't currently expose a stable public API
            // to convert `DynLiteralValue` -> `AnyValue` in our dependency surface, so
            // we conservatively parse the debug representation for the primitive types
            // we need for chunk planning.
            //
            // If parsing fails, we return None and planning will fall back to AllChunks.
            let s = format!("{d:?}");
            let s = s.trim();
            if let Some(rest) = s.strip_prefix("dyn int:") {
                let v = rest.trim().parse::<i64>().ok()?;
                return Some(AnyValue::Int64(v).into_static());
            }
            if let Some(rest) = s.strip_prefix("dyn float:") {
                let v = rest.trim().parse::<f64>().ok()?;
                return Some(AnyValue::Float64(v).into_static());
            }
            if let Some(rest) = s.strip_prefix("dyn bool:") {
                let v = rest.trim().parse::<bool>().ok()?;
                return Some(AnyValue::Boolean(v).into_static());
            }
            None
        }
        other => {
            // This is intentionally conservative, but we don’t want silent fallthrough.
            if cfg!(debug_assertions) {
                eprintln!("chunk_plan: unsupported LiteralValue variant for planning: {other:?}");
            }
            None
        }
    }
}

pub(super) fn literal_to_scalar(
    lit: &LiteralValue,
    time_encoding: Option<&TimeEncoding>,
) -> Option<CoordScalar> {
    let parse_temporal_from_str = |s: &str| -> Option<CoordScalar> {
        let s = s.trim();

        // Datetime-like: "YYYY-MM-DD HH:MM:SS[.fffffffff]" or "YYYY-MM-DD"
        // Treat as UTC for planning purposes.
        {
            use chrono::{NaiveDate, NaiveDateTime, TimeZone, Utc};
            if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
                return Some(CoordScalar::DatetimeNs(
                    Utc.from_utc_datetime(&dt).timestamp_nanos_opt()?,
                ));
            }
            if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f") {
                return Some(CoordScalar::DatetimeNs(
                    Utc.from_utc_datetime(&dt).timestamp_nanos_opt()?,
                ));
            }
            if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
                let dt = d.and_hms_opt(0, 0, 0)?;
                return Some(CoordScalar::DatetimeNs(
                    Utc.from_utc_datetime(&dt).timestamp_nanos_opt()?,
                ));
            }
        }

        // Duration-like: "1h" (we only need hours for our current predicates).
        if let Some(hours) = s.strip_suffix('h').and_then(|v| v.trim().parse::<i64>().ok()) {
            return Some(CoordScalar::DurationNs(hours.saturating_mul(3_600_000_000_000)));
        }

        None
    };

    match lit {
        LiteralValue::Dyn(d) => {
            // Parse common dyn literal debug formats that Polars uses for Python values.
            // This is intentionally conservative; returning None will disable planning.
            let s = format!("{d:?}");
            let s = s.trim();

            if let Some(rest) = s.strip_prefix("dyn int:") {
                let v = rest.trim().parse::<i64>().ok()?;
                return Some(CoordScalar::I64(v));
            }
            if let Some(rest) = s.strip_prefix("dyn float:") {
                let v = rest.trim().parse::<f64>().ok()?;
                return Some(CoordScalar::F64(v));
            }
            if let Some(rest) = s.strip_prefix("dyn bool:") {
                let v = rest.trim().parse::<bool>().ok()?;
                return Some(CoordScalar::I64(i64::from(v)));
            }

            parse_temporal_from_str(s)
        }
        _ => {
            let parsed = match literal_anyvalue(lit)? {
                AnyValue::Int64(v) => Some(CoordScalar::I64(v)),
                AnyValue::Int32(v) => Some(CoordScalar::I64(v as i64)),
                AnyValue::Int16(v) => Some(CoordScalar::I64(v as i64)),
                AnyValue::Int8(v) => Some(CoordScalar::I64(v as i64)),
                AnyValue::UInt64(v) => Some(CoordScalar::U64(v)),
                AnyValue::UInt32(v) => Some(CoordScalar::U64(v as u64)),
                AnyValue::UInt16(v) => Some(CoordScalar::U64(v as u64)),
                AnyValue::UInt8(v) => Some(CoordScalar::U64(v as u64)),
                AnyValue::Float64(v) => Some(CoordScalar::F64(v)),
                AnyValue::Float32(v) => Some(CoordScalar::F64(v as f64)),
                AnyValue::Datetime(value, time_unit, _) => {
                    let ns = match time_unit {
                        polars::prelude::TimeUnit::Nanoseconds => value,
                        polars::prelude::TimeUnit::Microseconds => value * 1_000,
                        polars::prelude::TimeUnit::Milliseconds => value * 1_000_000,
                    };
                    let _ = time_encoding;
                    Some(CoordScalar::DatetimeNs(ns))
                }
                AnyValue::Date(days) => {
                    let ns = days as i64 * 86400 * 1_000_000_000;
                    let _ = time_encoding;
                    Some(CoordScalar::DatetimeNs(ns))
                }
                AnyValue::Duration(value, time_unit) => {
                    let ns = match time_unit {
                        polars::prelude::TimeUnit::Nanoseconds => value,
                        polars::prelude::TimeUnit::Microseconds => value * 1_000,
                        polars::prelude::TimeUnit::Milliseconds => value * 1_000_000,
                    };
                    let _ = time_encoding;
                    Some(CoordScalar::DurationNs(ns))
                }
                other => {
                    if cfg!(debug_assertions) {
                        eprintln!("chunk_plan: unsupported AnyValue for planning: {other:?}");
                    }
                    None
                }
            };

            // Last-ditch: parse the debug formatting of the literal itself, which is what our
            // error messages expose (and what Python tends to show for datetime/timedelta literals).
            parsed.or_else(|| parse_temporal_from_str(&format!("{lit:?}")))
        }
    }
}

pub(super) fn col_lit(col_side: &Expr, lit_side: &Expr) -> Option<(String, LiteralValue)> {
    let col_side = strip_wrappers(col_side);
    let lit_side = strip_wrappers(lit_side);
    if let (Expr::Column(name), Expr::Literal(lit)) = (col_side, lit_side) {
        Some((name.to_string(), lit.clone()))
    } else {
        None
    }
}

pub(crate) fn strip_wrappers(mut e: &Expr) -> &Expr {
    loop {
        match e {
            Expr::Alias(inner, _) => e = inner.as_ref(),
            Expr::Cast { expr, .. } => e = expr.as_ref(),
            _ => return e,
        }
    }
}

pub(super) fn reverse_operator(op: Operator) -> Operator {
    match op {
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        _ => op,
    }
}

#[allow(dead_code)]
pub(super) fn chunk_ranges_for_index_range(
    idx: IndexRange,
    chunk_size: u64,
    grid_dim: u64,
) -> Option<DimChunkRange> {
    if idx.is_empty() {
        return None;
    }
    let chunk_start = idx.start / chunk_size;
    let last = idx.end_exclusive.saturating_sub(1);
    let chunk_end = last / chunk_size;
    if chunk_start >= grid_dim {
        return None;
    }
    let end = chunk_end.min(grid_dim.saturating_sub(1));
    Some(DimChunkRange {
        start_chunk: chunk_start,
        end_chunk_inclusive: end,
    })
}

#[allow(dead_code)]
pub(super) fn rect_all_dims(grid_shape: &[u64]) -> Vec<DimChunkRange> {
    grid_shape
        .iter()
        .map(|&g| DimChunkRange {
            start_chunk: 0,
            end_chunk_inclusive: g.saturating_sub(1),
        })
        .collect()
}

#[allow(dead_code)]
pub(super) fn and_nodes(a: ChunkPlanNode, b: ChunkPlanNode) -> ChunkPlanNode {
    match (a, b) {
        (ChunkPlanNode::Empty, _) | (_, ChunkPlanNode::Empty) => ChunkPlanNode::Empty,
        (ChunkPlanNode::AllChunks, x) | (x, ChunkPlanNode::AllChunks) => x,
        (ChunkPlanNode::Explicit(xs), ChunkPlanNode::Explicit(ys)) => {
            // Exact set intersection on explicit chunk coords.
            let set: std::collections::BTreeSet<Vec<u64>> = ys.into_iter().collect();
            ChunkPlanNode::Explicit(xs.into_iter().filter(|v| set.contains(v)).collect())
        }
        (ChunkPlanNode::Explicit(xs), _) | (_, ChunkPlanNode::Explicit(xs)) => {
            // Conservative: we don't have enough info to intersect Explicit with Rect/Union
            // without chunk grid context, so keep the explicit set.
            ChunkPlanNode::Explicit(xs)
        }
        (ChunkPlanNode::Union(xs), y) => {
            ChunkPlanNode::Union(xs.into_iter().map(|x| and_nodes(x, y.clone())).collect())
        }
        (x, ChunkPlanNode::Union(ys)) => {
            ChunkPlanNode::Union(ys.into_iter().map(|y| and_nodes(x.clone(), y)).collect())
        }
        (ChunkPlanNode::Rect(a), ChunkPlanNode::Rect(b)) => {
            if a.len() != b.len() {
                return ChunkPlanNode::Empty;
            }
            let mut out = Vec::with_capacity(a.len());
            for i in 0..a.len() {
                let Some(r) = a[i].intersect(&b[i]) else {
                    return ChunkPlanNode::Empty;
                };
                out.push(r);
            }
            ChunkPlanNode::Rect(out)
        }
    }
}

#[allow(dead_code)]
pub(super) fn or_nodes(a: ChunkPlanNode, b: ChunkPlanNode) -> ChunkPlanNode {
    match (a, b) {
        (ChunkPlanNode::Empty, x) | (x, ChunkPlanNode::Empty) => x,
        (ChunkPlanNode::AllChunks, _) | (_, ChunkPlanNode::AllChunks) => ChunkPlanNode::AllChunks,
        (ChunkPlanNode::Explicit(mut xs), ChunkPlanNode::Explicit(ys)) => {
            xs.extend(ys);
            ChunkPlanNode::Explicit(xs)
        }
        (ChunkPlanNode::Explicit(xs), y) => ChunkPlanNode::Union(vec![ChunkPlanNode::Explicit(xs), y]),
        (x, ChunkPlanNode::Explicit(ys)) => ChunkPlanNode::Union(vec![x, ChunkPlanNode::Explicit(ys)]),
        (ChunkPlanNode::Union(mut xs), ChunkPlanNode::Union(ys)) => {
            xs.extend(ys);
            ChunkPlanNode::Union(xs)
        }
        (ChunkPlanNode::Union(mut xs), y) => {
            xs.push(y);
            ChunkPlanNode::Union(xs)
        }
        (x, ChunkPlanNode::Union(mut ys)) => {
            ys.insert(0, x);
            ChunkPlanNode::Union(ys)
        }
        (x, y) => ChunkPlanNode::Union(vec![x, y]),
    }
}
