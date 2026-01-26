use crate::{IntoIStr, IStr};
use crate::chunk_plan::prelude::*;
use crate::chunk_plan::indexing::types::CoordScalar;

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
                AnyValue::DatetimeOwned(value, time_unit, _) => {
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

pub(super) fn col_lit(col_side: &Expr, lit_side: &Expr) -> Option<(IStr, LiteralValue)> {
    let col_side = strip_wrappers(col_side);
    let lit_side = strip_wrappers(lit_side);
    if let (Expr::Column(name), Expr::Literal(lit)) = (col_side, lit_side) {
        Some((name.to_string().istr(), lit.clone()))
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
