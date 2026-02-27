use crate::chunk_plan::indexing::types::CoordScalar;
use crate::chunk_plan::prelude::*;
use crate::errors::BackendError;
use crate::{IStr, IntoIStr};

pub(crate) fn apply_time_encoding(
    raw: i64,
    te: Option<&TimeEncoding>,
) -> CoordScalar {
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

pub(super) fn literal_anyvalue<'a>(
    lit: &'a LiteralValue,
) -> Option<AnyValue<'a>> {
    match lit {
        LiteralValue::Scalar(s) => {
            Some(s.clone().into_value())
        }
        LiteralValue::Series(s) => {
            // Polars may represent some Python literals (notably datetimes) as a length-1 Series literal.
            if s.len() != 1 {
                return None;
            }
            let v: AnyValue<'a> =
                s.get(0).ok()?;
            Some(v)
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
            if let Some(rest) =
                s.strip_prefix("dyn int:")
            {
                let v = rest
                    .trim()
                    .parse::<i64>()
                    .ok()?;
                return Some(AnyValue::Int64(v));
            }
            if let Some(rest) =
                s.strip_prefix("dyn float:")
            {
                let v = rest
                    .trim()
                    .parse::<f64>()
                    .ok()?;
                return Some(AnyValue::Float64(
                    v,
                ));
            }
            if let Some(rest) =
                s.strip_prefix("dyn bool:")
            {
                let v = rest
                    .trim()
                    .parse::<bool>()
                    .ok()?;
                return Some(AnyValue::Boolean(
                    v,
                ));
            }
            None
        }
        LiteralValue::Range(range) => {
            todo!()
        }
    }
}

pub(super) fn literal_to_scalar(
    lit: &LiteralValue,
) -> Result<CoordScalar, BackendError> {
    let anyval = literal_anyvalue(lit);
    if let Some(anyval) = anyval {
        Ok(CoordScalar::try_from(anyval)?)
    } else {
        Err(BackendError::UnsupportedLiteral {
            lit: lit.clone(),
        })
    }
}

pub(super) fn col_lit(
    col_side: &Expr,
    lit_side: &Expr,
) -> Option<(IStr, LiteralValue)> {
    let col_side = strip_wrappers(col_side);
    let lit_side = strip_wrappers(lit_side);
    if let (
        Expr::Column(name),
        Expr::Literal(lit),
    ) = (col_side, lit_side)
    {
        Some((
            name.to_string().istr(),
            lit.clone(),
        ))
    } else {
        None
    }
}

pub(crate) fn strip_wrappers(
    mut e: &Expr,
) -> &Expr {
    loop {
        match e {
            Expr::Alias(inner, _) => {
                e = inner.as_ref()
            }
            Expr::Cast { expr, .. } => {
                e = expr.as_ref()
            }
            _ => return e,
        }
    }
}

pub(super) fn reverse_operator(
    op: Operator,
) -> Operator {
    match op {
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        _ => op,
    }
}
