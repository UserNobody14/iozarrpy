use super::expr_walk::walk_expr;
use super::literals::{
    literal_to_scalar, reverse_operator,
    strip_wrappers,
};
use crate::chunk_plan::exprs::compile_ctx::LazyCompileCtx;
use crate::chunk_plan::indexing::types::ValueRangePresent;
use crate::chunk_plan::prelude::*;
use crate::{IStr, IntoIStr};

/// Try to extract a column name and value range from a comparison expression.
pub(super) fn try_expr_to_value_range_lazy(
    expr: &Expr,
) -> Option<(IStr, ValueRangePresent)> {
    let expr = strip_wrappers(expr);
    let Expr::BinaryExpr { left, op, right } =
        expr
    else {
        return None;
    };
    if !matches!(
        op,
        Operator::Eq
            | Operator::GtEq
            | Operator::Gt
            | Operator::LtEq
            | Operator::Lt
    ) {
        return None;
    }

    let (col, lit, op_eff) = if let (
        Expr::Column(name),
        Expr::Literal(lit),
    ) = (
        strip_wrappers(left),
        strip_wrappers(right),
    ) {
        (name.istr(), lit, *op)
    } else if let (
        Expr::Literal(lit),
        Expr::Column(name),
    ) = (
        strip_wrappers(left),
        strip_wrappers(right),
    ) {
        (name.istr(), lit, reverse_operator(*op))
    } else {
        return None;
    };

    let Ok(scalar) = literal_to_scalar(lit)
    else {
        return None;
    };

    let vr = ValueRangePresent::from_polars_op(
        op_eff, scalar,
    )
    .ok()?;

    Some((col, vr))
}

pub(super) fn expr_to_col_name(
    e: &Expr,
) -> Option<IStr> {
    if let Expr::Column(name) = strip_wrappers(e)
    {
        Some(name.istr())
    } else {
        None
    }
}

/// Extract column names from an expression (for lazy interpolation).
pub(super) fn extract_column_names_lazy(
    expr: &Expr,
) -> Option<Vec<IStr>> {
    let mut out: Vec<IStr> = Vec::new();
    walk_expr(expr, &mut |e| {
        if let Expr::Column(name) = e {
            out.push(name.istr());
        }
    });
    out.sort();
    out.dedup();
    if out.is_empty() { None } else { Some(out) }
}

/// Extract a literal struct Series from an expression.
pub(super) fn extract_literal_struct_series_lazy(
    expr: &Expr,
) -> Option<polars::prelude::Series> {
    use super::literals::strip_wrappers;
    use polars::prelude::LiteralValue;
    let expr = strip_wrappers(expr);
    let Expr::Literal(lit) = expr else {
        return None;
    };
    match lit {
        LiteralValue::Series(s) => {
            Some((**s).clone())
        }
        _ => None,
    }
}

/// Extract scalar values from a Series.
pub(super) fn series_values_scalar_lazy(
    s: &polars::prelude::Series,
) -> Option<
    Vec<crate::chunk_plan::indexing::types::CoordScalar>,
>{
    use crate::chunk_plan::indexing::types::CoordScalar;
    use polars::prelude::DataType;
    use polars::prelude::TimeUnit;

    let mut out = Vec::with_capacity(s.len());

    match s.dtype() {
        DataType::Int64 => {
            let ca = s.i64().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(CoordScalar::I64(v));
            }
        }
        DataType::UInt64 => {
            let ca = s.u64().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(CoordScalar::U64(v));
            }
        }
        DataType::Float64 => {
            let ca = s.f64().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(CoordScalar::F64(
                    v.into(),
                ));
            }
        }
        DataType::Float32 => {
            let ca = s.f32().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(CoordScalar::F64(
                    (v as f64).into(),
                ));
            }
        }
        DataType::Int32 => {
            let ca = s.i32().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(CoordScalar::I64(
                    v as i64,
                ));
            }
        }
        DataType::Int16 => {
            let ca = s.i16().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(CoordScalar::I64(
                    v as i64,
                ));
            }
        }
        DataType::Int8 => {
            let ca = s.i8().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(CoordScalar::I64(
                    v as i64,
                ));
            }
        }
        DataType::UInt32 => {
            let ca = s.u32().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(CoordScalar::U64(
                    v as u64,
                ));
            }
        }
        DataType::UInt16 => {
            let ca = s.u16().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(CoordScalar::U64(
                    v as u64,
                ));
            }
        }
        DataType::UInt8 => {
            let ca = s.u8().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(CoordScalar::U64(
                    v as u64,
                ));
            }
        }
        DataType::Datetime(tu, _) => {
            // Use physical representation for Datetime
            let phys = s.to_physical_repr();
            let ca = phys.i64().ok()?;
            for v in ca.into_iter().flatten() {
                let ns = match tu {
                    TimeUnit::Nanoseconds => v,
                    TimeUnit::Microseconds => {
                        v.saturating_mul(1_000)
                    }
                    TimeUnit::Milliseconds => v
                        .saturating_mul(
                            1_000_000,
                        ),
                };
                out.push(
                    CoordScalar::DatetimeNs(ns),
                );
            }
        }
        DataType::Date => {
            // Use physical representation for Date (i32 days since epoch)
            let phys = s.to_physical_repr();
            let ca = phys.i32().ok()?;
            for v in ca.into_iter().flatten() {
                let ns = (v as i64)
                    .saturating_mul(
                        86_400_000_000_000,
                    );
                out.push(
                    CoordScalar::DatetimeNs(ns),
                );
            }
        }
        DataType::Duration(tu) => {
            // Use physical representation for Duration
            let phys = s.to_physical_repr();
            let ca = phys.i64().ok()?;
            for v in ca.into_iter().flatten() {
                let ns = match tu {
                    TimeUnit::Nanoseconds => v,
                    TimeUnit::Microseconds => {
                        v.saturating_mul(1_000)
                    }
                    TimeUnit::Milliseconds => v
                        .saturating_mul(
                            1_000_000,
                        ),
                };
                out.push(
                    CoordScalar::DurationNs(ns),
                );
            }
        }
        DataType::Time => {
            // Use physical representation for Time (i64 nanoseconds since midnight)
            let phys = s.to_physical_repr();
            let ca = phys.i64().ok()?;
            for v in ca.into_iter().flatten() {
                out.push(
                    CoordScalar::DurationNs(v),
                );
            }
        }
        DataType::Boolean => todo!(),
        DataType::UInt128 => todo!(),
        DataType::Int128 => todo!(),
        DataType::Float16 => todo!(),
        DataType::Decimal(_, _) => todo!(),
        DataType::String => todo!(),
        DataType::Binary => todo!(),
        DataType::BinaryOffset => todo!(),
        DataType::Array(data_type, _) => todo!(),
        DataType::List(data_type) => todo!(),
        DataType::Object(_) => todo!(),
        DataType::Null => todo!(),
        DataType::Categorical(
            categories,
            categorical_mapping,
        ) => todo!(),
        DataType::Enum(
            frozen_categories,
            categorical_mapping,
        ) => todo!(),
        DataType::Struct(fields) => todo!(),
        DataType::Extension(
            extension_type_instance,
            data_type,
        ) => todo!(),
        DataType::Unknown(unknown_kind) => {
            todo!()
        }
    }

    Some(out)
}
