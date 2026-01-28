use chrono::{
    NaiveDate, NaiveDateTime, TimeZone, Utc,
};
use zarrs::array::Array;

use crate::meta::types::TimeEncoding;

pub(crate) fn extract_time_encoding<
    TStorage: ?Sized,
>(
    array: &Array<TStorage>,
) -> Option<TimeEncoding> {
    let attrs = array.attributes();

    // xarray: attrs["units"] like "hours since 2024-01-01 00:00:00"
    let units = attrs
        .get("units")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // xarray: attrs["dtype"] like "timedelta64[ns]"
    let dtype = attrs
        .get("dtype")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Duration handling:
    // Some xarray-written datasets set dtype="timedelta64[ns]" but also set units="hours"
    // and store raw values in *hours* (not nanoseconds). In that case we must respect `units`.
    if let Some(dtype_str) = dtype {
        if let Some(dtype_unit_ns) =
            parse_timedelta_dtype(&dtype_str)
        {
            if let Some(units_str) =
                units.as_deref()
            {
                if !units_str.contains(" since ")
                {
                    if let Some(unit_ns) =
                        parse_duration_units(
                            units_str,
                        )
                    {
                        return Some(
                            TimeEncoding {
                                epoch_ns: 0,
                                unit_ns,
                                is_duration: true,
                            },
                        );
                    }
                }
            }
            return Some(TimeEncoding {
                epoch_ns: 0,
                unit_ns: dtype_unit_ns,
                is_duration: true,
            });
        }
    }

    // Datetime handling via CF units like "hours since 2024-01-01 00:00:00".
    if let Some(units_str) = units.as_deref() {
        if let Some((unit_ns, epoch_ns)) =
            parse_cf_time_units(units_str)
        {
            return Some(TimeEncoding {
                epoch_ns,
                unit_ns,
                is_duration: false,
            });
        }
    }

    // Duration handling via simple units like "hours" (no "since" clause).
    if let Some(units_str) = units.as_deref() {
        if let Some(unit_ns) =
            parse_duration_units(units_str)
        {
            return Some(TimeEncoding {
                epoch_ns: 0,
                unit_ns,
                is_duration: true,
            });
        }
    }

    None
}

fn parse_duration_units(
    units: &str,
) -> Option<i64> {
    let unit_str = units.trim().to_lowercase();
    match unit_str.as_str() {
        "nanoseconds" | "nanosecond" | "ns" => {
            Some(1)
        }
        "microseconds" | "microsecond" | "us"
        | "µs" => Some(1_000),
        "milliseconds" | "millisecond" | "ms" => {
            Some(1_000_000)
        }
        "seconds" | "second" | "s" => {
            Some(1_000_000_000)
        }
        "minutes" | "minute" | "min" | "m" => {
            Some(60 * 1_000_000_000)
        }
        "hours" | "hour" | "h" | "hr" => {
            Some(3600 * 1_000_000_000)
        }
        "days" | "day" | "d" => {
            Some(86400 * 1_000_000_000)
        }
        other => {
            if cfg!(debug_assertions) {
                eprintln!(
                    "meta: unsupported duration units in attrs: '{other}' (raw='{units}')"
                );
            }
            None
        }
    }
}

fn parse_cf_time_units(
    units: &str,
) -> Option<(i64, i64)> {
    let parts: Vec<&str> =
        units.splitn(2, " since ").collect();
    if parts.len() != 2 {
        return None;
    }

    let unit_str = parts[0].trim().to_lowercase();
    let epoch_str = parts[1].trim();

    let unit_ns: i64 = match unit_str.as_str() {
        "nanoseconds" | "nanosecond" | "ns" => 1,
        "microseconds" | "microsecond" | "us"
        | "µs" => 1_000,
        "milliseconds" | "millisecond" | "ms" => {
            1_000_000
        }
        "seconds" | "second" | "s" => {
            1_000_000_000
        }
        "minutes" | "minute" | "min" => {
            60 * 1_000_000_000
        }
        "hours" | "hour" | "h" | "hr" => {
            3600 * 1_000_000_000
        }
        "days" | "day" | "d" => {
            86400 * 1_000_000_000
        }
        other => {
            if cfg!(debug_assertions) {
                eprintln!(
                    "meta: unsupported CF time unit in attrs: '{other}' (raw='{units}')"
                );
            }
            return None;
        }
    };

    let epoch_ns =
        parse_datetime_to_ns(epoch_str)?;
    Some((unit_ns, epoch_ns))
}

fn parse_datetime_to_ns(s: &str) -> Option<i64> {
    if let Ok(dt) = NaiveDateTime::parse_from_str(
        s,
        "%Y-%m-%d %H:%M:%S",
    ) {
        return Some(
            Utc.from_utc_datetime(&dt)
                .timestamp_nanos_opt()?,
        );
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(
        s,
        "%Y-%m-%d %H:%M:%S%.f",
    ) {
        return Some(
            Utc.from_utc_datetime(&dt)
                .timestamp_nanos_opt()?,
        );
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(
        s,
        "%Y-%m-%dT%H:%M:%S",
    ) {
        return Some(
            Utc.from_utc_datetime(&dt)
                .timestamp_nanos_opt()?,
        );
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(
        s,
        "%Y-%m-%dT%H:%M:%S%.f",
    ) {
        return Some(
            Utc.from_utc_datetime(&dt)
                .timestamp_nanos_opt()?,
        );
    }
    if let Ok(d) =
        NaiveDate::parse_from_str(s, "%Y-%m-%d")
    {
        let dt = d.and_hms_opt(0, 0, 0)?;
        return Some(
            Utc.from_utc_datetime(&dt)
                .timestamp_nanos_opt()?,
        );
    }
    None
}

fn parse_timedelta_dtype(
    dtype_str: &str,
) -> Option<i64> {
    let s = dtype_str.trim().to_lowercase();
    if !s.starts_with("timedelta64[")
        || !s.ends_with(']')
    {
        return None;
    }
    let unit = s
        .trim_start_matches("timedelta64[")
        .trim_end_matches(']');
    match unit {
        "ns" => Some(1),
        "us" | "µs" => Some(1_000),
        "ms" => Some(1_000_000),
        "s" => Some(1_000_000_000),
        "m" => Some(60 * 1_000_000_000),
        "h" => Some(3600 * 1_000_000_000),
        "d" => Some(86400 * 1_000_000_000),
        other => {
            if cfg!(debug_assertions) {
                eprintln!(
                    "meta: unsupported timedelta64 unit in attrs: '{other}' (raw='{dtype_str}')"
                );
            }
            None
        }
    }
}
