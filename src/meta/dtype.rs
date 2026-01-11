use polars::prelude::{DataType as PlDataType, TimeUnit};

use crate::meta::types::TimeEncoding;

pub(crate) fn zarr_dtype_to_polars(
    zarr_identifier: &str,
    time_encoding: Option<&TimeEncoding>,
) -> PlDataType {
    if let Some(te) = time_encoding {
        return if te.is_duration {
            PlDataType::Duration(TimeUnit::Nanoseconds)
        } else {
            PlDataType::Datetime(TimeUnit::Nanoseconds, None)
        };
    }

    match zarr_identifier {
        "bool" => PlDataType::Boolean,
        "int8" => PlDataType::Int8,
        "int16" => PlDataType::Int16,
        "int32" => PlDataType::Int32,
        "int64" => PlDataType::Int64,
        "uint8" => PlDataType::UInt8,
        "uint16" => PlDataType::UInt16,
        "uint32" => PlDataType::UInt32,
        "uint64" => PlDataType::UInt64,
        "float16" | "bfloat16" => PlDataType::Float32,
        "float32" => PlDataType::Float32,
        "float64" => PlDataType::Float64,
        "string" => PlDataType::String,
        _ => PlDataType::Binary,
    }
}

