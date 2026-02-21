use std::borrow::Cow;

use polars::prelude::{
    DataType as PlDataType, TimeUnit,
};
use zarrs::{
    array::DataType as ZarrDataType,
    plugin::ZarrVersion,
};

use crate::meta::types::TimeEncoding;

pub(crate) fn zarr_dtype_to_polars(
    zarr_dtype: &ZarrDataType,
    time_encoding: Option<&TimeEncoding>,
) -> PlDataType {
    if let Some(te) = time_encoding {
        return if te.is_duration {
            PlDataType::Duration(
                TimeUnit::Nanoseconds,
            )
        } else {
            PlDataType::Datetime(
                TimeUnit::Nanoseconds,
                None,
            )
        };
    }
    let binding = zarr_dtype
        .name(ZarrVersion::V3)
        .map(|s| s.to_owned())
        .unwrap_or_else(|| {
            Cow::Borrowed("binary")
        })
        .into_owned();
    let zarr_identifier = binding.as_str();

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
        "float16" | "bfloat16" => {
            PlDataType::Float32
        }
        "float32" => PlDataType::Float32,
        "float64" => PlDataType::Float64,
        "string" => PlDataType::String,
        _ => PlDataType::Binary,
    }
}
