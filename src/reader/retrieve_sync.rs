use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;

use crate::reader::ColumnData;
use crate::reader::limits::max_chunk_elems;

pub(crate) fn retrieve_chunk(
    array: &Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
    chunk: &[u64],
) -> Result<ColumnData, String> {
    let id = array.data_type().identifier();
    match id {
        "bool" => Ok(ColumnData::Bool(
            array.retrieve_chunk::<Vec<bool>>(chunk).map_err(to_string_err)?,
        )),
        "int8" => Ok(ColumnData::I8(
            array.retrieve_chunk::<Vec<i8>>(chunk).map_err(to_string_err)?,
        )),
        "int16" => Ok(ColumnData::I16(
            array.retrieve_chunk::<Vec<i16>>(chunk).map_err(to_string_err)?,
        )),
        "int32" => Ok(ColumnData::I32(
            array.retrieve_chunk::<Vec<i32>>(chunk).map_err(to_string_err)?,
        )),
        "int64" => Ok(ColumnData::I64(
            array.retrieve_chunk::<Vec<i64>>(chunk).map_err(to_string_err)?,
        )),
        "uint8" => Ok(ColumnData::U8(
            array.retrieve_chunk::<Vec<u8>>(chunk).map_err(to_string_err)?,
        )),
        "uint16" => Ok(ColumnData::U16(
            array.retrieve_chunk::<Vec<u16>>(chunk).map_err(to_string_err)?,
        )),
        "uint32" => Ok(ColumnData::U32(
            array.retrieve_chunk::<Vec<u32>>(chunk).map_err(to_string_err)?,
        )),
        "uint64" => Ok(ColumnData::U64(
            array.retrieve_chunk::<Vec<u64>>(chunk).map_err(to_string_err)?,
        )),
        "float32" => Ok(ColumnData::F32(
            array.retrieve_chunk::<Vec<f32>>(chunk).map_err(to_string_err)?,
        )),
        "float64" => Ok(ColumnData::F64(
            array.retrieve_chunk::<Vec<f64>>(chunk).map_err(to_string_err)?,
        )),
        other => Err(format!("unsupported zarr dtype: {other}")),
    }
}

pub(crate) fn retrieve_1d_subset(
    array: &Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
    start: u64,
    len: u64,
) -> Result<ColumnData, String> {
    if len as u128 > max_chunk_elems() as u128 {
        return Err(
            "refusing to allocate extremely large coordinate subset; set RAINBEAR_MAX_CHUNK_ELEMS to override"
                .to_string(),
        );
    }
    let subset = ArraySubset::new_with_ranges(&[start..(start + len)]);
    let id = array.data_type().identifier();
    match id {
        "bool" => Ok(ColumnData::Bool(
            array
                .retrieve_array_subset::<Vec<bool>>(&subset)
                .map_err(to_string_err)?,
        )),
        "int8" => Ok(ColumnData::I8(
            array
                .retrieve_array_subset::<Vec<i8>>(&subset)
                .map_err(to_string_err)?,
        )),
        "int16" => Ok(ColumnData::I16(
            array
                .retrieve_array_subset::<Vec<i16>>(&subset)
                .map_err(to_string_err)?,
        )),
        "int32" => Ok(ColumnData::I32(
            array
                .retrieve_array_subset::<Vec<i32>>(&subset)
                .map_err(to_string_err)?,
        )),
        "int64" => Ok(ColumnData::I64(
            array
                .retrieve_array_subset::<Vec<i64>>(&subset)
                .map_err(to_string_err)?,
        )),
        "uint8" => Ok(ColumnData::U8(
            array
                .retrieve_array_subset::<Vec<u8>>(&subset)
                .map_err(to_string_err)?,
        )),
        "uint16" => Ok(ColumnData::U16(
            array
                .retrieve_array_subset::<Vec<u16>>(&subset)
                .map_err(to_string_err)?,
        )),
        "uint32" => Ok(ColumnData::U32(
            array
                .retrieve_array_subset::<Vec<u32>>(&subset)
                .map_err(to_string_err)?,
        )),
        "uint64" => Ok(ColumnData::U64(
            array
                .retrieve_array_subset::<Vec<u64>>(&subset)
                .map_err(to_string_err)?,
        )),
        "float32" => Ok(ColumnData::F32(
            array
                .retrieve_array_subset::<Vec<f32>>(&subset)
                .map_err(to_string_err)?,
        )),
        "float64" => Ok(ColumnData::F64(
            array
                .retrieve_array_subset::<Vec<f64>>(&subset)
                .map_err(to_string_err)?,
        )),
        other => Err(format!("unsupported zarr dtype: {other}")),
    }
}

fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

