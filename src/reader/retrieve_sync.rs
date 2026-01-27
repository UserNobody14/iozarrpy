use zarrs::array::Array;

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
    if len == 0 {
        let id = array.data_type().identifier();
        return ColumnData::empty_for_dtype(id)
            .ok_or_else(|| format!("unsupported zarr dtype: {id}"));
    }

    // Get chunk grid info for dimension 0 (1D array).
    let chunk_shape = array.chunk_shape(&[0]).map_err(to_string_err)?;
    let chunk_size = chunk_shape.first().map(|s| s.get()).unwrap_or(len);

    let end = start + len;
    let first_chunk = start / chunk_size;
    let last_chunk = (end - 1) / chunk_size;

    let id = array.data_type().identifier();
    let mut result =
        ColumnData::empty_for_dtype(id).ok_or_else(|| format!("unsupported zarr dtype: {id}"))?;

    for chunk_idx in first_chunk..=last_chunk {
        let chunk_data = retrieve_chunk(array, &[chunk_idx])?;
        let chunk_start = chunk_idx * chunk_size;
        let chunk_end = chunk_start + chunk_data.len() as u64;

        // Calculate slice within this chunk.
        let slice_start = if start > chunk_start {
            (start - chunk_start) as usize
        } else {
            0
        };
        let slice_end = if end < chunk_end {
            (end - chunk_start) as usize
        } else {
            chunk_data.len()
        };

        if slice_start == 0 && slice_end == chunk_data.len() {
            // Take the whole chunk.
            result.extend(chunk_data);
        } else {
            // Take a slice.
            result.extend(chunk_data.slice(slice_start, slice_end - slice_start));
        }
    }

    Ok(result)
}

/// Retrieve a subset of an array (arbitrary region, not aligned to chunks).
pub(crate) fn retrieve_array_subset(
    array: &Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
    subset: &zarrs::array_subset::ArraySubset,
) -> Result<ColumnData, String> {
    let num_elems = subset.num_elements_usize();
    if num_elems as u128 > max_chunk_elems() as u128 {
        return Err(
            "refusing to allocate extremely large array subset; set RAINBEAR_MAX_CHUNK_ELEMS to override"
                .to_string(),
        );
    }
    if num_elems == 0 {
        let id = array.data_type().identifier();
        return ColumnData::empty_for_dtype(id)
            .ok_or_else(|| format!("unsupported zarr dtype: {id}"));
    }

    let id = array.data_type().identifier();
    match id {
        "bool" => Ok(ColumnData::Bool(
            array.retrieve_array_subset::<Vec<bool>>(subset).map_err(to_string_err)?,
        )),
        "int8" => Ok(ColumnData::I8(
            array.retrieve_array_subset::<Vec<i8>>(subset).map_err(to_string_err)?,
        )),
        "int16" => Ok(ColumnData::I16(
            array.retrieve_array_subset::<Vec<i16>>(subset).map_err(to_string_err)?,
        )),
        "int32" => Ok(ColumnData::I32(
            array.retrieve_array_subset::<Vec<i32>>(subset).map_err(to_string_err)?,
        )),
        "int64" => Ok(ColumnData::I64(
            array.retrieve_array_subset::<Vec<i64>>(subset).map_err(to_string_err)?,
        )),
        "uint8" => Ok(ColumnData::U8(
            array.retrieve_array_subset::<Vec<u8>>(subset).map_err(to_string_err)?,
        )),
        "uint16" => Ok(ColumnData::U16(
            array.retrieve_array_subset::<Vec<u16>>(subset).map_err(to_string_err)?,
        )),
        "uint32" => Ok(ColumnData::U32(
            array.retrieve_array_subset::<Vec<u32>>(subset).map_err(to_string_err)?,
        )),
        "uint64" => Ok(ColumnData::U64(
            array.retrieve_array_subset::<Vec<u64>>(subset).map_err(to_string_err)?,
        )),
        "float32" => Ok(ColumnData::F32(
            array.retrieve_array_subset::<Vec<f32>>(subset).map_err(to_string_err)?,
        )),
        "float64" => Ok(ColumnData::F64(
            array.retrieve_array_subset::<Vec<f64>>(subset).map_err(to_string_err)?,
        )),
        other => Err(format!("unsupported zarr dtype: {other}")),
    }
}

fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

