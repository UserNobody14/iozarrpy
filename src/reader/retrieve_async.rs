use zarrs::array::codec::CodecOptions;
use zarrs::array::{
    Array, AsyncArrayShardedReadableExt,
    AsyncArrayShardedReadableExtCache,
};

use crate::reader::ColumnData;
use crate::reader::limits::max_chunk_elems;

// Re-export the cache type for use in backends
pub(crate) use zarrs::array::AsyncArrayShardedReadableExtCache as ShardedCacheAsync;

/// Retrieve a chunk using the sharded-aware async API.
///
/// This function works for both sharded and unsharded arrays.
/// For sharded arrays, the cache stores shard indexes to avoid
/// repeated retrieval and decoding.
///
/// The chunk indices should be inner chunk indices (from inner_chunk_grid).
pub(crate) async fn retrieve_chunk_async(
    array: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    cache: &AsyncArrayShardedReadableExtCache,
    chunk: &[u64],
) -> Result<ColumnData, String> {
    let id = array.data_type().identifier();
    let options = CodecOptions::default();
    match id {
        "bool" => Ok(ColumnData::Bool(
            array
                .async_retrieve_inner_chunk_opt::<Vec<bool>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        "int8" => Ok(ColumnData::I8(
            array
                .async_retrieve_inner_chunk_opt::<Vec<i8>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        "int16" => Ok(ColumnData::I16(
            array
                .async_retrieve_inner_chunk_opt::<Vec<i16>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        "int32" => Ok(ColumnData::I32(
            array
                .async_retrieve_inner_chunk_opt::<Vec<i32>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        "int64" => Ok(ColumnData::I64(
            array
                .async_retrieve_inner_chunk_opt::<Vec<i64>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        "uint8" => Ok(ColumnData::U8(
            array
                .async_retrieve_inner_chunk_opt::<Vec<u8>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        "uint16" => Ok(ColumnData::U16(
            array
                .async_retrieve_inner_chunk_opt::<Vec<u16>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        "uint32" => Ok(ColumnData::U32(
            array
                .async_retrieve_inner_chunk_opt::<Vec<u32>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        "uint64" => Ok(ColumnData::U64(
            array
                .async_retrieve_inner_chunk_opt::<Vec<u64>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        "float32" => Ok(ColumnData::F32(
            array
                .async_retrieve_inner_chunk_opt::<Vec<f32>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        "float64" => Ok(ColumnData::F64(
            array
                .async_retrieve_inner_chunk_opt::<Vec<f64>>(
                    cache, chunk, &options,
                )
                .await
                .map_err(to_string_err)?,
        )),
        other => {
            Err(format!("unsupported zarr dtype: {other}"))
        }
    }
}

pub(crate) async fn retrieve_1d_subset_async(
    array: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    cache: &AsyncArrayShardedReadableExtCache,
    start: u64,
    len: u64,
) -> Result<ColumnData, String> {
    use zarrs::array::ArrayShardedExt;

    if len as u128 > max_chunk_elems() as u128 {
        return Err(
            "refusing to allocate extremely large coordinate subset; set RAINBEAR_MAX_CHUNK_ELEMS to override"
                .to_string(),
        );
    }
    if len == 0 {
        let id = array.data_type().identifier();
        return ColumnData::empty_for_dtype(id)
            .ok_or_else(|| {
                format!(
                    "unsupported zarr dtype: {id}"
                )
            });
    }

    // Use inner_chunk_grid for sharded arrays
    let inner_grid = array.inner_chunk_grid();
    let chunk_shape = inner_grid
        .chunk_shape_u64(&[0])
        .map_err(to_string_err)?
        .ok_or_else(|| {
            "could not determine chunk shape"
                .to_string()
        })?;
    let chunk_size = chunk_shape
        .first()
        .copied()
        .unwrap_or(len);

    let end = start + len;
    let first_chunk = start / chunk_size;
    let last_chunk = (end - 1) / chunk_size;

    let id = array.data_type().identifier();
    let mut result =
        ColumnData::empty_for_dtype(id)
            .ok_or_else(|| {
                format!(
                    "unsupported zarr dtype: {id}"
                )
            })?;

    for chunk_idx in first_chunk..=last_chunk {
        let chunk_data = retrieve_chunk_async(
            array,
            cache,
            &[chunk_idx],
        )
        .await?;
        let chunk_start = chunk_idx * chunk_size;
        let chunk_end =
            chunk_start + chunk_data.len() as u64;

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

        if slice_start == 0
            && slice_end == chunk_data.len()
        {
            // Take the whole chunk.
            result.extend(chunk_data);
        } else {
            // Take a slice.
            result.extend(chunk_data.slice(
                slice_start,
                slice_end - slice_start,
            ));
        }
    }

    Ok(result)
}

fn to_string_err<E: std::fmt::Display>(
    e: E,
) -> String {
    e.to_string()
}
