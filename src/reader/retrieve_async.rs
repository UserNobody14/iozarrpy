use zarrs::array::codec::CodecOptions;
use zarrs::array::{
    Array, AsyncArrayShardedReadableExt,
    AsyncArrayShardedReadableExtCache,
};

use crate::reader::ColumnData;

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

fn to_string_err<E: std::fmt::Display>(
    e: E,
) -> String {
    e.to_string()
}
