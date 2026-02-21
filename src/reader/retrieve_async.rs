use std::borrow::Cow;

use zarrs::array::{
    Array, AsyncArrayShardedReadableExt,
    AsyncArrayShardedReadableExtCache,
    CodecOptions,
};
use zarrs::plugin::ZarrVersion;

use crate::errors::{
    BackendError, BackendResult,
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
/// The chunk indices should be inner chunk indices (from subchunk_grid).
pub(crate) async fn retrieve_chunk_async(
    array: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    cache: &AsyncArrayShardedReadableExtCache,
    chunk: &[u64],
) -> BackendResult<ColumnData> {
    let idv = array
        .data_type()
        .name(ZarrVersion::V3)
        .map(|s| s.to_owned())
        .unwrap_or_else(|| {
            Cow::Borrowed("binary")
        })
        .into_owned();
    let id = idv.as_str();
    let options = CodecOptions::default();
    match id {
        "bool" => Ok(ColumnData::Bool(
            array
                .async_retrieve_subchunk_opt::<Vec<bool>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        "int8" => Ok(ColumnData::I8(
            array
                .async_retrieve_subchunk_opt::<Vec<i8>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        "int16" => Ok(ColumnData::I16(
            array
                .async_retrieve_subchunk_opt::<Vec<i16>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        "int32" => Ok(ColumnData::I32(
            array
                .async_retrieve_subchunk_opt::<Vec<i32>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        "int64" => Ok(ColumnData::I64(
            array
                .async_retrieve_subchunk_opt::<Vec<i64>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        "uint8" => Ok(ColumnData::U8(
            array
                .async_retrieve_subchunk_opt::<Vec<u8>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        "uint16" => Ok(ColumnData::U16(
            array
                .async_retrieve_subchunk_opt::<Vec<u16>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        "uint32" => Ok(ColumnData::U32(
            array
                .async_retrieve_subchunk_opt::<Vec<u32>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        "uint64" => Ok(ColumnData::U64(
            array
                .async_retrieve_subchunk_opt::<Vec<u64>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        "float32" => Ok(ColumnData::F32(
            array
                .async_retrieve_subchunk_opt::<Vec<f32>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        "float64" => Ok(ColumnData::F64(
            array
                .async_retrieve_subchunk_opt::<Vec<f64>>(
                    cache, chunk, &options,
                )
                .await
                ?,
        )),
        other => {
            Err(BackendError::Other(format!(
                "unsupported zarr dtype: {other}"
            )))
        }
    }
}

fn to_string_err<E: std::fmt::Display>(
    e: E,
) -> String {
    e.to_string()
}
