use std::borrow::Cow;

use zarrs::array::{
    Array, ArrayShardedReadableExt,
    ArrayShardedReadableExtCache, CodecOptions,
};
use zarrs::plugin::ZarrVersion;

use crate::errors::{
    BackendError, BackendResult,
};
use crate::reader::ColumnData;

// Re-export the cache type for use in backends
pub(crate) use zarrs::array::ArrayShardedReadableExtCache as ShardedCacheSync;

/// Retrieve a chunk using the sharded-aware API.
///
/// This function works for both sharded and unsharded arrays.
/// For sharded arrays, the cache stores shard indexes to avoid
/// repeated retrieval and decoding.
///
/// The chunk indices should be inner chunk indices (from subchunk_grid).
pub(crate) fn retrieve_chunk(
    array: &Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
    cache: &ArrayShardedReadableExtCache,
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
                .retrieve_subchunk_opt::<Vec<bool>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        "int8" => Ok(ColumnData::I8(
            array
                .retrieve_subchunk_opt::<Vec<i8>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        "int16" => Ok(ColumnData::I16(
            array
                .retrieve_subchunk_opt::<Vec<i16>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        "int32" => Ok(ColumnData::I32(
            array
                .retrieve_subchunk_opt::<Vec<i32>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        "int64" => Ok(ColumnData::I64(
            array
                .retrieve_subchunk_opt::<Vec<i64>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        "uint8" => Ok(ColumnData::U8(
            array
                .retrieve_subchunk_opt::<Vec<u8>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        "uint16" => Ok(ColumnData::U16(
            array
                .retrieve_subchunk_opt::<Vec<u16>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        "uint32" => Ok(ColumnData::U32(
            array
                .retrieve_subchunk_opt::<Vec<u32>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        "uint64" => Ok(ColumnData::U64(
            array
                .retrieve_subchunk_opt::<Vec<u64>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        "float32" => Ok(ColumnData::F32(
            array
                .retrieve_subchunk_opt::<Vec<f32>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        "float64" => Ok(ColumnData::F64(
            array
                .retrieve_subchunk_opt::<Vec<f64>>(
                    cache, chunk, &options,
                )
                ?,
        )),
        other => Err(BackendError::other(format!(
            "unsupported zarr dtype: {other}"
        ))),
    }
}
