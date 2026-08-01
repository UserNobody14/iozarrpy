use zarrs::array::{
    Array, AsyncArrayShardedReadableExt,
    AsyncArrayShardedReadableExtCache,
    CodecOptions,
};

use crate::errors::BackendResult;
use crate::reader::ColumnData;
use crate::reader::column_data::DecodedChunk;

// Re-export the cache type for use in backends
pub(crate) use zarrs::array::AsyncArrayShardedReadableExtCache as ShardedCacheAsync;

/// Retrieve a chunk using the sharded-aware async API.
///
/// This function works for both sharded and unsharded arrays.
/// For sharded arrays, the cache stores shard indexes to avoid
/// repeated retrieval and decoding.
///
/// The chunk indices should be inner chunk indices (from subchunk_grid).
/// Decodes directly into [`ColumnData`] (Arrow-backed) via zarrs's
/// [`FromArrayBytes`](zarrs::array::FromArrayBytes) trait — no
/// `Vec<T>`-shaped intermediate.
pub(crate) async fn retrieve_chunk_async(
    array: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    cache: &AsyncArrayShardedReadableExtCache,
    chunk: &[u64],
) -> BackendResult<ColumnData> {
    let options = CodecOptions::default();
    let DecodedChunk(cd) = array
        .async_retrieve_subchunk_opt::<DecodedChunk>(
            cache, chunk, &options,
        )
        .await?;
    Ok(cd)
}
