/// Tools for working with opened arrays.
use std::sync::Arc;

use zarrs::array::Array;
use zarrs::storage::{
    AsyncReadableWritableListableStorageTraits,
    ReadableWritableListableStorageTraits,
};

use crate::reader::{
    ShardedCacheAsync, ShardedCacheSync,
};

/// An opened array with its sharded cache for sync access.
pub struct OpenedArraySync {
    pub array: Arc<Array<dyn ReadableWritableListableStorageTraits>>,
    pub cache: ShardedCacheSync,
}

impl OpenedArraySync {
    pub fn new(
        array: Arc<Array<dyn ReadableWritableListableStorageTraits>>,
        cache: ShardedCacheSync,
    ) -> Self {
        Self { array, cache }
    }
}

/// An opened array with its sharded cache for async access.
pub struct OpenedArrayAsync {
    pub array: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>,
    pub cache: Arc<ShardedCacheAsync>,
}

impl OpenedArrayAsync {
    pub fn new(
        array: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>,
        cache: Arc<ShardedCacheAsync>,
    ) -> Self {
        Self { array, cache }
    }
}
