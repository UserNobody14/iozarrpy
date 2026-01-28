use std::sync::Arc;

use tokio::runtime::Runtime;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;
use zarrs::storage::storage_adapter::sync_to_async::SyncToAsyncSpawnBlocking;

pub(crate) struct TokioBlockOn(
    pub(crate) Arc<Runtime>,
);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(
        &self,
        future: F,
    ) -> F::Output {
        self.0.block_on(future)
    }
}

pub(crate) struct TokioSpawnBlocking;

impl SyncToAsyncSpawnBlocking
    for TokioSpawnBlocking
{
    fn spawn_blocking<F, R>(
        &self,
        f: F,
    ) -> impl core::future::Future<Output = R> + Send
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        async move {
            tokio::task::spawn_blocking(f)
                .await
                .expect("spawn_blocking failed")
        }
    }
}
