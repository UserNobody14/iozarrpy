//! Async raw chunk reader.
//!
//! See [`super::super::sync_scan::sync_chunk_to_df`] for the column-shape
//! invariants — the async path is structurally identical, only differing in
//! how the per-var chunk reads are dispatched (concurrent
//! `read_chunk_async` calls deduped by zarr path).

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use futures::stream::{
    FuturesUnordered, StreamExt,
};

use crate::chunk_plan::ChunkGridSignature;
use crate::errors::{
    BackendError, BackendResult,
};
use crate::meta::ZarrMeta;
use crate::reader::ColumnData;
use crate::scan::shared::{
    RawChunkRead, plan_var_reads,
};
use crate::shared::ChunkedDataBackendAsync;
use crate::shared::IStr;

/// Read one chunk's raw per-var data (async).
///
/// Reads each unique var path concurrently via the backend's async dispatch,
/// then deduplicates back into the leaf's var order.
pub async fn read_chunk_raw_from_grid_with_backend<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    idx: &[u64],
    sig: &ChunkGridSignature,
    vars: &[IStr],
    with_columns: Option<&BTreeSet<IStr>>,
    meta: &ZarrMeta,
) -> BackendResult<RawChunkRead> {
    let var_reads = plan_var_reads(
        meta,
        sig.dims(),
        idx,
        sig.retrieval_shape(),
        vars,
        with_columns,
    )?;

    let mut unique_paths: BTreeMap<
        IStr,
        Vec<u64>,
    > = BTreeMap::new();
    for vr in &var_reads {
        unique_paths
            .entry(vr.path)
            .or_insert_with(|| vr.indices.clone());
    }

    let mut read_futs = FuturesUnordered::new();
    for (path, indices) in &unique_paths {
        let path = *path;
        let indices = indices.clone();
        read_futs.push(async move {
            let data = backend
                .read_chunk_async(&path, &indices)
                .await?;
            Ok::<_, BackendError>((path, data))
        });
    }

    let mut loaded: BTreeMap<
        IStr,
        Arc<ColumnData>,
    > = BTreeMap::new();
    while let Some(res) = read_futs.next().await {
        let (path, data) = res?;
        loaded.insert(path, data);
    }

    let var_data: Vec<Arc<ColumnData>> = var_reads
        .iter()
        .map(|vr| {
            loaded.get(&vr.path).cloned().ok_or_else(
                || BackendError::Other {
                    msg: format!(
                        "internal: missing read for variable path {}",
                        vr.path
                    ),
                },
            )
        })
        .collect::<BackendResult<Vec<_>>>()?;

    Ok(RawChunkRead {
        var_reads,
        var_data,
    })
}
