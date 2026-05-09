//! Async chunk-to-DataFrame conversion using generic backend traits.
//!
//! See [`super::super::sync_scan::sync_chunk_to_df`] for the column-shape
//! invariants — the async path is structurally identical, only differing in
//! how the per-var chunk reads are dispatched.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use futures::stream::{
    FuturesUnordered, StreamExt,
};
use polars::prelude::*;
use snafu::ResultExt;

use crate::chunk_plan::indexing::grid_join_reader::synthetic_dim_key;
use crate::chunk_plan::{
    ChunkGridSignature, ChunkSubset,
};
use crate::errors::{
    BackendError, BackendResult, PolarsSnafu,
};
use crate::meta::ZarrMeta;
use crate::reader::{
    ColumnData, checked_chunk_len,
    compute_strides,
};
use crate::scan::shared::{
    build_coord_column, build_var_column,
    compute_in_bounds_mask, plan_var_reads,
};
use crate::shared::ChunkedDataBackendAsync;
use crate::shared::IStr;

/// Convert a chunk to DataFrame using signature and grid info (async).
///
/// An optional `chunk_subset` constrains which elements within the
/// chunk are included, avoiding unnecessary column-building work.
pub async fn chunk_to_df_from_grid_with_backend<
    B: ChunkedDataBackendAsync,
>(
    backend: &B,
    idx: &[u64],
    sig: &ChunkGridSignature,
    array_shape: &[u64],
    vars: &[IStr],
    with_columns: Option<&BTreeSet<IStr>>,
    chunk_subset: Option<&ChunkSubset>,
    meta: &ZarrMeta,
) -> BackendResult<DataFrame> {
    let chunk_shape = sig.retrieval_shape();
    let dims = sig.dims();

    let origin: Vec<u64> = idx
        .iter()
        .zip(chunk_shape.iter())
        .map(|(i, s)| i * s)
        .collect();

    let chunk_len =
        checked_chunk_len(chunk_shape)?;
    let strides = compute_strides(chunk_shape);

    let keep = compute_in_bounds_mask(
        chunk_len,
        chunk_shape,
        &origin,
        array_shape,
        &strides,
        chunk_subset,
    );

    let var_reads = plan_var_reads(
        meta,
        dims,
        idx,
        chunk_shape,
        vars,
        with_columns,
    )?;

    // Dedupe by path (a leaf may legitimately list the same array twice if its
    // var grouping unifies aliases).
    let mut unique_paths: BTreeMap<
        IStr,
        Vec<u64>,
    > = BTreeMap::new();
    for vr in &var_reads {
        unique_paths
            .entry(vr.path)
            .or_insert_with(|| {
                vr.indices.clone()
            });
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

    let mut cols: Vec<Column> =
        Vec::with_capacity(
            dims.len() + var_reads.len(),
        );
    let height = keep.len();

    for (dim_idx, dim_name) in
        dims.iter().enumerate()
    {
        let key = synthetic_dim_key(*dim_name);
        cols.push(build_coord_column(
            key.as_str(),
            dim_idx,
            &keep,
            &strides,
            chunk_shape,
            &origin,
            None,
            None,
        ));
    }

    for vr in &var_reads {
        let data = loaded
            .get(&vr.path)
            .ok_or_else(|| BackendError::Other {
                msg: format!(
                    "internal: missing read for variable path {}",
                    vr.path
                ),
            })?;
        let encoding = meta
            .array_by_path(vr.name)
            .and_then(|m| m.encoding.as_ref());
        cols.push(build_var_column(
            &vr.name,
            data,
            &vr.var_dims,
            &vr.var_chunk_shape,
            &vr.offsets,
            dims,
            chunk_shape,
            &strides,
            &keep,
            encoding,
        ));
    }

    DataFrame::new(height, cols).context(
        PolarsSnafu {
            message: "Error creating DataFrame"
                .to_string(),
        },
    )
}
