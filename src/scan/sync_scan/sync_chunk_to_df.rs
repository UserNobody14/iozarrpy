//! Synchronous chunk-to-DataFrame conversion using generic backend traits.
//!
//! Each chunk produces a [`polars::prelude::DataFrame`] whose dim columns are
//! synthetic integer positions named `__<dim>` (the join key downstream),
//! plus one column per leaf variable with its raw decoded values. User-facing
//! `<dim>` coordinate values are not materialized here — they are joined in by
//! the [`crate::chunk_plan::indexing::GridJoinTree`]'s coord-broadcast leaves.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

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
    VarRead, build_coord_column, build_var_column,
    compute_in_bounds_mask, plan_var_reads,
};
use crate::shared::ChunkedDataBackendSync;
use crate::shared::IStr;

/// Convert a chunk to DataFrame using signature and grid info (sync).
///
/// An optional `chunk_subset` constrains which elements within the
/// chunk are included, avoiding unnecessary column-building work.
pub fn chunk_to_df_from_grid_with_backend<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    idx: Vec<u64>,
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
        &idx,
        chunk_shape,
        vars,
        with_columns,
    )?;

    let mut loaded: BTreeMap<
        IStr,
        Arc<ColumnData>,
    > = BTreeMap::new();
    for vr in &var_reads {
        if loaded.contains_key(&vr.path) {
            continue;
        }
        let data = backend.read_chunk_sync(
            &vr.path,
            &vr.indices,
        )?;
        loaded.insert(vr.path, data);
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
        let VarRead {
            name,
            path,
            var_dims,
            var_chunk_shape,
            offsets,
            ..
        } = vr;
        let data = loaded
            .get(path)
            .ok_or_else(|| BackendError::Other {
                msg: format!(
                    "internal: missing read for variable path {path}",
                ),
            })?
            .clone();
        let encoding = meta
            .array_by_path(*name)
            .and_then(|m| m.encoding.as_ref());
        cols.push(build_var_column(
            name,
            data,
            var_dims,
            var_chunk_shape,
            offsets,
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
