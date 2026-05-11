//! Raw per-chunk read used by the join-aware batch assembler.
//!
//! Returns the per-var raw [`crate::reader::ColumnData`] (full chunk shape, in
//! C-order over `var_chunk_shape`) plus the [`VarRead`] descriptors that name
//! the dim mapping back to the leaf's primary signature. The
//! [`crate::chunk_plan::indexing::joined_assembly`] layer scatters these
//! directly into the joined output frame; no per-chunk DataFrame and no
//! synthetic `__<dim>` keys are materialized.
//!
//! Sync and async paths share the read planning / dedupe / assembly logic;
//! only the I/O dispatch (the closure that fetches each var path) differs.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::chunk_plan::ChunkGridSignature;
use crate::errors::{
    BackendError, BackendResult,
};
use crate::meta::ZarrMeta;
use crate::reader::ColumnData;
use crate::scan::shared::{
    chunk_read_plan::VarRead, plan_var_reads,
};
use crate::shared::IStr;

/// Output of a single raw chunk read for one leaf.
pub struct RawChunkRead {
    /// Var reads in the order required by the leaf, after `with_columns`
    /// filtering. Carries dim mapping (`var_dims`, `var_chunk_shape`,
    /// `offsets`) needed to project the chunk's element data into the
    /// leaf's primary dim space.
    pub var_reads: Vec<VarRead>,
    /// Raw column data per var, parallel to `var_reads`. Multiple `VarRead`s
    /// that share a `path` share the same `Arc`.
    pub var_data: Vec<Arc<ColumnData>>,
}

/// Plan reads for one chunk and fetch each unique var path via `read`.
///
/// Generic over a `read` closure so sync and async backends share the
/// dedupe-by-path bookkeeping. The closure returns `Result<Arc<ColumnData>, _>`
/// for one (path, indices) pair; the caller decides sync/async dispatch.
pub fn assemble_raw_chunk<F>(
    sig: &ChunkGridSignature,
    idx: &[u64],
    vars: &[IStr],
    with_columns: Option<&BTreeSet<IStr>>,
    meta: &ZarrMeta,
    mut read: F,
) -> BackendResult<RawChunkRead>
where
    F: FnMut(
        IStr,
        &[u64],
    )
        -> BackendResult<Arc<ColumnData>>,
{
    let var_reads = plan_var_reads(
        meta,
        sig.dims(),
        idx,
        sig.retrieval_shape(),
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
        let data = read(vr.path, &vr.indices)?;
        loaded.insert(vr.path, data);
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
