//! Synchronous raw chunk reader.
//!
//! Returns the per-var raw [`crate::reader::ColumnData`] (full chunk shape, in
//! C-order over `var_chunk_shape`) plus [`crate::scan::shared::chunk_read_plan::VarRead`]
//! descriptors. The
//! [`crate::chunk_plan::indexing::joined_assembly`] layer scatters this directly
//! into the joined output frame; no per-chunk DataFrame is materialized.

use crate::chunk_plan::ChunkGridSignature;
use crate::errors::BackendResult;
use crate::meta::ZarrMeta;
use crate::scan::shared::{
    RawChunkRead, assemble_raw_chunk,
};
use crate::shared::ChunkedDataBackendSync;
use crate::shared::IStr;

/// Read one chunk's raw per-var data (sync).
///
/// `with_columns`, when `Some`, drops vars not in the projection.
pub fn read_chunk_raw_from_grid_with_backend<
    B: ChunkedDataBackendSync,
>(
    backend: &B,
    idx: &[u64],
    sig: &ChunkGridSignature,
    vars: &[IStr],
    with_columns: Option<
        &std::collections::BTreeSet<IStr>,
    >,
    meta: &ZarrMeta,
) -> BackendResult<RawChunkRead> {
    assemble_raw_chunk(
        sig,
        idx,
        vars,
        with_columns,
        meta,
        |path, indices| {
            backend.read_chunk_sync(&path, indices)
        },
    )
}
