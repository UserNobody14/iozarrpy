//! Inline per-chunk read planner used by both sync and async chunk-to-df paths.
//!
//! For one chunk of a primary [`crate::chunk_plan::ChunkGridSignature`], plan
//! the (deduped) list of variable chunk reads needed to materialize that chunk's
//! [`polars::prelude::DataFrame`]. Dim columns are always synthetic integer
//! positions (no per-chunk coordinate-array reads); user-facing `<dim>` values
//! come from explicit 1D dim-coord leaves joined onto the data leaves via the
//! [`crate::chunk_plan::indexing::GridJoinTree`].

use std::collections::BTreeSet;

use crate::errors::{
    BackendError, BackendResult,
};
use crate::meta::ZarrMeta;
use crate::scan::shared::{
    compute_var_chunk_indices,
    should_include_column,
};
use crate::shared::IStr;

/// One physical zarr chunk read for a single variable in a leaf's chunk plan.
#[derive(Debug, Clone)]
pub struct VarRead {
    pub name: IStr,
    pub path: IStr,
    pub indices: Vec<u64>,
    pub var_dims: Vec<IStr>,
    pub var_chunk_shape: Vec<u64>,
    pub offsets: Vec<u64>,
}

/// Plan the variable reads for one chunk of a primary grid signature.
///
/// `vars` is the leaf's variable list (already pre-filtered to share the same
/// signature). Each var's chunk indices are computed from `primary_idx` via
/// [`compute_var_chunk_indices`] for the rare case where var dims differ from
/// primary dims (typically same-shape; the fast path applies).
///
/// `with_columns`, when `Some`, drops vars not in the projection. Vars whose
/// names happen to match a primary dim (1D dim-coord leaves) are **not**
/// skipped: their values are needed to populate the user-facing `<dim>` column.
pub fn plan_var_reads(
    meta: &ZarrMeta,
    primary_dims: &[IStr],
    primary_idx: &[u64],
    primary_chunk_shape: &[u64],
    vars: &[IStr],
    with_columns: Option<&BTreeSet<IStr>>,
) -> BackendResult<Vec<VarRead>> {
    let mut out: Vec<VarRead> =
        Vec::with_capacity(vars.len());
    let mut seen: BTreeSet<IStr> =
        BTreeSet::new();
    for &name in vars {
        if !seen.insert(name) {
            continue;
        }
        if !should_include_column(
            &name,
            with_columns,
        ) {
            continue;
        }
        let Some(var_meta) =
            meta.array_by_path(name)
        else {
            return Err(
                BackendError::UnknownDataVar {
                    name,
                    available_vars: meta
                        .all_data_var_paths(),
                },
            );
        };
        let var_dims: Vec<IStr> = var_meta
            .dims
            .iter()
            .cloned()
            .collect();
        let (chunk_indices, offsets) =
            compute_var_chunk_indices(
                primary_idx,
                primary_chunk_shape,
                primary_dims,
                &var_dims,
                &var_meta.chunk_shape,
                &var_meta.shape,
            );
        out.push(VarRead {
            name,
            path: var_meta.path,
            indices: chunk_indices,
            var_dims,
            var_chunk_shape: var_meta
                .chunk_shape
                .to_vec(),
            offsets,
        });
    }
    Ok(out)
}
