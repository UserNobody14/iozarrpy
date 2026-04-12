//! Consolidated chunk groups ready for read/combine.
//!
//! After [`super::plan::GroupedChunkPlan::iter_consolidated_chunks`], callers apply
//! [`GridGroupExecutionOpts`]: drop redundant 1D coordinate-only grids (all modes),
//! and optionally narrow groups for **streaming batch** I/O.
//!
//! ## When `streaming_batch_io_cut` applies
//!
//! [`build_streaming_schedule`](super::streaming_batch_plan::build_streaming_schedule)
//! returns [`ScheduleBuilt::JoinClosed`] when every batch can include aligned chunk
//! reads across groups (shared join dimensions). Then iterators **skip** this cut so
//! auxiliary grids (e.g. `station_id` on `point`) stay available for the same merge as
//! full scans.
//!
//! For [`ScheduleBuilt::Legacy`] (no shared dimensions between groups, diagonal merge),
//! batches are still built sequentially; the cut narrows groups so a batch never
//! contains only a slice that cannot satisfy the Polars predicate in isolation.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::plan::GroupedChunkPlan;
use crate::IStr;
use crate::chunk_plan::ChunkSubset;
use crate::chunk_plan::ConsolidatedGridGroup;
use crate::errors::BackendResult;
use crate::meta::ZarrMeta;
use crate::scan::column_policy::group_supplies_array_or_1d_enrichable;

/// Owned version of [`ConsolidatedGridGroup`]: the plan is often dropped while
/// streaming iteration continues.
pub(crate) struct OwnedGridGroup {
    pub sig: Arc<
        crate::chunk_plan::ChunkGridSignature,
    >,
    pub vars: Vec<IStr>,
    pub chunk_indices: Vec<Vec<u64>>,
    pub chunk_subsets: Vec<Option<ChunkSubset>>,
    pub array_shape: Vec<u64>,
}

impl OwnedGridGroup {
    #[inline]
    pub(crate) fn from_consolidated(
        group: ConsolidatedGridGroup<'_>,
    ) -> Self {
        Self {
            sig: Arc::new(group.sig.clone()),
            vars: group.vars,
            chunk_indices: group.chunk_indices,
            chunk_subsets: group.chunk_subsets,
            array_shape: group.array_shape,
        }
    }
}

/// Predicate / projection narrowing for **streaming** batch reads only.
pub(crate) struct StreamingBatchIoCut<'a> {
    pub predicate_refs: &'a BTreeSet<IStr>,
    pub with_columns: Option<&'a BTreeSet<IStr>>,
}

/// Post-plan filters before issuing chunk reads.
pub(crate) struct GridGroupExecutionOpts<'a> {
    /// When true, discard all groups (predicate is a literal `false`).
    pub literal_false_clear: bool,
    /// Drop 1D coord-only grids redundant with a multi-dim data grid (sync + streaming).
    pub drop_redundant_1d_coords: bool,
    /// If set, drop groups that cannot supply predicate / output columns in isolation.
    pub streaming_batch_io_cut:
        Option<StreamingBatchIoCut<'a>>,
}

impl GroupedChunkPlan {
    /// Consolidated groups plus execution-time filtering (redundant coords, streaming cut).
    pub(crate) fn owned_grid_groups_for_io(
        &self,
        meta: &ZarrMeta,
        opts: GridGroupExecutionOpts<'_>,
    ) -> BackendResult<Vec<OwnedGridGroup>> {
        let mut groups: Vec<OwnedGridGroup> = self
            .iter_consolidated_chunks()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(OwnedGridGroup::from_consolidated)
            .collect();

        if opts.literal_false_clear {
            groups.clear();
            return Ok(groups);
        }
        if opts.drop_redundant_1d_coords {
            groups =
                filter_redundant_coord_only_groups(groups, meta);
        }
        if let Some(cut) =
            opts.streaming_batch_io_cut
        {
            groups = apply_streaming_batch_io_cut(
                groups,
                cut.predicate_refs,
                cut.with_columns,
                meta,
            );
        }
        Ok(groups)
    }
}

pub(crate) fn streaming_grid_chunk_read_count(
    groups: &[OwnedGridGroup],
) -> usize {
    groups
        .iter()
        .map(|g| g.chunk_indices.len())
        .sum()
}

fn array_dims_match_signature(
    am: &crate::meta::ZarrArrayMeta,
    sig_dims: &[IStr],
) -> bool {
    if am.shape.len() < 2
        || am.dims.len() != am.shape.len()
        || am.dims.len() != sig_dims.len()
    {
        return false;
    }
    let sig_set: BTreeSet<IStr> =
        sig_dims.iter().cloned().collect();
    let var_set: BTreeSet<IStr> =
        am.dims.iter().cloned().collect();
    sig_set == var_set
}

fn filter_redundant_coord_only_groups(
    groups: Vec<OwnedGridGroup>,
    meta: &ZarrMeta,
) -> Vec<OwnedGridGroup> {
    let n = groups.len();
    let drop: Vec<bool> = (0..n)
        .map(|i| {
            let g = &groups[i];
            let dims = g.sig.dims();
            if dims.len() != 1 {
                return false;
            }
            let d = dims[0];
            if g.vars.len() != 1 || g.vars[0] != d {
                return false;
            }
            groups.iter().enumerate().any(|(j, other)| {
                if j == i {
                    return false;
                }
                let od = other.sig.dims();
                if od.len() < 2 || !od.iter().any(|x| *x == d) {
                    return false;
                }
                other.vars.iter().any(|v| {
                    meta.array_by_path(*v)
                        .map(|am| {
                            array_dims_match_signature(
                                &am, od,
                            )
                        })
                        .unwrap_or(false)
                })
            })
        })
        .collect();

    groups
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop[*i])
        .map(|(_, g)| g)
        .collect()
}

/// Narrow groups for legacy streaming when batches are not join-closed.
pub(crate) fn apply_streaming_batch_io_cut(
    groups: Vec<OwnedGridGroup>,
    predicate_refs: &BTreeSet<IStr>,
    output_columns: Option<&BTreeSet<IStr>>,
    meta: &ZarrMeta,
) -> Vec<OwnedGridGroup> {
    let all_dims: BTreeSet<IStr> = meta
        .dim_analysis
        .all_dims
        .iter()
        .cloned()
        .collect();

    groups
        .into_iter()
        .filter(|g| {
            let sig_dims: BTreeSet<IStr> =
                g.sig.dims().iter().cloned().collect();
            let vars: BTreeSet<IStr> =
                g.vars.iter().cloned().collect();

            for c in predicate_refs {
                if all_dims.contains(c) {
                    if !sig_dims.contains(c) {
                        return false;
                    }
                } else if meta.array_by_path(c).is_some()
                    && !group_supplies_array_or_1d_enrichable(
                        c,
                        &sig_dims,
                        &vars,
                        meta,
                    )
                {
                    return false;
                }
            }

            if let Some(out) = output_columns {
                for name in out {
                    if all_dims.contains(name) {
                        continue;
                    }
                    if meta.array_by_path(name).is_none() {
                        continue;
                    }
                    if !group_supplies_array_or_1d_enrichable(
                        name,
                        &sig_dims,
                        &vars,
                        meta,
                    ) {
                        return false;
                    }
                }
            }

            true
        })
        .collect()
}
