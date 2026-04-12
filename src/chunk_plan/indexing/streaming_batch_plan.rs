//! Join-closed streaming batches for multi-grid zarr reads.
//!
//! Batches are built so each unit contains every chunk read needed for
//! [`crate::shared::combine_chunk_dataframes`] to match a **slice** of the full-scan
//! merge on shared dimension columns (full outer join on the intersection of dim
//! columns present in every schema group).
//!
//! When no shared dimensions exist between groups, [`ScheduleBuilt::Legacy`] defers
//! to sequential per-group batching (see [`super::grid_execution`] streaming cut).

use std::collections::BTreeSet;

use crate::IStr;
use crate::meta::ZarrMeta;

use super::grid_execution::OwnedGridGroup;

/// Max chunk reads coalesced into one driver step before flushing (memory / parallelism).
pub(crate) const MAX_DRIVER_CHUNKS_COALESCED:
    usize = 100;

/// One physical chunk read: index into [`OwnedGridGroup::chunk_indices`].
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash,
)]
pub(crate) struct ChunkReadRef {
    pub group_idx: usize,
    pub chunk_slot: usize,
}

/// A join-closed set of reads for one emitted streaming batch.
#[derive(Clone, Debug, Default)]
pub(crate) struct StreamingBatch {
    pub reads: Vec<ChunkReadRef>,
}

/// Result of [`build_streaming_schedule`].
pub(crate) enum ScheduleBuilt {
    /// Use [`StreamingSchedule::JoinClosed`] with these batches.
    JoinClosed { batches: Vec<StreamingBatch> },
    /// Keep sequential iterator + optional streaming I/O cut.
    Legacy,
}

/// Distinct `(group_idx, chunk_slot)` touched at least once across batches.
pub(crate) fn distinct_chunk_slots_in_batches(
    batches: &[StreamingBatch],
) -> usize {
    let mut seen = BTreeSet::new();
    for b in batches {
        for r in &b.reads {
            seen.insert((
                r.group_idx,
                r.chunk_slot,
            ));
        }
    }
    seen.len()
}

/// Build join-closed batches when shared join dimensions exist; otherwise [`ScheduleBuilt::Legacy`].
pub(crate) fn build_streaming_schedule(
    groups: &[OwnedGridGroup],
    meta: &ZarrMeta,
    batch_size: usize,
) -> ScheduleBuilt {
    if groups.is_empty() {
        return ScheduleBuilt::JoinClosed {
            batches: vec![],
        };
    }
    if groups.len() == 1 {
        return ScheduleBuilt::JoinClosed {
            batches: build_single_group_batches(
                groups, batch_size,
            ),
        };
    }

    let join_dims =
        join_dimension_intersection(groups, meta);
    if join_dims.is_empty() {
        return ScheduleBuilt::Legacy;
    }

    let driver_idx = pick_driver_group(groups);
    ScheduleBuilt::JoinClosed {
        batches: build_multi_group_batches(
            groups, driver_idx, &join_dims,
            batch_size,
        ),
    }
}

fn join_dimension_intersection(
    groups: &[OwnedGridGroup],
    meta: &ZarrMeta,
) -> Vec<IStr> {
    let all: BTreeSet<IStr> = meta
        .dim_analysis
        .all_dims
        .iter()
        .cloned()
        .collect();
    let mut acc: Option<BTreeSet<IStr>> = None;
    for g in groups {
        let s: BTreeSet<IStr> = g
            .sig
            .dims()
            .iter()
            .filter(|d| all.contains(d))
            .cloned()
            .collect();
        acc = Some(match acc {
            None => s,
            Some(i) => i
                .intersection(&s)
                .cloned()
                .collect(),
        });
    }
    let inter = acc.unwrap_or_default();
    meta.dim_analysis
        .all_dims
        .iter()
        .filter(|d| inter.contains(d))
        .cloned()
        .collect()
}

fn pick_driver_group(
    groups: &[OwnedGridGroup],
) -> usize {
    groups
        .iter()
        .enumerate()
        .max_by_key(|(_, g)| {
            let nd = g.sig.dims().len();
            let nc = g.chunk_indices.len();
            (nd, nc)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn build_single_group_batches(
    groups: &[OwnedGridGroup],
    batch_size: usize,
) -> Vec<StreamingBatch> {
    let g = &groups[0];
    let mut out = Vec::new();
    let mut cur = StreamingBatch::default();
    let mut cur_rows = 0usize;

    for slot in 0..g.chunk_indices.len() {
        let rows = chunk_element_count(g, slot);
        if cur_rows > 0
            && cur_rows.saturating_add(rows)
                > batch_size
        {
            out.push(cur);
            cur = StreamingBatch::default();
            cur_rows = 0;
        }
        cur.reads.push(ChunkReadRef {
            group_idx: 0,
            chunk_slot: slot,
        });
        cur_rows = cur_rows.saturating_add(rows);

        if cur.reads.len()
            >= MAX_DRIVER_CHUNKS_COALESCED
        {
            out.push(cur);
            cur = StreamingBatch::default();
            cur_rows = 0;
        }
    }
    if !cur.reads.is_empty() {
        out.push(cur);
    }
    out
}

fn build_multi_group_batches(
    groups: &[OwnedGridGroup],
    driver_idx: usize,
    join_dims: &[IStr],
    batch_size: usize,
) -> Vec<StreamingBatch> {
    let driver = &groups[driver_idx];
    let mut batches = Vec::new();
    let mut acc_slots: Vec<usize> = Vec::new();
    let mut acc_rows = 0usize;

    for slot in 0..driver.chunk_indices.len() {
        let rows =
            chunk_element_count(driver, slot);
        let overflow_rows = acc_rows > 0
            && acc_rows.saturating_add(rows)
                > batch_size;
        let overflow_count = acc_slots.len()
            >= MAX_DRIVER_CHUNKS_COALESCED;
        if overflow_rows || overflow_count {
            if !acc_slots.is_empty() {
                batches.push(
                    batch_for_driver_slots(
                        groups, driver_idx,
                        &acc_slots, join_dims,
                    ),
                );
            }
            acc_slots.clear();
            acc_rows = 0;
        }
        acc_slots.push(slot);
        acc_rows = acc_rows.saturating_add(rows);
    }
    if !acc_slots.is_empty() {
        batches.push(batch_for_driver_slots(
            groups, driver_idx, &acc_slots,
            join_dims,
        ));
    }
    batches
}

fn batch_for_driver_slots(
    groups: &[OwnedGridGroup],
    driver_idx: usize,
    driver_slots: &[usize],
    join_dims: &[IStr],
) -> StreamingBatch {
    let mut reads = Vec::new();
    for &slot in driver_slots {
        reads.push(ChunkReadRef {
            group_idx: driver_idx,
            chunk_slot: slot,
        });
    }
    for (j, g) in groups.iter().enumerate() {
        if j == driver_idx {
            continue;
        }
        let mut covered: BTreeSet<usize> =
            BTreeSet::new();
        for &ds in driver_slots {
            for slot in 0..g.chunk_indices.len() {
                if chunks_overlap_on_join_dims(
                    &groups[driver_idx],
                    ds,
                    g,
                    slot,
                    join_dims,
                ) {
                    covered.insert(slot);
                }
            }
        }
        for slot in covered {
            reads.push(ChunkReadRef {
                group_idx: j,
                chunk_slot: slot,
            });
        }
    }
    StreamingBatch { reads }
}

fn dim_axis(
    sig_dims: &[IStr],
    dim: IStr,
) -> Option<usize> {
    sig_dims.iter().position(|d| *d == dim)
}

fn axis_interval(
    g: &OwnedGridGroup,
    slot: usize,
    axis: usize,
) -> (u64, u64) {
    let idx = g.chunk_indices[slot][axis];
    let cs = g.sig.chunk_shape()[axis];
    let alen = g.array_shape[axis];
    let start = idx * cs;
    let end = (start + cs).min(alen);
    (start, end)
}

fn chunks_overlap_on_join_dims(
    ga: &OwnedGridGroup,
    slot_a: usize,
    gb: &OwnedGridGroup,
    slot_b: usize,
    join_dims: &[IStr],
) -> bool {
    for dim in join_dims {
        let Some(ia) =
            dim_axis(ga.sig.dims(), *dim)
        else {
            return false;
        };
        let Some(ib) =
            dim_axis(gb.sig.dims(), *dim)
        else {
            return false;
        };
        let (sa, ea) =
            axis_interval(ga, slot_a, ia);
        let (sb, eb) =
            axis_interval(gb, slot_b, ib);
        if ea <= sb || eb <= sa {
            return false;
        }
    }
    true
}

fn chunk_element_count(
    g: &OwnedGridGroup,
    slot: usize,
) -> usize {
    let idx = &g.chunk_indices[slot];
    let cs = g.sig.chunk_shape();
    let a = &g.array_shape;
    idx.iter()
        .zip(cs.iter())
        .zip(a.iter())
        .map(|((&i, &csh), &alen)| {
            let start = i * csh;
            let end = (start + csh).min(alen);
            (end - start) as usize
        })
        .product::<usize>()
        .max(1)
}
