//! Tree-driven streaming reader: turns a [`GridJoinTree`] into a sequence of
//! join-closed `DataFrame` batches.
//!
//! The reader unifies the eager and streaming code paths. Each batch is
//! produced by walking the tree:
//!
//! - [`GridJoinTree::Leaf`]: the leaf's chunk indices are walked in row-budget
//!   bounded slabs; a slab's chunks are read in parallel and `vstack`ed.
//! - [`GridJoinTree::Join`]: a "driver" leaf is picked (largest by chunk
//!   count); its chunks drive the slab pace. For every other subtree we collect
//!   the chunks that overlap on the join dims, recursively read each subtree's
//!   slab, then full-outer-join the per-subtree DataFrames on the join dim
//!   columns. Coalescing of duplicate keys keeps the result schema clean.
//! - [`GridJoinTree::Independent`]: subtrees are processed sequentially. Each
//!   subtree yields its own batches; `diagonal_concat` is applied at the end of
//!   the eager path (streaming yields one batch at a time).

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;
use snafu::ResultExt;

use super::grid_join_tree::GridJoinTree;
use super::plan::{ChunkSubset, OwnedGridGroup};
use crate::chunk_plan::ChunkGridSignature;
use crate::errors::{
    BackendError, BackendResult, PolarsSnafu,
};
use crate::shared::{FromManyIstrs, IStr};

/// Chunks read in one slab of one leaf.
#[derive(Clone, Debug)]
pub struct LeafSlab {
    pub leaf_idx: usize,
    pub chunk_slots: Vec<usize>,
}

/// One batch's worth of leaf reads.
#[derive(Clone, Debug, Default)]
pub struct ReaderBatch {
    pub slabs: Vec<LeafSlab>,
}

/// A flat plan describing what to read for a single batch and how to combine
/// the resulting per-leaf DataFrames into one output DataFrame.
#[derive(Clone, Debug)]
pub struct BatchPlan {
    pub batch: ReaderBatch,
    pub combine: CombineNode,
}

/// Recursive recipe for combining per-leaf DataFrames inside one batch.
#[derive(Clone, Debug)]
pub enum CombineNode {
    /// Take the DataFrame for `leaf_idx` as-is.
    Leaf { leaf_idx: usize },
    /// Full-outer-join children on the named columns (with key coalescing).
    Join {
        join_dims: Vec<IStr>,
        children: Vec<CombineNode>,
    },
    /// Diagonal-concatenate children (no shared keys).
    Concat { children: Vec<CombineNode> },
}

/// Walk a [`GridJoinTree`] and emit a stream of [`BatchPlan`]s sized by
/// `batch_size`. Every batch contains slabs from **all** subtrees that still
/// have data, so predicate filtering on any column produced by the tree always
/// sees a populated DataFrame.
pub struct BatchPlanner<'a> {
    leaves: Vec<&'a OwnedGridGroup>,
    /// Recursive schedule mirroring the tree shape.
    root: ScheduleNode,
}

/// Recursive schedule node mirroring [`GridJoinTree`]. Carries per-node cursors
/// so successive [`BatchPlanner::next_batch`] calls advance the right leaves.
#[derive(Debug)]
enum ScheduleNode {
    /// A standalone leaf (no surrounding join). Walks its own chunk indices.
    Leaf { leaf_idx: usize, cursor: usize },
    /// A join-closed subtree. Driven by `driver_leaf`; siblings project on
    /// `join_axes_per_participant` (parallel arrays with `participants`).
    JoinClosed {
        combine: CombineNode,
        driver_leaf: usize,
        driver_cursor: usize,
        participants: Vec<usize>,
        join_axes_per_participant:
            Vec<Vec<usize>>,
    },
    /// Independent subtrees — every child contributes a tick to every batch.
    Independent { children: Vec<ScheduleNode> },
}

impl<'a> BatchPlanner<'a> {
    pub fn new(tree: &'a GridJoinTree) -> Self {
        let leaves = tree.leaves();
        let root = build_schedule(tree, &leaves);
        Self { leaves, root }
    }

    pub fn next_batch(
        &mut self,
        batch_size: usize,
    ) -> Option<BatchPlan> {
        let mut slabs: Vec<LeafSlab> = Vec::new();
        let combine = advance_node(
            &mut self.root,
            &self.leaves,
            batch_size,
            &mut slabs,
        )?;
        Some(BatchPlan {
            batch: ReaderBatch { slabs },
            combine,
        })
    }
}

const MAX_DRIVER_SLABS_COALESCED: usize = 100;

/// Recursively advance a [`ScheduleNode`] by one tick. Returns the
/// [`CombineNode`] describing how the slabs produced this tick combine, or
/// `None` if the node is exhausted.
fn advance_node(
    node: &mut ScheduleNode,
    leaves: &[&OwnedGridGroup],
    batch_size: usize,
    slabs: &mut Vec<LeafSlab>,
) -> Option<CombineNode> {
    match node {
        ScheduleNode::Leaf {
            leaf_idx,
            cursor,
        } => {
            let g = leaves[*leaf_idx];
            let slots =
                take_slab(g, cursor, batch_size);
            if slots.is_empty() {
                return None;
            }
            slabs.push(LeafSlab {
                leaf_idx: *leaf_idx,
                chunk_slots: slots,
            });
            Some(CombineNode::Leaf {
                leaf_idx: *leaf_idx,
            })
        }
        ScheduleNode::JoinClosed {
            combine,
            driver_leaf,
            driver_cursor,
            participants,
            join_axes_per_participant,
        } => {
            let driver = leaves[*driver_leaf];
            let driver_slots = take_slab(
                driver,
                driver_cursor,
                batch_size,
            );
            if driver_slots.is_empty() {
                return None;
            }
            for (pi, &leaf_idx) in
                participants.iter().enumerate()
            {
                let slots = if leaf_idx
                    == *driver_leaf
                {
                    driver_slots.clone()
                } else {
                    let leaf = leaves[leaf_idx];
                    let driver_axes = &join_axes_per_participant[0];
                    let leaf_axes = &join_axes_per_participant[pi];
                    overlapping_chunks(
                        driver,
                        &driver_slots,
                        leaf,
                        driver_axes,
                        leaf_axes,
                    )
                };
                slabs.push(LeafSlab {
                    leaf_idx,
                    chunk_slots: slots,
                });
            }
            Some(combine.clone())
        }
        ScheduleNode::Independent {
            children,
        } => {
            let mut child_combines: Vec<
                CombineNode,
            > = Vec::new();
            for child in children {
                if let Some(c) = advance_node(
                    child, leaves, batch_size,
                    slabs,
                ) {
                    child_combines.push(c);
                }
            }
            match child_combines.len() {
                0 => None,
                1 => Some(
                    child_combines
                        .into_iter()
                        .next()
                        .unwrap(),
                ),
                _ => Some(CombineNode::Concat {
                    children: child_combines,
                }),
            }
        }
    }
}

/// Pull a row-budget bounded slab of chunk slots from `g`, advancing `cursor`.
fn take_slab(
    g: &OwnedGridGroup,
    cursor: &mut usize,
    batch_size: usize,
) -> Vec<usize> {
    let total = g.chunk_indices.len();
    let mut slots = Vec::new();
    let mut acc_rows = 0usize;
    while *cursor < total {
        let slot = *cursor;
        let rows = chunk_element_count(g, slot);
        if !slots.is_empty()
            && acc_rows.saturating_add(rows)
                > batch_size
        {
            break;
        }
        slots.push(slot);
        acc_rows = acc_rows.saturating_add(rows);
        *cursor += 1;
        if slots.len()
            >= MAX_DRIVER_SLABS_COALESCED
        {
            break;
        }
    }
    slots
}

fn build_schedule(
    tree: &GridJoinTree,
    leaves: &[&OwnedGridGroup],
) -> ScheduleNode {
    match tree {
        GridJoinTree::Leaf(g) => {
            let leaf_idx =
                leaf_index_of(leaves, g);
            ScheduleNode::Leaf {
                leaf_idx,
                cursor: 0,
            }
        }
        GridJoinTree::Independent(subs) => {
            ScheduleNode::Independent {
                children: subs
                    .iter()
                    .map(|s| {
                        build_schedule(s, leaves)
                    })
                    .collect(),
            }
        }
        GridJoinTree::Join { .. } => {
            // A `Join` node requires a single driver leaf and a flat list of
            // every participating leaf, regardless of nested Join structure.
            // Nested Independent inside a Join is invalid (the build phase
            // never produces it), but defensively we fall back to a recursive
            // Independent schedule rather than panicking.
            if has_independent_inside(tree) {
                let GridJoinTree::Join {
                    subtrees,
                    ..
                } = tree
                else {
                    unreachable!()
                };
                ScheduleNode::Independent {
                    children: subtrees
                        .iter()
                        .map(|s| {
                            build_schedule(
                                s, leaves,
                            )
                        })
                        .collect(),
                }
            } else {
                let combine =
                    combine_for(tree, leaves);
                let driver_leaf =
                    pick_driver_leaf(
                        tree, leaves,
                    );
                let mut participants = Vec::new();
                let mut join_axes_per_participant: Vec<Vec<usize>> = Vec::new();
                collect_join_participants(
                    tree,
                    leaves,
                    &[],
                    &mut participants,
                    &mut join_axes_per_participant,
                );
                // Move driver to index 0 so axis-projection in `advance_node`
                // can use participants[0] as the driver reference.
                if let Some(pos) =
                    participants.iter().position(
                        |&i| i == driver_leaf,
                    )
                    && pos != 0
                {
                    participants.swap(0, pos);
                    join_axes_per_participant
                        .swap(0, pos);
                }
                ScheduleNode::JoinClosed {
                    combine,
                    driver_leaf,
                    driver_cursor: 0,
                    participants,
                    join_axes_per_participant,
                }
            }
        }
    }
}

fn has_independent_inside(
    tree: &GridJoinTree,
) -> bool {
    match tree {
        GridJoinTree::Leaf(_) => false,
        GridJoinTree::Independent(_) => true,
        GridJoinTree::Join {
            subtrees, ..
        } => subtrees
            .iter()
            .any(has_independent_inside),
    }
}

fn pick_driver_leaf(
    tree: &GridJoinTree,
    leaves: &[&OwnedGridGroup],
) -> usize {
    let mut best: Option<(
        usize,
        (usize, usize),
    )> = None;
    visit_leaves_with_index(
        tree,
        leaves,
        &mut |idx, g| {
            let key = (
                g.sig.dims().len(),
                g.chunk_indices.len(),
            );
            if best
                .map(|(_, k)| key > k)
                .unwrap_or(true)
            {
                best = Some((idx, key));
            }
        },
    );
    best.map(|(i, _)| i).unwrap_or(0)
}

fn leaf_index_of(
    leaves: &[&OwnedGridGroup],
    g: &OwnedGridGroup,
) -> usize {
    leaves
        .iter()
        .position(|l| {
            std::ptr::eq(
                *l as *const _,
                g as *const _,
            )
        })
        .expect("leaf not found in leaves list")
}

fn visit_leaves_with_index(
    tree: &GridJoinTree,
    leaves: &[&OwnedGridGroup],
    f: &mut impl FnMut(usize, &OwnedGridGroup),
) {
    fn walk<'a>(
        node: &'a GridJoinTree,
        leaves: &[&'a OwnedGridGroup],
        f: &mut impl FnMut(usize, &OwnedGridGroup),
    ) {
        match node {
            GridJoinTree::Leaf(g) => {
                let idx =
                    leaf_index_of(leaves, g);
                f(idx, g);
            }
            GridJoinTree::Join {
                subtrees,
                ..
            }
            | GridJoinTree::Independent(
                subtrees,
            ) => {
                for s in subtrees {
                    walk(s, leaves, f);
                }
            }
        }
    }
    walk(tree, leaves, f);
}

/// Collect every leaf in a join-closed subtree, recording per-leaf join axes
/// (positions of accumulated join dims inside that leaf's signature).
fn collect_join_participants(
    tree: &GridJoinTree,
    leaves: &[&OwnedGridGroup],
    inherited_join: &[IStr],
    participants: &mut Vec<usize>,
    join_axes_per_leaf: &mut Vec<Vec<usize>>,
) {
    match tree {
        GridJoinTree::Leaf(g) => {
            participants
                .push(leaf_index_of(leaves, g));
            let axes: Vec<usize> = inherited_join
                .iter()
                .map(|d| {
                    g.sig.dims().iter().position(|sd| sd == d).expect(
                        "join dim should be present in leaf signature; tree build should guarantee this",
                    )
                })
                .collect();
            join_axes_per_leaf.push(axes);
        }
        GridJoinTree::Join {
            join_dims,
            subtrees,
        } => {
            let mut effective: Vec<IStr> =
                inherited_join.to_vec();
            for d in join_dims {
                if !effective.contains(d) {
                    effective.push(*d);
                }
            }
            for s in subtrees {
                collect_join_participants(
                    s,
                    leaves,
                    &effective,
                    participants,
                    join_axes_per_leaf,
                );
            }
        }
        GridJoinTree::Independent(_) => {
            unreachable!(
                "collect_join_participants should never see Independent (caller checks via has_independent_inside)"
            );
        }
    }
}

fn combine_for(
    tree: &GridJoinTree,
    leaves: &[&OwnedGridGroup],
) -> CombineNode {
    match tree {
        GridJoinTree::Leaf(g) => {
            CombineNode::Leaf {
                leaf_idx: leaf_index_of(
                    leaves, g,
                ),
            }
        }
        GridJoinTree::Join {
            join_dims,
            subtrees,
        } => CombineNode::Join {
            join_dims: join_dims
                .iter()
                .copied()
                .collect(),
            children: subtrees
                .iter()
                .map(|s| combine_for(s, leaves))
                .collect(),
        },
        GridJoinTree::Independent(subtrees) => {
            CombineNode::Concat {
                children: subtrees
                    .iter()
                    .map(|s| {
                        combine_for(s, leaves)
                    })
                    .collect(),
            }
        }
    }
}

// =============================================================================
// Geometry helpers (lifted from the deleted streaming_batch_plan.rs)
// =============================================================================

fn chunk_element_count(
    g: &OwnedGridGroup,
    slot: usize,
) -> usize {
    let idx = &g.chunk_indices[slot];
    let cs = g.sig.retrieval_shape();
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

fn axis_interval(
    g: &OwnedGridGroup,
    slot: usize,
    axis: usize,
) -> (u64, u64) {
    let idx = g.chunk_indices[slot][axis];
    let cs = g.sig.retrieval_shape()[axis];
    let alen = g.array_shape[axis];
    let start = idx * cs;
    let end = (start + cs).min(alen);
    (start, end)
}

/// Set of `leaf` chunk slots that overlap any of `driver_slots` on every join
/// dim. `driver_axes` and `leaf_axes` are positional axes (same length).
fn overlapping_chunks(
    driver: &OwnedGridGroup,
    driver_slots: &[usize],
    leaf: &OwnedGridGroup,
    driver_axes: &[usize],
    leaf_axes: &[usize],
) -> Vec<usize> {
    if leaf_axes.is_empty() {
        // No join dims at this level: every leaf chunk is a candidate.
        return (0..leaf.chunk_indices.len())
            .collect();
    }
    // Per-driver-axis: union of (start..end) intervals over driver slots.
    let driver_intervals: Vec<Vec<(u64, u64)>> =
        driver_axes
            .iter()
            .map(|&ax| {
                driver_slots
                    .iter()
                    .map(|&s| {
                        axis_interval(
                            driver, s, ax,
                        )
                    })
                    .collect()
            })
            .collect();

    let mut covered: BTreeSet<usize> =
        BTreeSet::new();
    for slot in 0..leaf.chunk_indices.len() {
        let mut all_axes_overlap = true;
        for (k, &leaf_ax) in
            leaf_axes.iter().enumerate()
        {
            let (lstart, lend) = axis_interval(
                leaf, slot, leaf_ax,
            );
            let any_overlap = driver_intervals[k]
                .iter()
                .any(|(ds, de)| {
                    !(*de <= lstart
                        || lend <= *ds)
                });
            if !any_overlap {
                all_axes_overlap = false;
                break;
            }
        }
        if all_axes_overlap {
            covered.insert(slot);
        }
    }
    covered.into_iter().collect()
}

// =============================================================================
// DataFrame combination
// =============================================================================

/// Combine per-leaf DataFrames according to a [`CombineNode`].
///
/// `per_leaf` is keyed by leaf index; missing leaves are skipped (their slab was
/// empty for this batch).
pub fn combine_per_leaf(
    node: &CombineNode,
    per_leaf: &mut std::collections::BTreeMap<
        usize,
        DataFrame,
    >,
) -> BackendResult<Option<DataFrame>> {
    match node {
        CombineNode::Leaf { leaf_idx } => {
            Ok(per_leaf.remove(leaf_idx))
        }
        CombineNode::Join {
            join_dims,
            children,
        } => {
            let mut child_dfs: Vec<DataFrame> =
                Vec::new();
            for c in children {
                if let Some(df) =
                    combine_per_leaf(c, per_leaf)?
                {
                    child_dfs.push(df);
                }
            }
            if child_dfs.is_empty() {
                return Ok(None);
            }
            if child_dfs.len() == 1 {
                return Ok(Some(
                    child_dfs
                        .into_iter()
                        .next()
                        .unwrap(),
                ));
            }
            let join_keys: Vec<PlSmallStr> =
                Vec::<PlSmallStr>::from_istrs(
                    join_dims.iter().copied(),
                );
            // Filter to keys that actually exist in every child.
            let live_keys: Vec<PlSmallStr> =
                join_keys
                    .into_iter()
                    .filter(|k| {
                        child_dfs.iter().all(
                            |df| {
                                df.column(
                                    k.as_ref(),
                                )
                                .is_ok()
                            },
                        )
                    })
                    .collect();
            if live_keys.is_empty() {
                return polars::functions::concat_df_diagonal(&child_dfs)
                    .context(PolarsSnafu {
                        message: "Error concatenating join children with no live keys".to_string(),
                    })
                    .map(Some);
            }
            let mut iter = child_dfs.into_iter();
            let mut acc = iter.next().unwrap();
            for df in iter {
                acc = acc
                    .join(
                        &df,
                        live_keys.as_slice(),
                        live_keys.as_slice(),
                        JoinArgs::new(JoinType::Full)
                            .with_coalesce(JoinCoalesce::CoalesceColumns),
                        None,
                    )
                    .context(PolarsSnafu {
                        message: "Error joining grid subtree DataFrames".to_string(),
                    })?;
            }
            Ok(Some(acc))
        }
        CombineNode::Concat { children } => {
            let mut child_dfs: Vec<DataFrame> =
                Vec::new();
            for c in children {
                if let Some(df) =
                    combine_per_leaf(c, per_leaf)?
                {
                    child_dfs.push(df);
                }
            }
            match child_dfs.len() {
                0 => Ok(None),
                1 => Ok(Some(child_dfs.into_iter().next().unwrap())),
                _ => polars::functions::concat_df_diagonal(&child_dfs)
                    .context(PolarsSnafu {
                        message: "Error diagonal-concatenating independent subtrees".to_string(),
                    })
                    .map(Some),
            }
        }
    }
}

/// vstack a list of chunk DataFrames belonging to one leaf.
pub fn vstack_leaf(
    dfs: Vec<DataFrame>,
) -> BackendResult<Option<DataFrame>> {
    if dfs.is_empty() {
        return Ok(None);
    }
    let mut iter = dfs.into_iter();
    let first = iter.next().unwrap();
    let col_order: Vec<PlSmallStr> =
        first.get_column_names_owned();
    let mut acc = first;
    for df in iter {
        let reordered = df
            .select(col_order.as_slice())
            .context(PolarsSnafu {
                message: "Error reordering chunk columns within leaf slab".to_string(),
            })?;
        acc.vstack_mut(&reordered).context(PolarsSnafu {
            message: "Error vstacking chunk DataFrames within leaf slab".to_string(),
        })?;
    }
    Ok(Some(acc))
}

/// Per-leaf chunk read descriptors used by both sync and async drivers.
pub struct ChunkRead {
    pub leaf_idx: usize,
    pub sig: Arc<ChunkGridSignature>,
    pub array_shape: Vec<u64>,
    pub vars: Vec<IStr>,
    pub idx: Vec<u64>,
    pub subset: Option<ChunkSubset>,
}

/// Flatten a [`BatchPlan`]'s slabs into a list of individual chunk read tasks.
pub fn flatten_reads(
    plan: &BatchPlan,
    leaves: &[&OwnedGridGroup],
) -> Vec<ChunkRead> {
    let mut out = Vec::new();
    for slab in &plan.batch.slabs {
        let g = leaves[slab.leaf_idx];
        for &slot in &slab.chunk_slots {
            out.push(ChunkRead {
                leaf_idx: slab.leaf_idx,
                sig: g.sig.clone(),
                array_shape: g
                    .array_shape
                    .clone(),
                vars: g.vars.clone(),
                idx: g.chunk_indices[slot]
                    .clone(),
                subset: g.chunk_subsets[slot]
                    .clone(),
            });
        }
    }
    out
}

/// Group a flat list of `(leaf_idx, DataFrame)` reads into per-leaf vstacked
/// DataFrames, then run [`combine_per_leaf`].
pub fn assemble_batch_dataframe(
    plan: &BatchPlan,
    chunk_dfs: Vec<(usize, DataFrame)>,
) -> BackendResult<Option<DataFrame>> {
    use std::collections::BTreeMap;

    let mut grouped: BTreeMap<
        usize,
        Vec<DataFrame>,
    > = BTreeMap::new();
    for (leaf_idx, df) in chunk_dfs {
        grouped
            .entry(leaf_idx)
            .or_default()
            .push(df);
    }
    let mut per_leaf: BTreeMap<usize, DataFrame> =
        BTreeMap::new();
    for (leaf_idx, dfs) in grouped {
        if let Some(df) = vstack_leaf(dfs)? {
            per_leaf.insert(leaf_idx, df);
        }
    }
    combine_per_leaf(&plan.combine, &mut per_leaf)
}

// Suppress unused warnings; these helpers exist for max_chunks_to_read accounting.
#[allow(dead_code)]
fn _unused(e: BackendError) -> BackendError {
    e
}
