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
use crate::shared::IStr;

/// Synthetic column name used for a dim's integer-position join key. We add
/// a `__` prefix so it can never collide with a user var named after the dim
/// (e.g. a 1D coord array `y` for dim `y`).
pub fn synthetic_dim_key(dim: IStr) -> String {
    format!("__{}", dim.as_ref())
}

/// Inverse of [`synthetic_dim_key`]: strips the `__` prefix if present.
fn dim_name_from_key(key: &str) -> Option<&str> {
    key.strip_prefix("__")
}

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
///
/// Join keys are the synthetic `__<dim>` integer-position columns produced by
/// [`crate::scan`] for every chunk; the user-facing `<dim>` (and any
/// `<dim>`-named coord values supplied by a [`CombineNode::Join::coord_children`])
/// are reconciled by [`finalize_dim_columns`] at the end of every batch.
#[derive(Clone, Debug)]
pub enum CombineNode {
    /// Take the DataFrame for `leaf_idx` as-is.
    Leaf { leaf_idx: usize },
    /// Full-outer-join `children` on `__<dim>` columns (with key coalescing),
    /// then full-outer-join each `coord_children` entry onto the result on the
    /// same keys. `coord_children` carry 1D dim-coord arrays and broadcast
    /// their `<dim>` value column across every row sharing the same `__<dim>`.
    Join {
        join_dims: Vec<IStr>,
        children: Vec<CombineNode>,
        coord_children: Vec<CombineNode>,
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
        // coord_leaves are always plain leaves; no need to inspect them.
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
                coord_leaves,
                ..
            } => {
                for s in subtrees {
                    walk(s, leaves, f);
                }
                for c in coord_leaves {
                    let idx =
                        leaf_index_of(leaves, c);
                    f(idx, c);
                }
            }
            GridJoinTree::Independent(
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
            coord_leaves,
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
            for c in coord_leaves {
                // A coord leaf is a 1D group on dim `d`; its only join axis
                // is the position of `d` inside its (single-dim) signature,
                // which is always 0. We still compute it via lookup for
                // robustness in case effective contains other dims that don't
                // apply to this coord (we keep only the ones that do).
                let idx =
                    leaf_index_of(leaves, c);
                let axes: Vec<usize> = effective
                    .iter()
                    .filter_map(|d| {
                        c.sig
                            .dims()
                            .iter()
                            .position(|sd| {
                                sd == d
                            })
                    })
                    .collect();
                participants.push(idx);
                join_axes_per_leaf.push(axes);
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
            coord_leaves,
        } => CombineNode::Join {
            join_dims: join_dims
                .iter()
                .copied()
                .collect(),
            children: subtrees
                .iter()
                .map(|s| combine_for(s, leaves))
                .collect(),
            coord_children: coord_leaves
                .iter()
                .map(|g| CombineNode::Leaf {
                    leaf_idx: leaf_index_of(
                        leaves, g,
                    ),
                })
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
            coord_children,
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
            // Join keys live in DataFrames as `__<dim>` (synthetic positions
            // produced by the chunk-to-df layer). User-facing `<dim>` columns
            // are reconciled by `finalize_dim_columns` once per batch.
            let join_keys: Vec<PlSmallStr> =
                join_dims
                    .iter()
                    .map(|d| {
                        PlSmallStr::from(
                            synthetic_dim_key(*d),
                        )
                    })
                    .collect();

            let acc_opt = match child_dfs.len() {
                0 => None,
                1 => Some(
                    child_dfs
                        .into_iter()
                        .next()
                        .unwrap(),
                ),
                _ => Some(join_many(
                    child_dfs,
                    &join_keys,
                    &JoinType::Full,
                    "join subtree",
                )?),
            };

            // Apply coord broadcasts onto the structural-join result. Each
            // coord child has a single `__<dim>` synthetic key (its own dim,
            // *not* the structural join dims), so we left-join on that one
            // key per coord. Left join semantics: never invent rows for coord
            // values whose `__<dim>` index isn't present in the structural
            // side (avoids spurious null-data rows from FOJ).
            let mut acc = acc_opt;
            for c in coord_children {
                let Some(coord_df) =
                    combine_per_leaf(
                        c, per_leaf,
                    )?
                else {
                    continue;
                };
                acc = Some(match acc {
                    None => coord_df,
                    Some(prev) => {
                        let coord_keys: Vec<
                            PlSmallStr,
                        > = coord_df
                            .get_column_names_owned()
                            .into_iter()
                            .filter(|n| {
                                dim_name_from_key(
                                    n.as_str(),
                                )
                                .is_some()
                            })
                            .collect();
                        if coord_keys.is_empty() {
                            // Defensive: coord df with no `__<dim>` column
                            // can't be joined; concat diagonally so the
                            // values still appear (matches old fallback).
                            polars::functions::concat_df_diagonal(&[prev, coord_df])
                                .context(PolarsSnafu {
                                    message: "Error concatenating coord broadcast with no dim key".to_string(),
                                })?
                        } else {
                            join_many(
                                vec![
                                    prev,
                                    coord_df,
                                ],
                                &coord_keys,
                                &JoinType::Left,
                                "coord broadcast",
                            )?
                        }
                    }
                });
            }
            Ok(acc)
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

/// Join a non-empty list of DataFrames on `join_keys` (with key coalescing).
///
/// `join_type` selects the semantics: `Full` for structural sub-tree merges
/// where every participant contributes rows, `Left` for coord-broadcast
/// merges where the right-hand side only supplies values for keys already
/// present on the left.
///
/// Keys missing from one or more children are silently dropped from the
/// join condition; if no live keys remain, the result is a diagonal concat
/// (preserving the column union but no row alignment).
fn join_many(
    child_dfs: Vec<DataFrame>,
    join_keys: &[PlSmallStr],
    join_type: &JoinType,
    context: &str,
) -> BackendResult<DataFrame> {
    let live_keys: Vec<PlSmallStr> = join_keys
        .iter()
        .filter(|k| {
            child_dfs.iter().all(|df| {
                df.column(k.as_ref()).is_ok()
            })
        })
        .cloned()
        .collect();
    if live_keys.is_empty() {
        return polars::functions::concat_df_diagonal(&child_dfs)
            .context(PolarsSnafu {
                message: format!(
                    "Error concatenating {context} children with no live keys",
                ),
            });
    }
    let mut iter = child_dfs.into_iter();
    let mut acc = iter.next().unwrap();
    for df in iter {
        acc = acc
            .join(
                &df,
                live_keys.as_slice(),
                live_keys.as_slice(),
                JoinArgs::new(join_type.clone())
                    .with_coalesce(JoinCoalesce::CoalesceColumns),
                None,
            )
            .context(PolarsSnafu {
                message: format!("Error joining {context} DataFrames"),
            })?;
    }
    Ok(acc)
}

/// Collapse synthetic `__<dim>` integer-position columns to user-facing dim
/// names. For every column starting with `__`, if the bare `<dim>` column
/// already exists (e.g. coordinate values were broadcast in by a coord_leaf),
/// drop the `__<dim>`; otherwise rename it to `<dim>`.
pub fn finalize_dim_columns(
    df: DataFrame,
) -> BackendResult<DataFrame> {
    let names: Vec<PlSmallStr> =
        df.get_column_names_owned();
    let mut to_drop: Vec<PlSmallStr> = Vec::new();
    let mut renames: Vec<(
        PlSmallStr,
        PlSmallStr,
    )> = Vec::new();
    let existing: std::collections::HashSet<
        &str,
    > = names
        .iter()
        .map(|n| n.as_str())
        .collect();
    for n in &names {
        let Some(bare) =
            dim_name_from_key(n.as_str())
        else {
            continue;
        };
        if existing.contains(bare) {
            to_drop.push(n.clone());
        } else {
            renames.push((
                n.clone(),
                PlSmallStr::from(bare),
            ));
        }
    }
    let mut out = df;
    if !to_drop.is_empty() {
        out = out
            .drop_many(to_drop.iter().cloned());
    }
    for (old, new) in renames {
        out.rename(
            old.as_ref(),
            new.clone(),
        )
        .context(PolarsSnafu {
            message: format!(
                "Error renaming synthetic dim column {} to {}",
                old.as_str(),
                new.as_str()
            ),
        })?;
    }
    Ok(out)
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
                sig: Arc::clone(&g.sig),
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
    let combined = combine_per_leaf(
        &plan.combine,
        &mut per_leaf,
    )?;
    match combined {
        Some(df) => {
            Ok(Some(finalize_dim_columns(df)?))
        }
        None => Ok(None),
    }
}

// Suppress unused warnings; these helpers exist for max_chunks_to_read accounting.
#[allow(dead_code)]
fn _unused(e: BackendError) -> BackendError {
    e
}
