//! Tree-driven batch planner.
//!
//! Splits the historic monolithic reader into:
//! - **Scheduling** ([`BatchPlanner`] + [`ScheduleNode`]): walks the
//!   [`GridJoinTree`], yielding one [`BatchPlan`] per tick. Driver leaves pace
//!   the slab; siblings project to overlapping chunks via
//!   [`overlapping_chunks`].
//! - **Combination recipe** ([`JoinedCombine`]): describes how the per-chunk
//!   raw reads of a batch combine into one output DataFrame.
//! - **Assembly** ([`crate::chunk_plan::indexing::joined_assembly`]): consumes
//!   `JoinedCombine` plus raw [`crate::reader::ColumnData`] and emits the
//!   output frame in a single index-math gather pass — no polars joins, no
//!   synthetic `__<dim>` columns.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::grid_join_tree::GridJoinTree;
use super::plan::OwnedGridGroup;
use crate::chunk_plan::ChunkGridSignature;
use crate::shared::IStr;

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
/// the resulting per-chunk raw data into one output DataFrame.
#[derive(Clone, Debug)]
pub struct BatchPlan {
    pub batch: ReaderBatch,
    pub combine: JoinedCombine,
}

/// Recipe for combining the raw chunk reads of a batch into one output frame.
///
/// All cartesian semantics — no polars joins. The associated assembler handles
/// each variant via index-math gather. The [`GridJoinTree::Join`] tree's
/// nested structure is *flattened* here: cartesian-over-union is associative
/// and commutative, so every leaf in a join-closed subtree contributes to one
/// flat list of `structural_leaves`, with `coord_leaves` (1D dim-coord arrays)
/// kept separate so the assembler can broadcast them onto matching `<dim>`
/// columns.
#[derive(Clone, Debug)]
pub enum JoinedCombine {
    /// One leaf's chunks become one DataFrame via vstack with user-facing dim cols.
    Single { leaf_idx: usize },
    /// Cartesian-over-union join across several leaves.
    Joined {
        structural_leaves: Vec<usize>,
        coord_leaves: Vec<usize>,
    },
    /// Diagonal-concat children (no shared keys → independent subtrees).
    Concat { children: Vec<JoinedCombine> },
}

/// Walk a [`GridJoinTree`] and emit a stream of [`BatchPlan`]s sized by `batch_size`.
pub struct BatchPlanner<'a> {
    leaves: Vec<&'a OwnedGridGroup>,
    root: ScheduleNode,
}

/// Recursive schedule node mirroring [`GridJoinTree`]. Carries per-node cursors
/// so successive [`BatchPlanner::next_batch`] calls advance the right leaves.
#[derive(Debug)]
enum ScheduleNode {
    Leaf {
        leaf_idx: usize,
        cursor: usize,
    },
    JoinClosed {
        combine: JoinedCombine,
        driver_leaf: usize,
        driver_cursor: usize,
        participants: Vec<usize>,
        join_axes_per_participant: Vec<Vec<usize>>,
    },
    Independent {
        children: Vec<ScheduleNode>,
    },
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

fn advance_node(
    node: &mut ScheduleNode,
    leaves: &[&OwnedGridGroup],
    batch_size: usize,
    slabs: &mut Vec<LeafSlab>,
) -> Option<JoinedCombine> {
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
            Some(JoinedCombine::Single {
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
                    let driver_axes =
                        &join_axes_per_participant[0];
                    let leaf_axes =
                        &join_axes_per_participant[pi];
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
                JoinedCombine,
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
                _ => Some(JoinedCombine::Concat {
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
                    build_joined_combine(
                        tree, leaves,
                    );
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

/// Build the [`JoinedCombine`] recipe for a tree.
///
/// Crucially, [`GridJoinTree::Join`] nesting flattens here: cartesian-over-union
/// is associative and commutative, so the assembler doesn't need to know about
/// the nested join structure. Coord leaves bubble up unchanged.
fn build_joined_combine(
    tree: &GridJoinTree,
    leaves: &[&OwnedGridGroup],
) -> JoinedCombine {
    match tree {
        GridJoinTree::Leaf(g) => {
            JoinedCombine::Single {
                leaf_idx: leaf_index_of(
                    leaves, g,
                ),
            }
        }
        GridJoinTree::Join { .. } => {
            let mut structural = Vec::new();
            let mut coords = Vec::new();
            collect_flat_join_participants(
                tree,
                leaves,
                &mut structural,
                &mut coords,
            );
            JoinedCombine::Joined {
                structural_leaves: structural,
                coord_leaves: coords,
            }
        }
        GridJoinTree::Independent(subs) => {
            JoinedCombine::Concat {
                children: subs
                    .iter()
                    .map(|s| {
                        build_joined_combine(
                            s, leaves,
                        )
                    })
                    .collect(),
            }
        }
    }
}

fn collect_flat_join_participants(
    tree: &GridJoinTree,
    leaves: &[&OwnedGridGroup],
    structural: &mut Vec<usize>,
    coords: &mut Vec<usize>,
) {
    match tree {
        GridJoinTree::Leaf(g) => {
            structural.push(leaf_index_of(
                leaves, g,
            ));
        }
        GridJoinTree::Join {
            subtrees,
            coord_leaves,
            ..
        } => {
            for s in subtrees {
                collect_flat_join_participants(
                    s, leaves, structural, coords,
                );
            }
            for c in coord_leaves {
                coords.push(leaf_index_of(
                    leaves, c,
                ));
            }
        }
        GridJoinTree::Independent(_) => {
            unreachable!(
                "collect_flat_join_participants should never see Independent (caller checks via has_independent_inside)"
            );
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

// =============================================================================
// Geometry helpers used by the planner's overlap detection.
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
        return (0..leaf.chunk_indices.len())
            .collect();
    }
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
// Per-chunk read descriptor + flatten
// =============================================================================

/// Per-leaf chunk read descriptors used by both sync and async drivers.
///
/// Carries the minimum needed to dispatch a raw chunk read (`sig`, `vars`,
/// `idx`) plus the `slot` index into the leaf's `chunk_indices`, so the joined
/// assembler can recover the chunk's position in the slab without an extra
/// lookup. Per-chunk geometry (`array_shape`, `chunk_subset`) is read from the
/// owning leaf at assembly time.
pub struct ChunkRead {
    pub leaf_idx: usize,
    pub slot: usize,
    pub sig: Arc<ChunkGridSignature>,
    pub vars: Vec<IStr>,
    pub idx: Vec<u64>,
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
                slot,
                sig: Arc::clone(&g.sig),
                vars: g.vars.clone(),
                idx: g.chunk_indices[slot]
                    .clone(),
            });
        }
    }
    out
}
