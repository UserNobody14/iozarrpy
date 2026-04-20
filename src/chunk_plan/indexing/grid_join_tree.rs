//! Recursive grid join tree.
//!
//! Replaces the awkward two-mode `Legacy` / `JoinClosed` scheduling with a single
//! tree shape:
//!
//! - [`GridJoinTree::Leaf`] – a single grid group's chunks.
//! - [`GridJoinTree::Join`] – multiple subtrees that share at least one dimension,
//!   with `join_dims` the intersection of dims across every subtree.
//! - [`GridJoinTree::Independent`] – subtrees with no shared dimensions
//!   (top-level diagonal-concat).
//! - [`GridJoinTree::Group`] – wraps a single child subtree under a named group,
//!   used by the builder to mirror `meta.root.children` hierarchy. Treated
//!   transparently by every traversal helper.
//!
//! The build algorithm partitions groups into connected components via union-find
//! over their dimension sets. Within a component, if all groups share at least one
//! dim we emit a single [`GridJoinTree::Join`] over leaves; otherwise we recursively
//! split on the dim shared by the most groups, producing a balanced binary join tree.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use smallvec::SmallVec;

use super::types::ChunkGridSignature;
use crate::shared::IStr;

// =============================================================================
// Chunk Subset
// =============================================================================

/// Per-chunk local bounding box: ranges in chunk-local coordinates.
///
/// Represents the portion of a chunk that intersects the user's selection.
/// Used to constrain the `KeepMask` so only relevant elements are processed.
#[derive(Debug, Clone)]
pub struct ChunkSubset {
    pub(crate) ranges: SmallVec<[Range<u64>; 4]>,
}

impl ChunkSubset {
    /// Construct a [`ChunkSubset`] from explicit per-dim chunk-local ranges.
    ///
    /// Used by tests and benches that need to materialize a subset without
    /// going through `compute_chunk_subset`.
    pub fn from_ranges(
        ranges: Vec<Range<u64>>,
    ) -> Self {
        Self {
            ranges: ranges.into_iter().collect(),
        }
    }

    pub(crate) fn is_full_chunk(
        &self,
        chunk_shape: &[u64],
    ) -> bool {
        self.ranges.iter().zip(chunk_shape).all(
            |(r, &s)| r.start == 0 && r.end >= s,
        )
    }
}

// =============================================================================
// Leaf Group
// =============================================================================

/// A grid group with deduplicated chunk indices, ready for reading.
///
/// Owned across the whole tree-driven read pipeline. Each [`GridJoinTree::Leaf`]
/// owns exactly one [`LeafGroup`].
#[derive(Debug)]
pub struct LeafGroup {
    pub sig: Arc<ChunkGridSignature>,
    pub vars: Vec<IStr>,
    pub chunk_indices: Vec<Vec<u64>>,
    pub chunk_subsets: Vec<Option<ChunkSubset>>,
    pub array_shape: Vec<u64>,
}

impl LeafGroup {
    pub fn new(
        sig: Arc<ChunkGridSignature>,
        vars: Vec<IStr>,
        chunk_indices: Vec<Vec<u64>>,
        chunk_subsets: Vec<Option<ChunkSubset>>,
        array_shape: Vec<u64>,
    ) -> Self {
        Self {
            sig,
            vars,
            chunk_indices,
            chunk_subsets,
            array_shape,
        }
    }
}

// =============================================================================
// Grid Join Tree
// =============================================================================

/// Tree shape that drives all batched zarr reads.
#[derive(Debug)]
pub enum GridJoinTree {
    /// One grid group's chunks, indexed by integer dim positions.
    Leaf(LeafGroup),
    /// Multiple subtrees joined on the dimensions in `join_dims`.
    /// Every subtree's dim set contains every dim in `join_dims`.
    Join {
        join_dims: SmallVec<[IStr; 4]>,
        subtrees: Vec<GridJoinTree>,
    },
    /// Subtrees with no shared dims; combined via diagonal concat.
    Independent(Vec<GridJoinTree>),
    /// Wraps `child`'s combined DataFrame into a struct column named `name`.
    /// Inserted by the builder to mirror `meta.root.children` hierarchy.
    /// Treated transparently by every traversal helper here — the variant is
    /// purely a marker that affects the assembly layer (see `grid_join_reader`).
    Group {
        name: IStr,
        child: Box<GridJoinTree>,
    },
}

impl GridJoinTree {
    /// Build a join tree from a list of grid groups.
    ///
    /// Returns `None` if `groups` is empty.
    ///
    /// Note: `build` never produces a [`GridJoinTree::Group`] node — those are
    /// inserted by the higher-level builder once the join structure is known.
    pub fn build(
        groups: Vec<LeafGroup>,
    ) -> Option<GridJoinTree> {
        if groups.is_empty() {
            return None;
        }
        let components =
            connected_components(&groups);
        let mut subtrees: Vec<GridJoinTree> =
            Vec::with_capacity(components.len());
        // Move groups into Option slots so we can take ownership in arbitrary order.
        let mut owned: Vec<Option<LeafGroup>> =
            groups
                .into_iter()
                .map(Some)
                .collect();
        for comp in components {
            let comp_groups: Vec<LeafGroup> =
                comp.into_iter()
                    .map(|i| {
                        owned[i].take().expect(
                        "group consumed twice",
                    )
                    })
                    .collect();
            subtrees.push(build_component(
                comp_groups,
            ));
        }
        Some(if subtrees.len() == 1 {
            subtrees.into_iter().next().unwrap()
        } else {
            GridJoinTree::Independent(subtrees)
        })
    }

    /// Iterate every leaf grid in left-to-right traversal order.
    pub fn leaves(&self) -> Vec<&LeafGroup> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves<'a>(
        &'a self,
        out: &mut Vec<&'a LeafGroup>,
    ) {
        match self {
            GridJoinTree::Leaf(g) => out.push(g),
            GridJoinTree::Join {
                subtrees,
                ..
            }
            | GridJoinTree::Independent(
                subtrees,
            ) => {
                for s in subtrees {
                    s.collect_leaves(out);
                }
            }
            GridJoinTree::Group {
                child, ..
            } => {
                child.collect_leaves(out);
            }
        }
    }
}

fn build_component(
    mut groups: Vec<LeafGroup>,
) -> GridJoinTree {
    if groups.len() == 1 {
        return GridJoinTree::Leaf(
            groups.pop().unwrap(),
        );
    }

    let intersection = dim_intersection(&groups);
    if !intersection.is_empty() {
        let subtrees: Vec<GridJoinTree> = groups
            .into_iter()
            .map(GridJoinTree::Leaf)
            .collect();
        return GridJoinTree::Join {
            join_dims: intersection,
            subtrees,
        };
    }

    // Empty intersection (e.g. (t,x), (t,y), (x,y)). Pick the dim shared by the most
    // groups, partition into "has dim" and "lacks dim", recurse on each side,
    // and emit a binary Join over the two halves when they still share something.
    let split_dim = pick_majority_dim(&groups);
    let (with_dim, without_dim): (
        Vec<LeafGroup>,
        Vec<LeafGroup>,
    ) = groups.into_iter().partition(|g| {
        g.sig
            .dims()
            .iter()
            .any(|d| d == &split_dim)
    });

    if without_dim.is_empty() {
        let subtrees: Vec<GridJoinTree> =
            with_dim
                .into_iter()
                .map(GridJoinTree::Leaf)
                .collect();
        return GridJoinTree::Join {
            join_dims: smallvec::smallvec![
                split_dim
            ],
            subtrees,
        };
    }

    let subtrees = vec![
        build_component(with_dim),
        build_component(without_dim),
    ];
    let join_dims =
        subtree_intersection(&subtrees);
    if join_dims.is_empty() {
        GridJoinTree::Independent(subtrees)
    } else {
        GridJoinTree::Join {
            join_dims,
            subtrees,
        }
    }
}

fn dim_intersection(
    groups: &[LeafGroup],
) -> SmallVec<[IStr; 4]> {
    let mut iter = groups.iter();
    let first = match iter.next() {
        Some(g) => g.sig.dims(),
        None => return SmallVec::new(),
    };
    let mut acc: BTreeSet<IStr> =
        first.iter().copied().collect();
    for g in iter {
        let dims: BTreeSet<IStr> = g
            .sig
            .dims()
            .iter()
            .copied()
            .collect();
        acc = acc
            .intersection(&dims)
            .copied()
            .collect();
        if acc.is_empty() {
            break;
        }
    }
    // Preserve dim order of the first group for stable output.
    first
        .iter()
        .copied()
        .filter(|d| acc.contains(d))
        .collect()
}

/// Dims present in **every leaf** of **every subtree**. A valid join key must
/// exist on every participating leaf, otherwise the join coalesce fails.
fn subtree_intersection(
    subtrees: &[GridJoinTree],
) -> SmallVec<[IStr; 4]> {
    if subtrees.is_empty() {
        return SmallVec::new();
    }
    let leaf_dim_sets: Vec<BTreeSet<IStr>> =
        subtrees
            .iter()
            .flat_map(|s| {
                s.leaves().into_iter().map(|l| {
                    l.sig
                        .dims()
                        .iter()
                        .copied()
                        .collect()
                })
            })
            .collect();
    if leaf_dim_sets.is_empty() {
        return SmallVec::new();
    }
    let mut acc = leaf_dim_sets[0].clone();
    for s in &leaf_dim_sets[1..] {
        acc = acc
            .intersection(s)
            .copied()
            .collect();
        if acc.is_empty() {
            break;
        }
    }
    let mut out: SmallVec<[IStr; 4]> =
        SmallVec::new();
    if let Some(first_leaf) =
        subtrees[0].leaves().first()
    {
        for d in first_leaf.sig.dims() {
            if acc.contains(d) {
                out.push(*d);
            }
        }
    }
    out
}

fn pick_majority_dim(
    groups: &[LeafGroup],
) -> IStr {
    let mut counts: BTreeMap<IStr, usize> =
        BTreeMap::new();
    for g in groups {
        for d in g.sig.dims() {
            *counts.entry(*d).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(d, _)| d)
        .expect("at least one group has at least one dim")
}

/// Union-find connected-components over the shared-dim graph.
fn connected_components(
    groups: &[LeafGroup],
) -> Vec<Vec<usize>> {
    let n = groups.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(
        parent: &mut [usize],
        i: usize,
    ) -> usize {
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = i;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    let dim_sets: Vec<BTreeSet<IStr>> = groups
        .iter()
        .map(|g| {
            g.sig.dims().iter().copied().collect()
        })
        .collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if dim_sets[i]
                .intersection(&dim_sets[j])
                .next()
                .is_some()
            {
                let a = find(&mut parent, i);
                let b = find(&mut parent, j);
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }

    let mut comps: BTreeMap<usize, Vec<usize>> =
        BTreeMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        comps.entry(r).or_default().push(i);
    }
    comps.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::IntoIStr;

    fn mk(
        name: &str,
        dims: &[&str],
    ) -> LeafGroup {
        let dim_istrs: SmallVec<[IStr; 4]> = dims
            .iter()
            .map(|d| (*d).istr())
            .collect();
        let sig = Arc::new(
            ChunkGridSignature::from_dims_only(
                dim_istrs.clone(),
            ),
        );
        LeafGroup::new(
            sig,
            vec![name.istr()],
            vec![vec![0; dims.len()]],
            vec![None],
            dims.iter().map(|_| 1).collect(),
        )
    }

    #[test]
    fn single_grid_becomes_leaf() {
        let tree = GridJoinTree::build(vec![mk(
            "a",
            &["t"],
        )])
        .unwrap();
        assert!(matches!(
            tree,
            GridJoinTree::Leaf(_)
        ));
    }

    #[test]
    fn two_disjoint_become_independent() {
        let tree = GridJoinTree::build(vec![
            mk("a", &["x"]),
            mk("b", &["y"]),
        ])
        .unwrap();
        match tree {
            GridJoinTree::Independent(subs) => {
                assert_eq!(subs.len(), 2)
            }
            _ => panic!(
                "expected Independent, got {:?}",
                tree
            ),
        }
    }

    #[test]
    fn full_intersection_becomes_single_join() {
        let tree = GridJoinTree::build(vec![
            mk("a", &["t", "x"]),
            mk("b", &["t", "y"]),
            mk("c", &["t"]),
        ])
        .unwrap();
        match tree {
            GridJoinTree::Join {
                join_dims,
                subtrees,
            } => {
                assert_eq!(
                    join_dims.as_slice(),
                    &["t".istr()][..]
                );
                assert_eq!(subtrees.len(), 3);
                for s in &subtrees {
                    assert!(matches!(
                        s,
                        GridJoinTree::Leaf(_)
                    ));
                }
            }
            _ => panic!("expected Join"),
        }
    }

    #[test]
    fn triangle_recursively_splits() {
        // (t,x), (t,y), (x,y): pairwise share one dim each; intersection empty.
        let tree = GridJoinTree::build(vec![
            mk("a", &["t", "x"]),
            mk("b", &["t", "y"]),
            mk("c", &["x", "y"]),
        ])
        .unwrap();
        // Expect a Join (t shared between a,b) nested with c.
        match &tree {
            GridJoinTree::Join { .. }
            | GridJoinTree::Independent(_) => {
                let leaves = tree.leaves();
                assert_eq!(leaves.len(), 3);
            }
            other => panic!(
                "unexpected shape: {other:?}"
            ),
        }
    }

    #[test]
    fn three_grids_user_example_shape() {
        // 10x4x30 (a,b,blah), 5x3x30 (c,d,blah), 30 (blah).
        let tree = GridJoinTree::build(vec![
            mk("v1", &["a", "b", "blah"]),
            mk("v2", &["c", "d", "blah"]),
            mk("blah", &["blah"]),
        ])
        .unwrap();
        match tree {
            GridJoinTree::Join {
                join_dims,
                subtrees,
            } => {
                assert_eq!(
                    join_dims.as_slice(),
                    &["blah".istr()][..]
                );
                assert_eq!(subtrees.len(), 3);
            }
            other => panic!(
                "expected single Join on blah, got {other:?}"
            ),
        }
    }

    #[test]
    fn group_wraps_leaf_transparently() {
        let leaf = mk("a", &["t"]);
        let inner_var = leaf.vars[0];
        let tree = GridJoinTree::Group {
            name: "model_a".istr(),
            child: Box::new(GridJoinTree::Leaf(
                leaf,
            )),
        };
        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].vars[0], inner_var);
    }
}
