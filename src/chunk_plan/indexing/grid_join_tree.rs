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
//!
//! The build algorithm partitions groups into connected components via union-find
//! over their dimension sets. Within a component, if all groups share at least one
//! dim we emit a single [`GridJoinTree::Join`] over leaves; otherwise we recursively
//! split on the dim shared by the most groups, producing a balanced binary join tree.

use std::collections::{BTreeMap, BTreeSet};

use smallvec::SmallVec;

use super::plan::OwnedGridGroup;
use crate::shared::IStr;

/// Tree shape that drives all batched zarr reads.
#[derive(Debug)]
pub enum GridJoinTree {
    /// One grid group's chunks, indexed by integer dim positions.
    Leaf(OwnedGridGroup),
    /// Multiple subtrees joined on the dimensions in `join_dims`.
    /// Every subtree's dim set contains every dim in `join_dims`.
    ///
    /// `coord_leaves` are 1D dimension-coordinate arrays (`var name == dim
    /// name`, dim ∈ `join_dims`) that broadcast their values onto the join
    /// result via positional alignment on the synthetic `__<dim>` index. They
    /// are drained out of `subtrees` so the join's structural shape stays clean
    /// and a downstream coord-broadcast fast path is unambiguous.
    Join {
        join_dims: SmallVec<[IStr; 4]>,
        subtrees: Vec<GridJoinTree>,
        coord_leaves: Vec<OwnedGridGroup>,
    },
    /// Subtrees with no shared dims; combined via diagonal concat.
    Independent(Vec<GridJoinTree>),
}

impl GridJoinTree {
    /// Build a join tree from a list of grid groups.
    ///
    /// Returns `None` if `groups` is empty.
    pub fn build(
        groups: Vec<OwnedGridGroup>,
    ) -> Option<GridJoinTree> {
        if groups.is_empty() {
            return None;
        }
        let components =
            connected_components(&groups);
        let mut subtrees: Vec<GridJoinTree> =
            Vec::with_capacity(components.len());
        // Move groups into Option slots so we can take ownership in arbitrary order.
        let mut owned: Vec<
            Option<OwnedGridGroup>,
        > = groups
            .into_iter()
            .map(Some)
            .collect();
        for comp in components {
            let comp_groups: Vec<OwnedGridGroup> =
                comp.into_iter().map(|i| owned[i].take().expect("group consumed twice")).collect();
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
    pub fn leaves(&self) -> Vec<&OwnedGridGroup> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves<'a>(
        &'a self,
        out: &mut Vec<&'a OwnedGridGroup>,
    ) {
        match self {
            GridJoinTree::Leaf(g) => out.push(g),
            GridJoinTree::Join {
                subtrees,
                coord_leaves,
                ..
            } => {
                for s in subtrees {
                    s.collect_leaves(out);
                }
                for c in coord_leaves {
                    out.push(c);
                }
            }
            GridJoinTree::Independent(
                subtrees,
            ) => {
                for s in subtrees {
                    s.collect_leaves(out);
                }
            }
        }
    }
}

/// True when `g` is a single-dim group whose only var is the dim's coord
/// (var name == dim name). These are pure dim-coordinate arrays and broadcast
/// naturally onto any join that includes their dim.
fn is_dim_coord_group(
    g: &OwnedGridGroup,
) -> bool {
    g.sig.dims().len() == 1
        && g.vars.len() == 1
        && g.vars[0] == g.sig.dims()[0]
}

fn build_component(
    mut groups: Vec<OwnedGridGroup>,
) -> GridJoinTree {
    if groups.len() == 1 {
        return GridJoinTree::Leaf(
            groups.pop().unwrap(),
        );
    }

    // Compute the join-dim intersection over *structural* groups only. 1D
    // dim-coord arrays (var name == dim name) deliberately do not contribute,
    // otherwise they'd nuke an N-D var's intersection down to ∅ (e.g. a
    // `temperature(t,lt,y,x)` group joined with `t(t)`, `lt(lt)`, `y(y)`,
    // `x(x)` coord arrays — the structural intersection is `{t,lt,y,x}`
    // which lets every coord drain to `coord_leaves` in one Join).
    let intersection =
        structural_dim_intersection(&groups);
    if !intersection.is_empty() {
        // Any dim that appears in *some* structural group can host a
        // coord-broadcast: the join keeps `__<dim>` columns from every
        // participant, and the coord leaf joins onto whichever rows have
        // that key. Drain coord-only groups whose dim is in the structural
        // union; the truly orphan ones (dim absent from every structural
        // group) stay as Independent siblings below.
        let structural_union: BTreeSet<IStr> =
            groups
                .iter()
                .filter(|g| {
                    !is_dim_coord_group(g)
                })
                .flat_map(|g| {
                    g.sig.dims().iter().copied()
                })
                .collect();
        let (coord_leaves, rest): (
            Vec<OwnedGridGroup>,
            Vec<OwnedGridGroup>,
        ) = groups.into_iter().partition(|g| {
            is_dim_coord_group(g)
                && structural_union
                    .contains(&g.sig.dims()[0])
        });
        let (subtree_groups, leftover_coords): (
            Vec<OwnedGridGroup>,
            Vec<OwnedGridGroup>,
        ) = rest.into_iter().partition(|g| {
            let dims: BTreeSet<IStr> = g
                .sig
                .dims()
                .iter()
                .copied()
                .collect();
            intersection
                .iter()
                .all(|d| dims.contains(d))
        });
        let subtrees: Vec<GridJoinTree> =
            subtree_groups
                .into_iter()
                .map(GridJoinTree::Leaf)
                .collect();
        let join = finalize_join(
            intersection,
            subtrees,
            coord_leaves,
        );
        if leftover_coords.is_empty() {
            return join;
        }
        let mut indep = vec![join];
        for c in leftover_coords {
            indep.push(GridJoinTree::Leaf(c));
        }
        return GridJoinTree::Independent(indep);
    }

    // Empty intersection (e.g. (t,x), (t,y), (x,y)). Pick the dim shared by the most
    // groups, partition into "has dim" and "lacks dim", recurse on each side,
    // and emit a binary Join over the two halves when they still share something.
    let split_dim = pick_majority_dim(&groups);
    let (with_dim, without_dim): (
        Vec<OwnedGridGroup>,
        Vec<OwnedGridGroup>,
    ) = groups.into_iter().partition(|g| {
        g.sig
            .dims()
            .iter()
            .any(|d| d == &split_dim)
    });

    if without_dim.is_empty() {
        let join_dims: SmallVec<[IStr; 4]> =
            smallvec::smallvec![split_dim];
        let (coord_leaves, leaf_groups): (
            Vec<OwnedGridGroup>,
            Vec<OwnedGridGroup>,
        ) = with_dim.into_iter().partition(|g| {
            is_dim_coord_group(g)
                && g.sig.dims()[0] == split_dim
        });
        let subtrees: Vec<GridJoinTree> =
            leaf_groups
                .into_iter()
                .map(GridJoinTree::Leaf)
                .collect();
        return finalize_join(
            join_dims,
            subtrees,
            coord_leaves,
        );
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
            coord_leaves: Vec::new(),
        }
    }
}

/// Build a `Join` node, collapsing degenerate cases:
/// - 0 subtrees + 0 coords is impossible (caller must avoid).
/// - 1 subtree + 0 coords collapses to that subtree.
/// - 0 subtrees + 1 coord collapses to a leaf on that coord.
fn finalize_join(
    join_dims: SmallVec<[IStr; 4]>,
    mut subtrees: Vec<GridJoinTree>,
    mut coord_leaves: Vec<OwnedGridGroup>,
) -> GridJoinTree {
    if subtrees.is_empty()
        && coord_leaves.len() == 1
    {
        return GridJoinTree::Leaf(
            coord_leaves.pop().unwrap(),
        );
    }
    if subtrees.is_empty() {
        // Multiple coord leaves with no other subtree:
        // promote them to subtree leaves so the join still has structural
        // participants. This shouldn't normally happen (multiple dim-coord
        // arrays per dim is unusual) but we handle it for correctness.
        let subs: Vec<GridJoinTree> =
            coord_leaves
                .into_iter()
                .map(GridJoinTree::Leaf)
                .collect();
        return GridJoinTree::Join {
            join_dims,
            subtrees: subs,
            coord_leaves: Vec::new(),
        };
    }
    if coord_leaves.is_empty()
        && subtrees.len() == 1
    {
        return subtrees.pop().unwrap();
    }
    GridJoinTree::Join {
        join_dims,
        subtrees,
        coord_leaves,
    }
}

fn dim_intersection(
    groups: &[OwnedGridGroup],
) -> SmallVec<[IStr; 4]> {
    intersection_of(
        groups.iter().map(|g| g.sig.dims()),
    )
}

/// Like [`dim_intersection`] but skips 1D dim-coord groups so they don't
/// collapse the structural intersection. If every group is a coord group,
/// falls back to the all-group intersection so we still find shared dims.
fn structural_dim_intersection(
    groups: &[OwnedGridGroup],
) -> SmallVec<[IStr; 4]> {
    let structural: Vec<&OwnedGridGroup> = groups
        .iter()
        .filter(|g| !is_dim_coord_group(g))
        .collect();
    if structural.is_empty() {
        return dim_intersection(groups);
    }
    intersection_of(
        structural.iter().map(|g| g.sig.dims()),
    )
}

fn intersection_of<'a>(
    mut iter: impl Iterator<Item = &'a [IStr]>,
) -> SmallVec<[IStr; 4]> {
    let first = match iter.next() {
        Some(d) => d,
        None => return SmallVec::new(),
    };
    let mut acc: BTreeSet<IStr> =
        first.iter().copied().collect();
    for dims in iter {
        let dim_set: BTreeSet<IStr> =
            dims.iter().copied().collect();
        acc = acc
            .intersection(&dim_set)
            .copied()
            .collect();
        if acc.is_empty() {
            break;
        }
    }
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
    groups: &[OwnedGridGroup],
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
    groups: &[OwnedGridGroup],
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
    use std::sync::Arc;

    use super::*;
    use crate::chunk_plan::ChunkGridSignature;
    use crate::shared::IntoIStr;

    fn mk(
        name: &str,
        dims: &[&str],
    ) -> OwnedGridGroup {
        let dim_istrs: SmallVec<[IStr; 4]> = dims
            .iter()
            .map(|d| (*d).istr())
            .collect();
        let sig = Arc::new(
            ChunkGridSignature::from_dims_only(
                dim_istrs.clone(),
            ),
        );
        OwnedGridGroup::new(
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
        // a, b, c are all data vars (var name != dim name) so none get
        // promoted to coord_leaves; all stay as regular subtree leaves.
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
                coord_leaves,
            } => {
                assert_eq!(
                    join_dims.as_slice(),
                    &["t".istr()][..]
                );
                assert_eq!(subtrees.len(), 3);
                assert!(coord_leaves.is_empty());
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
        // The "blah" 1D coord group (var name == dim name) is drained into
        // `coord_leaves` rather than left as a regular subtree.
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
                coord_leaves,
            } => {
                assert_eq!(
                    join_dims.as_slice(),
                    &["blah".istr()][..]
                );
                assert_eq!(subtrees.len(), 2);
                assert_eq!(coord_leaves.len(), 1);
                assert_eq!(
                    coord_leaves[0].vars[0],
                    "blah".istr()
                );
            }
            other => panic!(
                "expected single Join on blah, got {other:?}"
            ),
        }
    }

    #[test]
    fn dim_coord_drains_to_coord_leaves() {
        // A single 2D var + its 1D dim-coord: structural intersection is the
        // var's full dim set (`y, x`), the coord drains to coord_leaves, and
        // the join keeps the structural shape (one subtree).
        let tree = GridJoinTree::build(vec![
            mk("temperature", &["y", "x"]),
            mk("y", &["y"]),
        ])
        .unwrap();
        match tree {
            GridJoinTree::Join {
                join_dims,
                subtrees,
                coord_leaves,
            } => {
                assert_eq!(
                    join_dims.as_slice(),
                    &["y".istr(), "x".istr()][..]
                );
                assert_eq!(subtrees.len(), 1);
                assert_eq!(coord_leaves.len(), 1);
                assert_eq!(
                    coord_leaves[0].vars[0],
                    "y".istr()
                );
            }
            other => panic!(
                "expected Join with coord_leaf, got {other:?}"
            ),
        }
    }

    #[test]
    fn aux_1d_coord_stays_as_subtree() {
        // `latitude` on `point` is an auxiliary 1D coord (var name != dim
        // name): it should remain a regular subtree, NOT become a coord_leaf.
        let tree = GridJoinTree::build(vec![
            mk("temperature", &["point"]),
            mk("latitude", &["point"]),
        ])
        .unwrap();
        match tree {
            GridJoinTree::Join {
                join_dims,
                subtrees,
                coord_leaves,
            } => {
                assert_eq!(
                    join_dims.as_slice(),
                    &["point".istr()][..]
                );
                assert_eq!(subtrees.len(), 2);
                assert!(
                    coord_leaves.is_empty(),
                    "auxiliary 1D coords must not be drained"
                );
            }
            other => panic!(
                "expected Join with two subtrees, got {other:?}"
            ),
        }
    }

    #[test]
    fn nd_var_with_full_coord_set_collapses_to_one_join()
     {
        // The exact shape produced by an unfiltered 4D scan with full coord
        // arrays: the structural intersection is the data var's full dim set
        // (because coord arrays don't contribute), and every coord drains to
        // coord_leaves under one Join. No Independent diagonal-concat.
        let tree = GridJoinTree::build(vec![
            mk(
                "temperature",
                &["time", "lead_time", "y", "x"],
            ),
            mk("time", &["time"]),
            mk("lead_time", &["lead_time"]),
            mk("y", &["y"]),
            mk("x", &["x"]),
        ])
        .unwrap();
        match tree {
            GridJoinTree::Join {
                join_dims,
                subtrees,
                coord_leaves,
            } => {
                assert_eq!(
                    join_dims.as_slice(),
                    &[
                        "time".istr(),
                        "lead_time".istr(),
                        "y".istr(),
                        "x".istr()
                    ][..]
                );
                assert_eq!(subtrees.len(), 1);
                assert_eq!(coord_leaves.len(), 4);
            }
            other => panic!(
                "expected single Join over all dims, got {other:?}"
            ),
        }
    }

    #[test]
    fn lone_dim_coord_is_just_a_leaf() {
        // No N-D var: the dim-coord group on its own collapses to a single
        // Leaf (no Join needed).
        let tree = GridJoinTree::build(vec![mk(
            "y",
            &["y"],
        )])
        .unwrap();
        assert!(matches!(
            tree,
            GridJoinTree::Leaf(_)
        ));
    }
}
