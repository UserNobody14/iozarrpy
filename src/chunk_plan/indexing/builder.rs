//! `GridJoinTreeBuilder` — entry point for compiling a Polars [`Expr`] (or any
//! other front-end query AST) into a [`GridJoinTree`] with resolved
//! per-dimension index ranges.
//!
//! The pipeline is:
//!
//! 1. [`crate::chunk_plan::compile_expr`] walks the [`Expr`] and emits an
//!    internal [`crate::chunk_plan::exprs::expr_plan::ExprPlan`] with
//!    `LazyArraySelection` constraints (no I/O).
//! 2. [`compile_into_builder_sync`] / [`compile_into_builder_async`] resolve
//!    those constraints against the backend (binary search on cached
//!    coordinate chunks) and accumulate the resulting per-dim index ranges
//!    as a disjunction of [`HyperRect`] atoms inside a
//!    [`GridJoinTreeBuilder`].
//! 3. [`GridJoinTreeBuilder::finalize`] groups the resolved variables by
//!    [`ChunkGridSignature`], projects the global atom list to each
//!    signature's dim subset, wraps [`GridJoinTree::Group`] nodes around
//!    each top-level child of [`ZarrMeta::root`], and returns the
//!    [`GridJoinTree`] consumed by the reader.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use polars::prelude::Expr;
use smallvec::SmallVec;
use zarrs::array::{ArraySubset, ChunkGrid};

use crate::chunk_plan::coord_resolve::{
    Expansion, resolve_value_range_async,
    resolve_value_range_sync,
};
use crate::chunk_plan::exprs::expr_plan::{
    ExprPlan, VarSet as ExprVarSet,
};
use crate::chunk_plan::indexing::grid_join_tree::{
    ChunkSubset, GridJoinTree,
    LeafGroup as OwnedGridGroup,
};
use crate::chunk_plan::indexing::index_set::RectangleSet;
use crate::chunk_plan::indexing::lazy_selection::{
    LazyArraySelection, LazyDimConstraint,
    LazyHyperRectangle,
};
use crate::chunk_plan::indexing::types::{
    ChunkGridSignature, ValueRangePresent,
};
use crate::errors::BackendError;
use crate::meta::ZarrMeta;
use crate::shared::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync, IStr,
};

// ============================================================================
// Public types
// ============================================================================

/// Statistics about a single planning run.
#[derive(Debug, Default, Clone)]
pub struct PlannerStats {
    /// Number of distinct dims for which at least one constraint was
    /// resolved against the backend.
    pub dims_resolved: usize,
    /// Number of variables referenced by the predicate.
    pub vars_referenced: usize,
}

/// Variable selection accumulator.
#[derive(Debug, Clone)]
pub enum VarSet {
    /// All variables in the dataset (the absorbing element for union and the
    /// identity for intersect).
    All,
    /// A specific (possibly empty) set of named variables.
    Specific(BTreeSet<IStr>),
}

impl VarSet {
    fn union(&self, other: &VarSet) -> VarSet {
        match (self, other) {
            (VarSet::All, _) | (_, VarSet::All) => {
                VarSet::All
            }
            (
                VarSet::Specific(a),
                VarSet::Specific(b),
            ) => {
                let mut out = a.clone();
                out.extend(b.iter().copied());
                VarSet::Specific(out)
            }
        }
    }
}

// ============================================================================
// HyperRect - one atom in the disjunction
// ============================================================================

/// A single resolved hyper-rectangle with per-dim index ranges.
///
/// A missing entry for some dim means "unconstrained along that dim"
/// (i.e., select the full extent). This lets us represent constraints that
/// reference only a subset of the dataset's dims and project them onto
/// specific arrays at finalize time.
#[derive(Debug, Clone, Default)]
struct HyperRect {
    dims: BTreeMap<IStr, Vec<Range<u64>>>,
}

impl HyperRect {
    /// The "everything" hyperrect — no dim constraints.
    fn full() -> Self {
        Self::default()
    }

    /// True iff the hyperrect represents the empty set.
    fn is_empty(&self) -> bool {
        self.dims.values().any(|ranges| {
            ranges.is_empty()
                || ranges
                    .iter()
                    .all(|r| r.start >= r.end)
        })
    }

    /// Cross-product intersection. Returns `None` if any shared-dim
    /// intersection is empty.
    fn intersect_with(
        &self,
        other: &HyperRect,
    ) -> Option<HyperRect> {
        let mut out = self.dims.clone();
        for (dim, other_ranges) in &other.dims {
            match out.get_mut(dim) {
                Some(existing) => {
                    let mut intersected: Vec<
                        Range<u64>,
                    > = Vec::new();
                    for a in existing.iter() {
                        for b in other_ranges.iter() {
                            let lo =
                                a.start.max(b.start);
                            let hi =
                                a.end.min(b.end);
                            if lo < hi {
                                intersected
                                    .push(lo..hi);
                            }
                        }
                    }
                    if intersected.is_empty() {
                        return None;
                    }
                    *existing = intersected;
                }
                None => {
                    out.insert(
                        *dim,
                        other_ranges.clone(),
                    );
                }
            }
        }
        let r = HyperRect { dims: out };
        if r.is_empty() { None } else { Some(r) }
    }
}

// ============================================================================
// Builder state
// ============================================================================

/// Internal builder state.
#[derive(Debug, Clone)]
enum BuilderState {
    /// No constraint has been applied yet (selects everything).
    NoConstraint,
    /// At least one constraint resolved to the empty set; downstream
    /// operations short-circuit until [`GridJoinTreeBuilder::finalize`].
    Empty,
    /// Active state: a disjunction of hyperrect atoms + a var set.
    Active {
        /// Disjunction of [`HyperRect`]s. Each atom is itself a
        /// cartesian product across its constrained dims.
        atoms: Vec<HyperRect>,
        vars: VarSet,
    },
}

// ============================================================================
// GridJoinTreeBuilder
// ============================================================================

/// Accumulator for a [`GridJoinTree`] under construction.
///
/// The builder is generic over the backend so the same call sites work for
/// both sync and async backends; constraint-add methods are gated behind
/// the appropriate trait bound.
pub struct GridJoinTreeBuilder<'a, B> {
    meta: &'a ZarrMeta,
    backend: &'a B,
    state: BuilderState,
}

impl<'a, B> GridJoinTreeBuilder<'a, B> {
    pub fn new(meta: &'a ZarrMeta, backend: &'a B) -> Self {
        Self {
            meta,
            backend,
            state: BuilderState::NoConstraint,
        }
    }

    /// Mark `name` as a referenced variable. Idempotent. No-op when the
    /// builder is in [`BuilderState::Empty`].
    pub fn add_var(&mut self, name: IStr) {
        match &mut self.state {
            BuilderState::Empty => {}
            BuilderState::NoConstraint => {
                let mut s = BTreeSet::new();
                s.insert(name);
                self.state = BuilderState::Active {
                    atoms: vec![HyperRect::full()],
                    vars: VarSet::Specific(s),
                };
            }
            BuilderState::Active {
                vars, ..
            } => match vars {
                VarSet::All => {}
                VarSet::Specific(s) => {
                    s.insert(name);
                }
            },
        }
    }

    /// Mark every dataset variable as referenced.
    pub fn add_all_vars(&mut self) {
        match &mut self.state {
            BuilderState::Empty => {}
            BuilderState::NoConstraint => {
                self.state = BuilderState::Active {
                    atoms: vec![HyperRect::full()],
                    vars: VarSet::All,
                };
            }
            BuilderState::Active {
                vars, ..
            } => {
                *vars = VarSet::All;
            }
        }
    }

    /// Force the builder into the empty state (selects nothing).
    pub fn set_empty(&mut self) {
        self.state = BuilderState::Empty;
    }

    /// AND of two constraint sets.
    pub fn intersect(
        &mut self,
        other: GridJoinTreeBuilder<'a, B>,
    ) {
        let other_state = other.state;
        let new = match (
            std::mem::replace(
                &mut self.state,
                BuilderState::Empty,
            ),
            other_state,
        ) {
            (BuilderState::Empty, _)
            | (_, BuilderState::Empty) => {
                BuilderState::Empty
            }
            (BuilderState::NoConstraint, x)
            | (x, BuilderState::NoConstraint) => x,
            (
                BuilderState::Active {
                    atoms: l_atoms,
                    vars: l_vars,
                },
                BuilderState::Active {
                    atoms: r_atoms,
                    vars: r_vars,
                },
            ) => {
                let mut new_atoms: Vec<HyperRect> =
                    Vec::new();
                for a in &l_atoms {
                    for b in &r_atoms {
                        if let Some(merged) =
                            a.intersect_with(b)
                        {
                            new_atoms.push(merged);
                        }
                    }
                }
                if new_atoms.is_empty() {
                    BuilderState::Empty
                } else {
                    BuilderState::Active {
                        atoms: new_atoms,
                        vars: l_vars.union(&r_vars),
                    }
                }
            }
        };
        self.state = new;
    }

    /// OR of two constraint sets. A `NoConstraint` side dominates.
    pub fn union(
        &mut self,
        other: GridJoinTreeBuilder<'a, B>,
    ) {
        let other_state = other.state;
        let new = match (
            std::mem::replace(
                &mut self.state,
                BuilderState::Empty,
            ),
            other_state,
        ) {
            (BuilderState::NoConstraint, _)
            | (_, BuilderState::NoConstraint) => {
                BuilderState::NoConstraint
            }
            (BuilderState::Empty, x)
            | (x, BuilderState::Empty) => x,
            (
                BuilderState::Active {
                    atoms: mut l_atoms,
                    vars: l_vars,
                },
                BuilderState::Active {
                    atoms: r_atoms,
                    vars: r_vars,
                },
            ) => {
                l_atoms.extend(r_atoms);
                BuilderState::Active {
                    atoms: l_atoms,
                    vars: l_vars.union(&r_vars),
                }
            }
        };
        self.state = new;
    }

    /// `self \ other` = `self ∩ ¬other`.
    pub fn difference(
        &mut self,
        other: GridJoinTreeBuilder<'a, B>,
    ) {
        let other_state = other.state;
        match (&self.state, &other_state) {
            (BuilderState::Empty, _) => {
                self.state = BuilderState::Empty;
            }
            (_, BuilderState::Empty) => {
                // self stays the same.
            }
            (_, BuilderState::NoConstraint) => {
                self.state = BuilderState::Empty;
            }
            (BuilderState::NoConstraint, _) => {
                let mut neg =
                    GridJoinTreeBuilder::<B> {
                        meta: self.meta,
                        backend: self.backend,
                        state: other_state,
                    };
                neg.negate();
                self.state = neg.state;
            }
            _ => {
                let mut neg =
                    GridJoinTreeBuilder::<B> {
                        meta: self.meta,
                        backend: self.backend,
                        state: other_state,
                    };
                neg.negate();
                self.intersect(neg);
            }
        }
    }

    /// In-place complement of the current state.
    pub fn negate(&mut self) {
        let state = std::mem::replace(
            &mut self.state,
            BuilderState::Empty,
        );
        self.state = match state {
            BuilderState::NoConstraint => {
                BuilderState::Empty
            }
            BuilderState::Empty => {
                BuilderState::NoConstraint
            }
            BuilderState::Active {
                atoms,
                vars,
            } => {
                // ¬(A1 ∨ A2 ∨ ... ∨ An) = ¬A1 ∧ ¬A2 ∧ ... ∧ ¬An
                let mut acc: Vec<HyperRect> =
                    vec![HyperRect::full()];
                for atom in &atoms {
                    let neg_atom_atoms: Vec<HyperRect> =
                        self.complement_of_atom(
                            atom,
                        );
                    let mut next: Vec<HyperRect> =
                        Vec::new();
                    for a in &acc {
                        for b in &neg_atom_atoms {
                            if let Some(merged) =
                                a.intersect_with(b)
                            {
                                next.push(merged);
                            }
                        }
                    }
                    acc = next;
                    if acc.is_empty() {
                        break;
                    }
                }
                if acc.is_empty() {
                    BuilderState::Empty
                } else {
                    BuilderState::Active {
                        atoms: acc,
                        vars,
                    }
                }
            }
        };
    }

    /// Complement of a single atom = union of "complement along one dim",
    /// across all constrained dims of the atom.
    fn complement_of_atom(
        &self,
        atom: &HyperRect,
    ) -> Vec<HyperRect> {
        if atom.dims.is_empty() {
            // Atom = "everything"; its complement is the empty set
            // (represented as no atoms).
            return Vec::new();
        }
        let mut out: Vec<HyperRect> = Vec::new();
        for (dim, ranges) in &atom.dims {
            let dim_len = self
                .lookup_dim_len(dim)
                .unwrap_or(u64::MAX);
            let comp =
                complement_one_dim(dim_len, ranges);
            if comp.is_empty() {
                continue;
            }
            let mut h = HyperRect::default();
            h.dims.insert(*dim, comp);
            out.push(h);
        }
        out
    }

    /// `(self \ other) ∪ (other \ self)`.
    pub fn exclusive_or(
        &mut self,
        other: GridJoinTreeBuilder<'a, B>,
    ) {
        let self_clone = self.state.clone();
        let other_clone = other.state.clone();

        let mut left =
            GridJoinTreeBuilder::<B> {
                meta: self.meta,
                backend: self.backend,
                state: self_clone,
            };
        let other_for_left =
            GridJoinTreeBuilder::<B> {
                meta: self.meta,
                backend: self.backend,
                state: other_clone,
            };
        left.difference(other_for_left);

        let mut right = GridJoinTreeBuilder::<B> {
            meta: self.meta,
            backend: self.backend,
            state: other.state,
        };
        let self_for_right =
            GridJoinTreeBuilder::<B> {
                meta: self.meta,
                backend: self.backend,
                state: std::mem::replace(
                    &mut self.state,
                    BuilderState::Empty,
                ),
            };
        right.difference(self_for_right);

        self.state = left.state;
        self.union(right);
    }

    /// Stats snapshot.
    pub fn stats(&self) -> PlannerStats {
        match &self.state {
            BuilderState::NoConstraint => {
                PlannerStats::default()
            }
            BuilderState::Empty => {
                PlannerStats::default()
            }
            BuilderState::Active {
                atoms,
                vars,
            } => {
                let mut dim_set: BTreeSet<IStr> =
                    BTreeSet::new();
                for atom in atoms {
                    dim_set.extend(
                        atom.dims.keys().copied(),
                    );
                }
                PlannerStats {
                    dims_resolved: dim_set.len(),
                    vars_referenced: match vars {
                        VarSet::All => self
                            .meta
                            .all_array_paths()
                            .len(),
                        VarSet::Specific(s) => {
                            s.len()
                        }
                    },
                }
            }
        }
    }

    /// Look up a dim's length.
    fn lookup_dim_len(
        &self,
        dim: &IStr,
    ) -> Option<u64> {
        self.meta
            .dim_analysis
            .dim_lengths
            .get(dim)
            .copied()
            .or_else(|| {
                self.meta
                    .array_by_path(*dim)
                    .and_then(|a| {
                        a.shape.first().copied()
                    })
            })
    }

    /// Materialize the accumulated state into a [`GridJoinTree`].
    pub fn finalize(
        self,
    ) -> Result<Option<GridJoinTree>, BackendError>
    {
        let GridJoinTreeBuilder {
            meta,
            backend: _,
            state,
        } = self;

        let groups = match state {
            BuilderState::Empty => {
                return Ok(None);
            }
            BuilderState::NoConstraint => {
                build_groups_from_active(
                    meta,
                    &[HyperRect::full()],
                    &VarSet::All,
                )?
            }
            BuilderState::Active {
                atoms,
                vars,
            } => build_groups_from_active(
                meta, &atoms, &vars,
            )?,
        };

        if groups.is_empty() {
            return Ok(None);
        }

        let Some(tree) = GridJoinTree::build(groups)
        else {
            return Ok(None);
        };

        Ok(Some(wrap_root_groups(tree, meta)))
    }
}

impl<'a, B: ChunkedDataBackendSync>
    GridJoinTreeBuilder<'a, B>
{
    /// Resolve a value-range constraint synchronously and intersect into
    /// the per-dim accumulator.
    pub fn add_constraint(
        &mut self,
        dim: IStr,
        vr: &ValueRangePresent,
        expansion: Expansion,
    ) -> Result<(), BackendError> {
        let dim_len = self
            .lookup_dim_len(&dim)
            .ok_or_else(|| {
                BackendError::other(format!(
                    "unknown dim '{}' (no dim length \
                     in meta and no coordinate \
                     array)",
                    AsRef::<str>::as_ref(&dim),
                ))
            })?;
        let ranges = resolve_value_range_sync(
            self.backend,
            &dim,
            self.meta,
            dim_len,
            vr,
            expansion,
        )
        .map_err(|e| {
            BackendError::other(format!(
                "value-range resolution failed for \
                 dim '{}': {e}",
                AsRef::<str>::as_ref(&dim),
            ))
        })?;
        self.intersect_dim(dim, ranges);
        Ok(())
    }
}

impl<'a, B: ChunkedDataBackendAsync>
    GridJoinTreeBuilder<'a, B>
{
    /// Resolve a value-range constraint asynchronously.
    pub async fn add_constraint_async(
        &mut self,
        dim: IStr,
        vr: &ValueRangePresent,
        expansion: Expansion,
    ) -> Result<(), BackendError> {
        let dim_len = self
            .lookup_dim_len(&dim)
            .ok_or_else(|| {
                BackendError::other(format!(
                    "unknown dim '{}' (no dim length \
                     in meta and no coordinate \
                     array)",
                    AsRef::<str>::as_ref(&dim),
                ))
            })?;
        let ranges = resolve_value_range_async(
            self.backend,
            &dim,
            self.meta,
            dim_len,
            vr,
            expansion,
        )
        .await
        .map_err(|e| {
            BackendError::other(format!(
                "value-range resolution failed for \
                 dim '{}': {e}",
                AsRef::<str>::as_ref(&dim),
            ))
        })?;
        self.intersect_dim(dim, ranges);
        Ok(())
    }
}

impl<'a, B> GridJoinTreeBuilder<'a, B> {
    fn intersect_dim(
        &mut self,
        dim: IStr,
        ranges: Vec<Range<u64>>,
    ) {
        if ranges.is_empty() {
            self.state = BuilderState::Empty;
            return;
        }
        match &mut self.state {
            BuilderState::Empty => {}
            BuilderState::NoConstraint => {
                let mut atom = HyperRect::default();
                atom.dims.insert(dim, ranges);
                self.state = BuilderState::Active {
                    atoms: vec![atom],
                    vars: VarSet::Specific(
                        BTreeSet::new(),
                    ),
                };
            }
            BuilderState::Active {
                atoms,
                ..
            } => {
                let mut new_atoms: Vec<HyperRect> =
                    Vec::new();
                for atom in atoms.iter() {
                    let mut other =
                        HyperRect::default();
                    other.dims.insert(
                        dim,
                        ranges.clone(),
                    );
                    if let Some(merged) =
                        atom.intersect_with(&other)
                    {
                        new_atoms.push(merged);
                    }
                }
                if new_atoms.is_empty() {
                    self.state =
                        BuilderState::Empty;
                } else {
                    *atoms = new_atoms;
                }
            }
        }
    }
}

// ============================================================================
// Free helpers
// ============================================================================

/// Per-dim 1D set complement against `0..dim_len`.
fn complement_one_dim(
    dim_len: u64,
    ranges: &[Range<u64>],
) -> Vec<Range<u64>> {
    let mut sorted: Vec<Range<u64>> =
        ranges.iter().cloned().collect();
    sorted.sort_by_key(|r| r.start);
    let mut out = Vec::new();
    let mut cur = 0u64;
    for r in sorted {
        if r.start > cur {
            out.push(cur..r.start);
        }
        cur = cur.max(r.end);
    }
    if cur < dim_len {
        out.push(cur..dim_len);
    }
    out
}

// ============================================================================
// Active-state -> Vec<OwnedGridGroup>
// ============================================================================

/// Produce the owned grid groups for an active accumulator.
///
/// `atoms` is a disjunction of [`HyperRect`]s in the *global* dim space;
/// per-array projection drops constraints on dims the array doesn't have
/// and fills in unconstrained dims with the array's full extent.
fn build_groups_from_active(
    meta: &ZarrMeta,
    atoms: &[HyperRect],
    vars: &VarSet,
) -> Result<Vec<OwnedGridGroup>, BackendError> {
    let var_list: Vec<IStr> = match vars {
        VarSet::Specific(s) if s.is_empty() => {
            return Ok(Vec::new());
        }
        VarSet::All => meta.all_array_paths(),
        VarSet::Specific(s) => {
            s.iter().copied().collect()
        }
    };
    if var_list.is_empty() {
        return Ok(Vec::new());
    }

    // Group vars by full (dims + chunk + shard) signature.
    let mut sig_cache: BTreeMap<
        ChunkGridSignature,
        Arc<ChunkGridSignature>,
    > = BTreeMap::new();
    let mut by_sig: BTreeMap<
        Arc<ChunkGridSignature>,
        Vec<IStr>,
    > = BTreeMap::new();
    for var in &var_list {
        let Some(arr_meta) =
            meta.array_by_path(*var)
        else {
            continue;
        };
        let zeros: Vec<u64> =
            vec![0u64; arr_meta.shape.len()];
        let outer_chunk_shape: Option<
            SmallVec<[u64; 4]>,
        > = arr_meta
            .outer_chunk_grid
            .chunk_shape(&zeros)
            .map_err(|e| {
                BackendError::other(format!(
                    "outer chunk shape for '{}': {e:?}",
                    AsRef::<str>::as_ref(var),
                ))
            })?
            .map(|v| {
                v.into_iter()
                    .map(|n| n.get())
                    .collect()
            });
        let inner_chunk_shape: Option<
            SmallVec<[u64; 4]>,
        > = match arr_meta
            .inner_chunk_grid
            .as_ref()
        {
            Some(grid) => grid
                .chunk_shape(&zeros)
                .map_err(|e| {
                    BackendError::other(format!(
                        "inner chunk shape for '{}': {e:?}",
                        AsRef::<str>::as_ref(var),
                    ))
                })?
                .map(|v| {
                    v.into_iter()
                        .map(|n| n.get())
                        .collect()
                }),
            None => None,
        };
        let sig = ChunkGridSignature::new(
            arr_meta.dims.clone(),
            outer_chunk_shape,
            inner_chunk_shape,
        )?;
        let sig_arc = sig_cache
            .entry(sig.clone())
            .or_insert_with(|| Arc::new(sig))
            .clone();
        by_sig
            .entry(sig_arc)
            .or_default()
            .push(*var);
    }

    let mut groups: Vec<OwnedGridGroup> = Vec::new();
    for (sig, sig_vars) in by_sig {
        let dims = sig.dims();
        let dim_len_vec: Vec<u64> = dims
            .iter()
            .map(|d| {
                meta.dim_analysis
                    .dim_lengths
                    .get(d)
                    .copied()
                    .or_else(|| {
                        meta.array_by_path(*d)
                            .and_then(|a| {
                                a.shape
                                    .first()
                                    .copied()
                            })
                    })
                    .ok_or_else(|| {
                        BackendError::other(
                            format!(
                                "no length for dim \
                                 '{}'",
                                AsRef::<str>::as_ref(d),
                            ),
                        )
                    })
            })
            .collect::<Result<_, _>>()?;

        let dims_sv: SmallVec<[IStr; 4]> =
            dims.iter().copied().collect();
        let shape_sv: SmallVec<[u64; 4]> =
            dim_len_vec.iter().copied().collect();

        // Project each atom to the signature's dim subset.
        let mut rect_set = RectangleSet::empty(
            dims_sv.clone(),
            shape_sv.clone(),
        );
        for atom in atoms {
            let per_dim: SmallVec<
                [Vec<Range<u64>>; 4],
            > = dims
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    atom.dims.get(d).cloned().unwrap_or_else(
                        || vec![0..dim_len_vec[i]],
                    )
                })
                .collect();
            // Skip atoms that contain an empty range
            // (would yield an empty rectangle anyway).
            if per_dim.iter().any(|r| r.is_empty()) {
                continue;
            }
            let one = RectangleSet::from_per_dim(
                dims_sv.clone(),
                shape_sv.clone(),
                per_dim,
            );
            rect_set = rect_set.union(&one);
        }

        let subsets: Vec<ArraySubset> = rect_set
            .iter_subsets()
            .collect();
        if subsets.is_empty() {
            continue;
        }

        // Derive the chunk grid for this signature group.
        let representative = sig_vars[0];
        let arr_meta = meta
            .array_by_path(representative)
            .ok_or_else(|| {
                BackendError::other(format!(
                    "no array meta for '{}'",
                    AsRef::<str>::as_ref(
                        &representative,
                    ),
                ))
            })?;
        let chunk_grid: Arc<ChunkGrid> = match &arr_meta
            .inner_chunk_grid
        {
            Some(g) => g.clone(),
            None => arr_meta.outer_chunk_grid.clone(),
        };
        let array_shape: Vec<u64> =
            chunk_grid.array_shape().to_vec();
        let chunk_shape: Vec<u64> = arr_meta
            .chunk_shape
            .iter()
            .copied()
            .collect();

        let mut seen: BTreeSet<Vec<u64>> =
            BTreeSet::new();
        for subset in &subsets {
            let Some(indices) = chunk_grid
                .chunks_in_array_subset(subset)
                .map_err(|e| {
                    BackendError::other(format!(
                        "chunks_in_array_subset \
                         failed: {e:?}",
                    ))
                })?
            else {
                continue;
            };
            for idx in indices.indices() {
                seen.insert(idx.to_vec());
            }
        }
        let chunk_indices: Vec<Vec<u64>> =
            seen.into_iter().collect();
        let chunk_subsets: Vec<Option<ChunkSubset>> =
            chunk_indices
                .iter()
                .map(|idx| {
                    compute_chunk_subset_local(
                        idx,
                        &chunk_shape,
                        &array_shape,
                        &subsets,
                    )
                })
                .collect();

        groups.push(OwnedGridGroup::new(
            sig,
            sig_vars,
            chunk_indices,
            chunk_subsets,
            array_shape,
        ));
    }

    // Drop redundant 1D dim-coord groups: when a 1D group is just the dim
    // coordinate (vars = [dim]) and another group in this plan already
    // covers that dim, the multi-dim group's reader will materialize the
    // dim column from the coord array, so the standalone group adds
    // duplicate reads and forces an extra `Independent` concat.
    let other_groups_dims: BTreeSet<IStr> = groups
        .iter()
        .filter(|g| g.sig.dims().len() > 1)
        .flat_map(|g| {
            g.sig.dims().iter().copied().collect::<Vec<_>>()
        })
        .collect();
    let groups: Vec<OwnedGridGroup> = groups
        .into_iter()
        .filter(|g| {
            let dims = g.sig.dims();
            if dims.len() != 1 {
                return true;
            }
            let dim = dims[0];
            if !other_groups_dims.contains(&dim) {
                return true;
            }
            !(g.vars.len() == 1
                && g.vars[0] == dim)
        })
        .collect();

    Ok(groups)
}

/// Compute the chunk-local subset for a given chunk index.
fn compute_chunk_subset_local(
    chunk_idx: &[u64],
    chunk_shape: &[u64],
    array_shape: &[u64],
    subsets: &[ArraySubset],
) -> Option<ChunkSubset> {
    let ndim = chunk_idx.len();

    let chunk_start: Vec<u64> = chunk_idx
        .iter()
        .zip(chunk_shape)
        .map(|(i, s)| i * s)
        .collect();
    let chunk_end: Vec<u64> = chunk_start
        .iter()
        .zip(chunk_shape)
        .zip(array_shape)
        .map(|((s, cs), a)| (s + cs).min(*a))
        .collect();

    let mut bbox_start: Vec<u64> = chunk_end.clone();
    let mut bbox_end: Vec<u64> = chunk_start.clone();

    for subset in subsets {
        let ranges = subset.to_ranges();
        for d in 0..ndim {
            let inter_start =
                ranges[d].start.max(chunk_start[d]);
            let inter_end =
                ranges[d].end.min(chunk_end[d]);
            if inter_start < inter_end {
                bbox_start[d] =
                    bbox_start[d].min(inter_start);
                bbox_end[d] =
                    bbox_end[d].max(inter_end);
            }
        }
    }

    let local_ranges: Vec<Range<u64>> = bbox_start
        .iter()
        .zip(bbox_end.iter())
        .zip(chunk_start.iter())
        .map(|((s, e), cs)| (s - cs)..(e - cs))
        .collect();
    let actual_chunk_shape: Vec<u64> = chunk_end
        .iter()
        .zip(chunk_start.iter())
        .map(|(e, s)| e - s)
        .collect();

    let subset =
        ChunkSubset::from_ranges(local_ranges);
    if is_full_chunk_local(
        &subset,
        &actual_chunk_shape,
    ) {
        None
    } else {
        Some(subset)
    }
}

/// Mirror of the (crate-private) `ChunkSubset::is_full_chunk` check.
fn is_full_chunk_local(
    subset: &ChunkSubset,
    chunk_shape: &[u64],
) -> bool {
    subset.ranges.iter().zip(chunk_shape).all(
        |(r, &s)| r.start == 0 && r.end >= s,
    )
}

// ============================================================================
// meta.root.children -> GridJoinTree::Group wrapping
// ============================================================================

/// Wrap the tree with one [`GridJoinTree::Group`] node per top-level child of
/// [`ZarrMeta::root`].
fn wrap_root_groups(
    tree: GridJoinTree,
    meta: &ZarrMeta,
) -> GridJoinTree {
    let mut out = tree;
    for child_name in meta.root.children.keys() {
        out = GridJoinTree::Group {
            name: *child_name,
            child: Box::new(out),
        };
    }
    out
}

// ============================================================================
// Top-level entry points
// ============================================================================

/// Compile a Polars [`Expr`] into a [`GridJoinTree`] synchronously.
pub fn compile_to_tree_sync<
    B: ChunkedDataBackendSync,
>(
    expr: &Expr,
    meta: &ZarrMeta,
    backend: &B,
) -> Result<
    (Option<GridJoinTree>, PlannerStats),
    BackendError,
> {
    let mut builder =
        GridJoinTreeBuilder::new(meta, backend);
    compile_into_builder_sync(expr, &mut builder)?;
    let stats = builder.stats();
    let tree = builder.finalize()?;
    Ok((tree, stats))
}

/// Async mirror of [`compile_to_tree_sync`].
pub async fn compile_to_tree_async<
    B: ChunkedDataBackendAsync,
>(
    expr: &Expr,
    meta: &ZarrMeta,
    backend: &B,
) -> Result<
    (Option<GridJoinTree>, PlannerStats),
    BackendError,
> {
    let mut builder =
        GridJoinTreeBuilder::new(meta, backend);
    compile_into_builder_async(
        expr, &mut builder,
    )
    .await?;
    let stats = builder.stats();
    let tree = builder.finalize()?;
    Ok((tree, stats))
}

/// Drive `compile_expr` and walk the resulting [`ExprPlan`] into the
/// builder, resolving every dim constraint against the backend.
fn compile_into_builder_sync<
    B: ChunkedDataBackendSync,
>(
    expr: &Expr,
    builder: &mut GridJoinTreeBuilder<'_, B>,
) -> Result<(), BackendError> {
    use crate::chunk_plan::LazyCompileCtx;
    use crate::chunk_plan::compile_expr;
    use crate::chunk_plan::compute_dims_and_lengths_unified;

    let meta = builder.meta;
    let (dims, _) =
        compute_dims_and_lengths_unified(meta);
    let mut ctx =
        LazyCompileCtx::new(meta, &dims);
    let plan = compile_expr(expr, &mut ctx)?;
    apply_plan_sync(&plan, builder)
}

async fn compile_into_builder_async<
    B: ChunkedDataBackendAsync,
>(
    expr: &Expr,
    builder: &mut GridJoinTreeBuilder<'_, B>,
) -> Result<(), BackendError> {
    use crate::chunk_plan::LazyCompileCtx;
    use crate::chunk_plan::compile_expr;
    use crate::chunk_plan::compute_dims_and_lengths_unified;

    let meta = builder.meta;
    let (dims, _) =
        compute_dims_and_lengths_unified(meta);
    let mut ctx =
        LazyCompileCtx::new(meta, &dims);
    let plan = compile_expr(expr, &mut ctx)?;
    apply_plan_async(&plan, builder).await
}

// ============================================================================
// ExprPlan -> builder walker
// ============================================================================

fn apply_plan_sync<B: ChunkedDataBackendSync>(
    plan: &ExprPlan,
    builder: &mut GridJoinTreeBuilder<'_, B>,
) -> Result<(), BackendError> {
    match plan {
        ExprPlan::NoConstraint => {
            builder.add_all_vars();
            Ok(())
        }
        ExprPlan::Empty => {
            builder.set_empty();
            Ok(())
        }
        ExprPlan::Active {
            vars,
            constraints,
        } => {
            let mut sub = compile_lazy_selection_sync(
                constraints,
                builder.meta,
                builder.backend,
            )?;
            apply_vars_to_builder(
                &mut sub, vars,
            );
            builder.intersect(sub);
            Ok(())
        }
    }
}

async fn apply_plan_async<
    'p,
    'a,
    B: ChunkedDataBackendAsync,
>(
    plan: &'p ExprPlan,
    builder: &mut GridJoinTreeBuilder<'a, B>,
) -> Result<(), BackendError>
where
    'a: 'p,
{
    match plan {
        ExprPlan::NoConstraint => {
            builder.add_all_vars();
            Ok(())
        }
        ExprPlan::Empty => {
            builder.set_empty();
            Ok(())
        }
        ExprPlan::Active {
            vars,
            constraints,
        } => {
            let mut sub = compile_lazy_selection_async(
                constraints,
                builder.meta,
                builder.backend,
            )
            .await?;
            apply_vars_to_builder(
                &mut sub, vars,
            );
            builder.intersect(sub);
            Ok(())
        }
    }
}

fn apply_vars_to_builder<B>(
    builder: &mut GridJoinTreeBuilder<'_, B>,
    vars: &ExprVarSet,
) {
    match vars {
        ExprVarSet::All => builder.add_all_vars(),
        ExprVarSet::Specific(items) => {
            if items.is_empty() {
                // Empty var-set with constraints means no rows will be
                // produced because there is nothing to read.
                builder.set_empty();
            } else {
                for v in items {
                    builder.add_var(*v);
                }
            }
        }
    }
}

fn compile_lazy_selection_sync<
    'a,
    B: ChunkedDataBackendSync,
>(
    sel: &LazyArraySelection,
    meta: &'a ZarrMeta,
    backend: &'a B,
) -> Result<GridJoinTreeBuilder<'a, B>, BackendError>
{
    match sel {
        LazyArraySelection::Rectangles(rects) => {
            let mut acc =
                GridJoinTreeBuilder::new(
                    meta, backend,
                );
            acc.set_empty();
            for rect in rects {
                let rect_b =
                    compile_rectangle_sync(
                        rect, meta, backend,
                    )?;
                acc.union(rect_b);
            }
            Ok(acc)
        }
        LazyArraySelection::Difference(a, b) => {
            let mut a_b =
                compile_lazy_selection_sync(
                    a, meta, backend,
                )?;
            let b_b =
                compile_lazy_selection_sync(
                    b, meta, backend,
                )?;
            a_b.difference(b_b);
            Ok(a_b)
        }
        LazyArraySelection::Union(a, b) => {
            let mut a_b =
                compile_lazy_selection_sync(
                    a, meta, backend,
                )?;
            let b_b =
                compile_lazy_selection_sync(
                    b, meta, backend,
                )?;
            a_b.union(b_b);
            Ok(a_b)
        }
        LazyArraySelection::BooleanNot(inner) => {
            let mut b =
                compile_lazy_selection_sync(
                    inner, meta, backend,
                )?;
            b.negate();
            Ok(b)
        }
    }
}

async fn compile_lazy_selection_async<
    'sel,
    'a,
    B: ChunkedDataBackendAsync,
>(
    sel: &'sel LazyArraySelection,
    meta: &'a ZarrMeta,
    backend: &'a B,
) -> Result<GridJoinTreeBuilder<'a, B>, BackendError>
where
    'a: 'sel,
{
    fn rec<
        'sel,
        'a,
        B: ChunkedDataBackendAsync,
    >(
        sel: &'sel LazyArraySelection,
        meta: &'a ZarrMeta,
        backend: &'a B,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                Output = Result<
                    GridJoinTreeBuilder<'a, B>,
                    BackendError,
                >,
            > + Send + 'sel,
        >,
    >
    where
        'a: 'sel,
    {
        Box::pin(async move {
            match sel {
                LazyArraySelection::Rectangles(
                    rects,
                ) => {
                    let mut acc =
                        GridJoinTreeBuilder::new(
                            meta, backend,
                        );
                    acc.set_empty();
                    for rect in rects {
                        let rect_b =
                            compile_rectangle_async(
                                rect, meta,
                                backend,
                            )
                            .await?;
                        acc.union(rect_b);
                    }
                    Ok(acc)
                }
                LazyArraySelection::Difference(
                    a,
                    b,
                ) => {
                    let mut a_b =
                        rec(a, meta, backend)
                            .await?;
                    let b_b =
                        rec(b, meta, backend)
                            .await?;
                    a_b.difference(b_b);
                    Ok(a_b)
                }
                LazyArraySelection::Union(a, b) => {
                    let mut a_b =
                        rec(a, meta, backend)
                            .await?;
                    let b_b =
                        rec(b, meta, backend)
                            .await?;
                    a_b.union(b_b);
                    Ok(a_b)
                }
                LazyArraySelection::BooleanNot(
                    inner,
                ) => {
                    let mut b =
                        rec(inner, meta, backend)
                            .await?;
                    b.negate();
                    Ok(b)
                }
            }
        })
    }
    rec(sel, meta, backend).await
}

fn compile_rectangle_sync<
    'a,
    B: ChunkedDataBackendSync,
>(
    rect: &LazyHyperRectangle,
    meta: &'a ZarrMeta,
    backend: &'a B,
) -> Result<GridJoinTreeBuilder<'a, B>, BackendError>
{
    let mut b =
        GridJoinTreeBuilder::new(meta, backend);
    if rect.is_empty() {
        b.set_empty();
        return Ok(b);
    }
    if rect.is_all() {
        // Unconstrained rect: add a single "full" atom so subsequent
        // intersections / unions use Active state, not NoConstraint.
        b.state = BuilderState::Active {
            atoms: vec![HyperRect::full()],
            vars: VarSet::Specific(BTreeSet::new()),
        };
        return Ok(b);
    }
    for (dim, constraint) in rect.dims() {
        match constraint {
            LazyDimConstraint::All => {}
            LazyDimConstraint::Empty => {
                b.set_empty();
                return Ok(b);
            }
            LazyDimConstraint::Unresolved(vr) => {
                b.add_constraint(
                    *dim,
                    vr,
                    Expansion::Exact,
                )?;
            }
            LazyDimConstraint::InterpolationRange(
                vr,
            ) => {
                b.add_constraint(
                    *dim,
                    vr,
                    Expansion::InterpolationNeighbor,
                )?;
            }
            LazyDimConstraint::WrappingInterpolationRange(
                vr,
            ) => {
                b.add_constraint(
                    *dim,
                    vr,
                    Expansion::WrappingGhost,
                )?;
            }
        }
    }
    Ok(b)
}

async fn compile_rectangle_async<
    'a,
    B: ChunkedDataBackendAsync,
>(
    rect: &'_ LazyHyperRectangle,
    meta: &'a ZarrMeta,
    backend: &'a B,
) -> Result<GridJoinTreeBuilder<'a, B>, BackendError>
{
    let mut b =
        GridJoinTreeBuilder::new(meta, backend);
    if rect.is_empty() {
        b.set_empty();
        return Ok(b);
    }
    if rect.is_all() {
        b.state = BuilderState::Active {
            atoms: vec![HyperRect::full()],
            vars: VarSet::Specific(BTreeSet::new()),
        };
        return Ok(b);
    }
    for (dim, constraint) in rect.dims() {
        match constraint {
            LazyDimConstraint::All => {}
            LazyDimConstraint::Empty => {
                b.set_empty();
                return Ok(b);
            }
            LazyDimConstraint::Unresolved(vr) => {
                b.add_constraint_async(
                    *dim,
                    vr,
                    Expansion::Exact,
                )
                .await?;
            }
            LazyDimConstraint::InterpolationRange(
                vr,
            ) => {
                b.add_constraint_async(
                    *dim,
                    vr,
                    Expansion::InterpolationNeighbor,
                )
                .await?;
            }
            LazyDimConstraint::WrappingInterpolationRange(
                vr,
            ) => {
                b.add_constraint_async(
                    *dim,
                    vr,
                    Expansion::WrappingGhost,
                )
                .await?;
            }
        }
    }
    Ok(b)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complement_one_dim_basic() {
        let c = complement_one_dim(
            10,
            &[2..4, 6..8],
        );
        assert_eq!(c, vec![0..2, 4..6, 8..10]);
    }

    #[test]
    fn complement_one_dim_empty_input_full() {
        let c = complement_one_dim(5, &[]);
        assert_eq!(c, vec![0..5]);
    }

    #[test]
    fn hyperrect_intersect_disjoint_dims() {
        let mut a = HyperRect::default();
        a.dims.insert(
            "a".to_string().as_str().into(),
            vec![0..5],
        );
        let mut b = HyperRect::default();
        b.dims.insert(
            "b".to_string().as_str().into(),
            vec![0..5],
        );
        let merged = a.intersect_with(&b).unwrap();
        assert_eq!(merged.dims.len(), 2);
    }

    #[test]
    fn hyperrect_intersect_overlapping_same_dim() {
        let mut a = HyperRect::default();
        a.dims.insert(
            "x".to_string().as_str().into(),
            vec![0..5],
        );
        let mut b = HyperRect::default();
        b.dims.insert(
            "x".to_string().as_str().into(),
            vec![3..10],
        );
        let merged = a.intersect_with(&b).unwrap();
        assert_eq!(merged.dims.len(), 1);
        let v = merged
            .dims
            .values()
            .next()
            .unwrap()
            .clone();
        assert_eq!(v, vec![3..5]);
    }

    #[test]
    fn hyperrect_intersect_disjoint_same_dim_returns_none()
     {
        let mut a = HyperRect::default();
        a.dims.insert(
            "x".to_string().as_str().into(),
            vec![0..3],
        );
        let mut b = HyperRect::default();
        b.dims.insert(
            "x".to_string().as_str().into(),
            vec![5..10],
        );
        assert!(a.intersect_with(&b).is_none());
    }
}
