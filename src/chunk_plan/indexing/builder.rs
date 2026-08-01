//! `GridJoinTreeBuilder` — entry point for compiling a Polars [`Expr`] into
//! a [`GridJoinTree`] with resolved per-dimension index ranges.
//!
//! Pipeline:
//!
//! 1. [`crate::chunk_plan::compile_expr`] walks the [`Expr`] and emits an
//!    internal [`crate::chunk_plan::exprs::expr_plan::ExprPlan`] with a
//!    [`LazyArraySelection`] tree of unresolved value-range constraints.
//! 2. [`compile_into_builder_sync`] / [`compile_into_builder_async`] resolve
//!    those constraints against the backend (binary search on cached
//!    coordinate chunks) and accumulate the result as a [`RectangleSet`]
//!    over the dataset's full dim universe inside a
//!    [`GridJoinTreeBuilder`].
//! 3. [`GridJoinTreeBuilder::finalize`] groups the resolved variables by
//!    [`ChunkGridSignature`], projects the global rectangle set onto each
//!    signature's dim subset, wraps [`GridJoinTree::Group`] nodes around
//!    each top-level child of [`ZarrMeta::root`], and returns the
//!    [`GridJoinTree`] consumed by the reader.
//!
//! All boolean set algebra (union/intersect/difference/negate/xor) is
//! delegated to [`RectangleSet`] in [`crate::chunk_plan::indexing::index_set`];
//! this file only translates from the lazy expression IR into builder calls
//! and from the resolved global rectangle set into per-array chunk plans.

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
///
/// Returned alongside the [`GridJoinTree`] for diagnostics and tests; not
/// inspected by the reader hot path.
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct PlannerStats {
    /// Number of distinct dims for which at least one constraint was
    /// resolved against the backend (i.e., not full-extent).
    pub dims_resolved: usize,
    /// Number of variables referenced by the predicate.
    pub vars_referenced: usize,
}

/// Variable selection accumulator.
///
/// Distinct from [`ExprVarSet`] because the builder is consumed by the
/// reader, which needs an explicit `All` marker to expand at finalize time.
#[derive(Debug, Clone)]
pub enum VarSet {
    All,
    Specific(BTreeSet<IStr>),
}

impl VarSet {
    fn union(&self, other: &VarSet) -> VarSet {
        match (self, other) {
            (VarSet::All, _)
            | (_, VarSet::All) => VarSet::All,
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
// Builder state
// ============================================================================

/// Internal builder state.
///
/// `NoConstraint` is the identity for intersect and the absorbing element
/// for union; `Empty` is the absorbing element for intersect and the
/// identity for union. Promoting to `Active` is delayed until at least one
/// real constraint or var-reference is recorded so we don't allocate the
/// full-cube rectangle set unnecessarily.
#[derive(Debug, Clone)]
enum BuilderState {
    NoConstraint,
    Empty,
    Active { rects: RectangleSet, vars: VarSet },
}

// ============================================================================
// GridJoinTreeBuilder
// ============================================================================

/// Accumulator for a [`GridJoinTree`] under construction.
///
/// Generic over the backend so the same call sites work for both sync and
/// async backends; `add_constraint*` is gated behind the appropriate
/// trait bound. Internally everything operates on a single global
/// [`RectangleSet`] over [`DimensionAnalysis::all_dims`] — per-array
/// projection happens once at [`finalize`](Self::finalize).
pub struct GridJoinTreeBuilder<'a, B> {
    meta: &'a ZarrMeta,
    backend: &'a B,
    /// Global dim ordering used by the active [`RectangleSet`].
    dims: SmallVec<[IStr; 4]>,
    /// Length of each dim, in [`Self::dims`] order.
    shape: SmallVec<[u64; 4]>,
    state: BuilderState,
}

impl<'a, B> GridJoinTreeBuilder<'a, B> {
    pub fn new(
        meta: &'a ZarrMeta,
        backend: &'a B,
    ) -> Self {
        let dims: SmallVec<[IStr; 4]> = meta
            .dim_analysis
            .all_dims
            .iter()
            .copied()
            .collect();
        let shape: SmallVec<[u64; 4]> = dims
            .iter()
            .map(|d| {
                lookup_dim_len(meta, d)
                    .unwrap_or(0)
            })
            .collect();
        Self {
            meta,
            backend,
            dims,
            shape,
            state: BuilderState::NoConstraint,
        }
    }

    fn full_set(&self) -> RectangleSet {
        RectangleSet::full(
            self.dims.clone(),
            self.shape.clone(),
        )
    }

    /// Mark `name` as a referenced variable. Idempotent. No-op when the
    /// builder is in [`BuilderState::Empty`].
    pub fn add_var(&mut self, name: IStr) {
        match &mut self.state {
            BuilderState::Empty => {}
            BuilderState::NoConstraint => {
                let mut s = BTreeSet::new();
                s.insert(name);
                self.state =
                    BuilderState::Active {
                        rects: self.full_set(),
                        vars: VarSet::Specific(s),
                    };
            }
            BuilderState::Active {
                vars, ..
            } => {
                if let VarSet::Specific(s) = vars
                {
                    s.insert(name);
                }
            }
        }
    }

    /// Mark every dataset variable as referenced.
    pub fn add_all_vars(&mut self) {
        match &mut self.state {
            BuilderState::Empty => {}
            BuilderState::NoConstraint => {
                self.state =
                    BuilderState::Active {
                        rects: self.full_set(),
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
        let me = std::mem::replace(
            &mut self.state,
            BuilderState::Empty,
        );
        self.state = match (me, other.state) {
            (BuilderState::Empty, _)
            | (_, BuilderState::Empty) => {
                BuilderState::Empty
            }
            (BuilderState::NoConstraint, x)
            | (x, BuilderState::NoConstraint) => {
                x
            }
            (
                BuilderState::Active {
                    rects: a,
                    vars: va,
                },
                BuilderState::Active {
                    rects: b,
                    vars: vb,
                },
            ) => {
                let r = a.intersect(&b);
                if r.is_empty() {
                    BuilderState::Empty
                } else {
                    BuilderState::Active {
                        rects: r,
                        vars: va.union(&vb),
                    }
                }
            }
        };
    }

    /// OR of two constraint sets. A `NoConstraint` side dominates
    /// (everything ∪ X = everything).
    pub fn union(
        &mut self,
        other: GridJoinTreeBuilder<'a, B>,
    ) {
        let me = std::mem::replace(
            &mut self.state,
            BuilderState::Empty,
        );
        self.state = match (me, other.state) {
            (BuilderState::NoConstraint, _)
            | (_, BuilderState::NoConstraint) => {
                BuilderState::NoConstraint
            }
            (BuilderState::Empty, x)
            | (x, BuilderState::Empty) => x,
            (
                BuilderState::Active {
                    rects: a,
                    vars: va,
                },
                BuilderState::Active {
                    rects: b,
                    vars: vb,
                },
            ) => BuilderState::Active {
                rects: a.union(&b),
                vars: va.union(&vb),
            },
        };
    }

    /// `self \ other`.
    pub fn difference(
        &mut self,
        other: GridJoinTreeBuilder<'a, B>,
    ) {
        let me = std::mem::replace(
            &mut self.state,
            BuilderState::Empty,
        );
        self.state = match (me, other.state) {
            (BuilderState::Empty, _) => {
                BuilderState::Empty
            }
            (x, BuilderState::Empty) => x,
            (_, BuilderState::NoConstraint) => {
                BuilderState::Empty
            }
            (BuilderState::NoConstraint, b) => {
                let other_rects = match b {
                    BuilderState::Active {
                        rects,
                        ..
                    } => rects,
                    _ => unreachable!(),
                };
                let r = self
                    .full_set()
                    .difference(&other_rects);
                active_or_empty(
                    r,
                    VarSet::Specific(
                        BTreeSet::new(),
                    ),
                )
            }
            (
                BuilderState::Active {
                    rects: a,
                    vars: va,
                },
                BuilderState::Active {
                    rects: b,
                    vars: vb,
                },
            ) => active_or_empty(
                a.difference(&b),
                va.union(&vb),
            ),
        };
    }

    /// In-place complement of the current state.
    pub fn negate(&mut self) {
        let me = std::mem::replace(
            &mut self.state,
            BuilderState::Empty,
        );
        self.state = match me {
            BuilderState::NoConstraint => {
                BuilderState::Empty
            }
            BuilderState::Empty => {
                BuilderState::NoConstraint
            }
            BuilderState::Active {
                rects,
                vars,
            } => active_or_empty(
                rects.negate(),
                vars,
            ),
        };
    }

    /// Stats snapshot.
    pub fn stats(&self) -> PlannerStats {
        match &self.state {
            BuilderState::NoConstraint
            | BuilderState::Empty => {
                PlannerStats::default()
            }
            BuilderState::Active {
                rects,
                vars,
            } => {
                let mut dim_set: BTreeSet<IStr> =
                    BTreeSet::new();
                for rect in rects.rects.iter() {
                    for (i, dim_ranges) in rect
                        .per_dim
                        .iter()
                        .enumerate()
                    {
                        if !is_full_dim(
                            dim_ranges,
                            self.shape[i],
                        ) {
                            dim_set.insert(
                                self.dims[i],
                            );
                        }
                    }
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

    /// Materialize the accumulated state into a [`GridJoinTree`].
    pub fn finalize(
        self,
    ) -> Result<Option<GridJoinTree>, BackendError>
    {
        let GridJoinTreeBuilder {
            meta,
            backend: _,
            dims: _,
            shape: _,
            state,
        } = self;

        let (rects, vars) = match state {
            BuilderState::Empty => {
                return Ok(None);
            }
            BuilderState::NoConstraint => {
                // Treat an empty expression as "all variables, no
                // constraints" so the reader emits one chunk per array.
                let dims: SmallVec<[IStr; 4]> =
                    meta.dim_analysis
                        .all_dims
                        .iter()
                        .copied()
                        .collect();
                let shape: SmallVec<[u64; 4]> =
                    dims.iter()
                        .map(|d| {
                            lookup_dim_len(
                                meta, d,
                            )
                            .unwrap_or(0)
                        })
                        .collect();
                (
                    RectangleSet::full(
                        dims, shape,
                    ),
                    VarSet::All,
                )
            }
            BuilderState::Active {
                rects,
                vars,
            } => (rects, vars),
        };

        let groups =
            build_groups(meta, &rects, &vars)?;
        if groups.is_empty() {
            return Ok(None);
        }

        let Some(tree) =
            GridJoinTree::build(groups)
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
    /// the global rectangle set.
    pub fn add_constraint(
        &mut self,
        dim: IStr,
        vr: &ValueRangePresent,
        expansion: Expansion,
    ) -> Result<(), BackendError> {
        let dim_len =
            lookup_dim_len(self.meta, &dim)
                .ok_or_else(|| {
                    unknown_dim(dim)
                })?;
        let ranges = resolve_value_range_sync(
            self.backend,
            &dim,
            self.meta,
            dim_len,
            vr,
            expansion,
        )
        .map_err(|e| resolve_error(&dim, e))?;
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
        let dim_len =
            lookup_dim_len(self.meta, &dim)
                .ok_or_else(|| {
                    unknown_dim(dim)
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
        .map_err(|e| resolve_error(&dim, e))?;
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
        let one =
            RectangleSet::from_dim_constraint(
                self.dims.clone(),
                self.shape.clone(),
                dim,
                ranges,
            );
        match &mut self.state {
            BuilderState::Empty => {}
            BuilderState::NoConstraint => {
                self.state =
                    BuilderState::Active {
                        rects: one,
                        vars: VarSet::Specific(
                            BTreeSet::new(),
                        ),
                    };
            }
            BuilderState::Active {
                rects,
                ..
            } => {
                let r = rects.intersect(&one);
                if r.is_empty() {
                    self.state =
                        BuilderState::Empty;
                } else {
                    *rects = r;
                }
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn lookup_dim_len(
    meta: &ZarrMeta,
    dim: &IStr,
) -> Option<u64> {
    meta.dim_analysis
        .dim_lengths
        .get(dim)
        .copied()
        .or_else(|| {
            meta.array_by_path(*dim).and_then(
                |a| a.shape.first().copied(),
            )
        })
}

fn unknown_dim(dim: IStr) -> BackendError {
    BackendError::other(format!(
        "unknown dim '{}' (no dim length in meta and no \
         coordinate array)",
        AsRef::<str>::as_ref(&dim),
    ))
}

fn resolve_error<E: std::fmt::Display>(
    dim: &IStr,
    err: E,
) -> BackendError {
    BackendError::other(format!(
        "value-range resolution failed for dim '{}': {err}",
        AsRef::<str>::as_ref(dim),
    ))
}

fn is_full_dim(
    ranges: &[Range<u64>],
    dim_len: u64,
) -> bool {
    ranges.len() == 1
        && ranges[0].start == 0
        && ranges[0].end == dim_len
}

fn active_or_empty(
    rects: RectangleSet,
    vars: VarSet,
) -> BuilderState {
    if rects.is_empty() {
        BuilderState::Empty
    } else {
        BuilderState::Active { rects, vars }
    }
}

// ============================================================================
// Per-signature group construction
// ============================================================================

/// Materialize one [`OwnedGridGroup`] per chunk-grid signature.
///
/// Variables are grouped by `(dims, outer_chunk_shape, inner_chunk_shape)`;
/// for each group the global [`RectangleSet`] is projected onto the
/// signature's dim subset and converted into per-chunk [`ArraySubset`]s.
/// 1D dim-coord groups whose dim is already covered by a multi-dim group
/// in this plan are dropped (the multi-dim group's reader materializes the
/// dim column from its coord array, so the standalone group adds duplicate
/// reads and would force an extra `Independent` concat in the join tree).
fn build_groups(
    meta: &ZarrMeta,
    rects: &RectangleSet,
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

    let by_sig =
        group_vars_by_signature(meta, &var_list)?;

    let mut groups: Vec<OwnedGridGroup> =
        Vec::new();
    for (sig, sig_vars) in by_sig {
        if let Some(group) = build_one_group(
            meta, rects, sig, sig_vars,
        )? {
            groups.push(group);
        }
    }

    Ok(drop_redundant_dim_coord_groups(groups))
}

/// Group variables by their full `(dims, outer chunk shape, inner chunk
/// shape)` signature so each group can share one chunk-index plan.
fn group_vars_by_signature(
    meta: &ZarrMeta,
    vars: &[IStr],
) -> Result<
    BTreeMap<Arc<ChunkGridSignature>, Vec<IStr>>,
    BackendError,
> {
    let mut sig_cache: BTreeMap<
        ChunkGridSignature,
        Arc<ChunkGridSignature>,
    > = BTreeMap::new();
    let mut by_sig: BTreeMap<
        Arc<ChunkGridSignature>,
        Vec<IStr>,
    > = BTreeMap::new();
    for var in vars {
        let Some(arr_meta) =
            meta.array_by_path(*var)
        else {
            continue;
        };
        let zeros: Vec<u64> =
            vec![0u64; arr_meta.shape.len()];
        let outer_chunk_shape =
            chunk_shape_at_zero(
                &arr_meta.outer_chunk_grid,
                &zeros,
                "outer",
                var,
            )?;
        let inner_chunk_shape = match arr_meta
            .inner_chunk_grid
            .as_ref()
        {
            Some(g) => chunk_shape_at_zero(
                g, &zeros, "inner", var,
            )?,
            None => None,
        };
        let sig = ChunkGridSignature::new(
            arr_meta.dims.clone(),
            outer_chunk_shape,
            inner_chunk_shape,
        )?;
        let sig_arc = Arc::clone(
            sig_cache
                .entry(sig.clone())
                .or_insert_with(|| Arc::new(sig)),
        );
        by_sig
            .entry(sig_arc)
            .or_default()
            .push(*var);
    }
    Ok(by_sig)
}

fn chunk_shape_at_zero(
    grid: &ChunkGrid,
    zeros: &[u64],
    label: &str,
    var: &IStr,
) -> Result<
    Option<SmallVec<[u64; 4]>>,
    BackendError,
> {
    Ok(grid
        .chunk_shape(zeros)
        .map_err(|e| {
            BackendError::other(format!(
                "{label} chunk shape for '{}': {e:?}",
                AsRef::<str>::as_ref(var),
            ))
        })?
        .map(|v| {
            v.into_iter().map(|n| n.get()).collect()
        }))
}

fn build_one_group(
    meta: &ZarrMeta,
    rects: &RectangleSet,
    sig: Arc<ChunkGridSignature>,
    sig_vars: Vec<IStr>,
) -> Result<Option<OwnedGridGroup>, BackendError>
{
    let projected = rects.project(sig.dims());
    let subsets: Vec<ArraySubset> =
        projected.iter_subsets().collect();
    if subsets.is_empty() {
        return Ok(None);
    }

    let representative = sig_vars[0];
    let arr_meta = meta
        .array_by_path(representative)
        .ok_or_else(|| {
            BackendError::other(format!(
                "no array meta for '{}'",
                AsRef::<str>::as_ref(
                    &representative
                ),
            ))
        })?;
    let chunk_grid: Arc<ChunkGrid> =
        match &arr_meta.inner_chunk_grid {
            Some(g) => Arc::clone(g),
            None => Arc::clone(
                &arr_meta.outer_chunk_grid,
            ),
        };
    let array_shape: std::sync::Arc<[u64]> =
        chunk_grid.array_shape().to_vec().into();
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
                    "chunks_in_array_subset failed: \
                     {e:?}",
                ))
            })?
        else {
            continue;
        };
        for idx in indices.indices() {
            seen.insert(idx.to_vec());
        }
    }
    let chunk_indices: Vec<
        std::sync::Arc<[u64]>,
    > = seen
        .into_iter()
        .map(Into::into)
        .collect();
    let chunk_subsets: Vec<
        Option<std::sync::Arc<ChunkSubset>>,
    > = chunk_indices
        .iter()
        .map(|idx| {
            compute_chunk_subset(
                idx.as_ref(),
                &chunk_shape,
                array_shape.as_ref(),
                &subsets,
            )
            .map(std::sync::Arc::new)
        })
        .collect();

    Ok(Some(OwnedGridGroup::new(
        sig,
        sig_vars.into(),
        chunk_indices,
        chunk_subsets,
        array_shape,
    )))
}

/// Drop 1D dim-coord-only groups whose dim is also covered by a multi-dim
/// group in this plan; the multi-dim group's reader will materialize the
/// dim column from the coord array.
fn drop_redundant_dim_coord_groups(
    groups: Vec<OwnedGridGroup>,
) -> Vec<OwnedGridGroup> {
    let multi_dim_dims: BTreeSet<IStr> = groups
        .iter()
        .filter(|g| g.sig.dims().len() > 1)
        .flat_map(|g| g.sig.dims().to_vec())
        .collect();
    groups
        .into_iter()
        .filter(|g| {
            let dims = g.sig.dims();
            let is_dim_coord_only = dims.len()
                == 1
                && g.vars.len() == 1
                && g.vars[0] == dims[0];
            !(is_dim_coord_only
                && multi_dim_dims
                    .contains(&dims[0]))
        })
        .collect()
}

/// Compute the chunk-local [`ChunkSubset`] for a chunk index. Returns
/// `None` when the chunk is fully covered (no per-chunk slicing needed).
fn compute_chunk_subset(
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

    // Bounding box of the union of subset∩chunk intervals, per dim.
    // Initialize with inverse bounds so first intersection updates them.
    let mut bbox_start: Vec<u64> =
        std::iter::repeat(u64::MAX)
            .take(ndim)
            .collect();
    let mut bbox_end: Vec<u64> =
        std::iter::repeat(0u64)
            .take(ndim)
            .collect();

    for subset in subsets {
        let ranges = subset.to_ranges();
        for d in 0..ndim {
            let inter_start = ranges[d]
                .start
                .max(chunk_start[d]);
            let inter_end =
                ranges[d].end.min(chunk_end[d]);
            if inter_start < inter_end {
                bbox_start[d] = bbox_start[d]
                    .min(inter_start);
                bbox_end[d] =
                    bbox_end[d].max(inter_end);
            }
        }
    }

    let local_ranges: Vec<Range<u64>> =
        bbox_start
            .iter()
            .zip(bbox_end.iter())
            .zip(chunk_start.iter())
            .map(|((s, e), cs)| {
                (s - cs)..(e - cs)
            })
            .collect();
    let actual_chunk_shape: Vec<u64> = chunk_end
        .iter()
        .zip(chunk_start.iter())
        .map(|(e, s)| e - s)
        .collect();

    let subset =
        ChunkSubset::from_ranges(local_ranges);
    if is_full_chunk(&subset, &actual_chunk_shape)
    {
        None
    } else {
        Some(subset)
    }
}

fn is_full_chunk(
    subset: &ChunkSubset,
    chunk_shape: &[u64],
) -> bool {
    subset
        .ranges
        .iter()
        .zip(chunk_shape)
        .all(|(r, &s)| r.start == 0 && r.end >= s)
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
    let plan = compile_expr_to_plan(expr, meta)?;
    let mut builder =
        GridJoinTreeBuilder::new(meta, backend);
    apply_plan_sync(&plan, &mut builder)?;
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
    let plan = compile_expr_to_plan(expr, meta)?;
    let mut builder =
        GridJoinTreeBuilder::new(meta, backend);
    apply_plan_async(&plan, &mut builder).await?;
    let stats = builder.stats();
    let tree = builder.finalize()?;
    Ok((tree, stats))
}

fn compile_expr_to_plan(
    expr: &Expr,
    meta: &ZarrMeta,
) -> Result<ExprPlan, BackendError> {
    use crate::chunk_plan::LazyCompileCtx;
    use crate::chunk_plan::compile_expr;
    use crate::chunk_plan::compute_dims_and_lengths_unified;

    let (dims, _) =
        compute_dims_and_lengths_unified(meta);
    let mut ctx =
        LazyCompileCtx::new(meta, &dims);
    compile_expr(expr, &mut ctx)
}

// ============================================================================
// ExprPlan / LazyArraySelection -> builder walker
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
            let mut sub =
                compile_lazy_selection_sync(
                    constraints,
                    builder,
                )?;
            apply_vars_to_builder(&mut sub, vars);
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
            let mut sub =
                compile_lazy_selection_async(
                    constraints,
                    builder,
                )
                .await?;
            apply_vars_to_builder(&mut sub, vars);
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
                // Constraints with no var-set means no rows can be
                // produced because there's nothing to read.
                builder.set_empty();
            } else {
                for v in items {
                    builder.add_var(*v);
                }
            }
        }
    }
}

fn child_builder<'a, B>(
    parent: &GridJoinTreeBuilder<'a, B>,
) -> GridJoinTreeBuilder<'a, B> {
    GridJoinTreeBuilder {
        meta: parent.meta,
        backend: parent.backend,
        dims: parent.dims.clone(),
        shape: parent.shape.clone(),
        state: BuilderState::NoConstraint,
    }
}

fn compile_lazy_selection_sync<
    'a,
    B: ChunkedDataBackendSync,
>(
    sel: &LazyArraySelection,
    parent: &GridJoinTreeBuilder<'a, B>,
) -> Result<
    GridJoinTreeBuilder<'a, B>,
    BackendError,
> {
    match sel {
        LazyArraySelection::Rectangles(rects) => {
            let mut acc = child_builder(parent);
            acc.set_empty();
            for rect in rects {
                let rect_b =
                    compile_rectangle_sync(
                        rect, parent,
                    )?;
                acc.union(rect_b);
            }
            Ok(acc)
        }
        LazyArraySelection::Difference(a, b) => {
            let mut a_b =
                compile_lazy_selection_sync(
                    a, parent,
                )?;
            let b_b =
                compile_lazy_selection_sync(
                    b, parent,
                )?;
            a_b.difference(b_b);
            Ok(a_b)
        }
        LazyArraySelection::Union(a, b) => {
            let mut a_b =
                compile_lazy_selection_sync(
                    a, parent,
                )?;
            let b_b =
                compile_lazy_selection_sync(
                    b, parent,
                )?;
            a_b.union(b_b);
            Ok(a_b)
        }
        LazyArraySelection::BooleanNot(inner) => {
            let mut b =
                compile_lazy_selection_sync(
                    inner, parent,
                )?;
            b.negate();
            Ok(b)
        }
    }
}

fn compile_lazy_selection_async<
    'sel,
    'a,
    B: ChunkedDataBackendAsync,
>(
    sel: &'sel LazyArraySelection,
    parent: &'sel GridJoinTreeBuilder<'a, B>,
) -> std::pin::Pin<
    Box<
        dyn std::future::Future<
                Output = Result<
                    GridJoinTreeBuilder<'a, B>,
                    BackendError,
                >,
            > + Send
            + 'sel,
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
                    child_builder(parent);
                acc.set_empty();
                for rect in rects {
                    let rect_b =
                        compile_rectangle_async(
                            rect, parent,
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
                    compile_lazy_selection_async(
                        a, parent,
                    )
                    .await?;
                let b_b =
                    compile_lazy_selection_async(
                        b, parent,
                    )
                    .await?;
                a_b.difference(b_b);
                Ok(a_b)
            }
            LazyArraySelection::Union(a, b) => {
                let mut a_b =
                    compile_lazy_selection_async(
                        a, parent,
                    )
                    .await?;
                let b_b =
                    compile_lazy_selection_async(
                        b, parent,
                    )
                    .await?;
                a_b.union(b_b);
                Ok(a_b)
            }
            LazyArraySelection::BooleanNot(
                inner,
            ) => {
                let mut b =
                    compile_lazy_selection_async(
                        inner, parent,
                    )
                    .await?;
                b.negate();
                Ok(b)
            }
        }
    })
}

/// Classification of a single [`LazyDimConstraint`].
enum DimAction<'a> {
    /// Unconstrained along this dim — nothing to do.
    Skip,
    /// Proven empty along this dim — short-circuit to the empty rectangle.
    Empty,
    /// Resolve a value-range with the given binary-search expansion.
    Resolve(&'a ValueRangePresent, Expansion),
}

fn classify(
    c: &LazyDimConstraint,
) -> DimAction<'_> {
    match c {
        LazyDimConstraint::All => DimAction::Skip,
        LazyDimConstraint::Empty => DimAction::Empty,
        LazyDimConstraint::Unresolved(vr) => {
            DimAction::Resolve(vr, Expansion::Exact)
        }
        LazyDimConstraint::InterpolationRange(vr) => {
            DimAction::Resolve(
                vr,
                Expansion::InterpolationNeighbor,
            )
        }
        LazyDimConstraint::WrappingInterpolationRange(
            vr,
        ) => DimAction::Resolve(
            vr,
            Expansion::WrappingGhost,
        ),
    }
}

/// Common preamble for both sync/async rectangle compilation: handle the
/// `is_empty` / `is_all` shortcut cases. Returns `Some(builder)` if the
/// caller should return immediately, or `None` to proceed with per-dim
/// resolution into the returned builder.
fn rectangle_preamble<'a, B>(
    rect: &LazyHyperRectangle,
    parent: &GridJoinTreeBuilder<'a, B>,
) -> (GridJoinTreeBuilder<'a, B>, bool) {
    let mut b = child_builder(parent);
    if rect.is_empty() {
        b.set_empty();
        return (b, true);
    }
    if rect.is_all() {
        // Promote NoConstraint -> Active with the full set so subsequent
        // intersections / unions don't get short-circuited as identity.
        b.state = BuilderState::Active {
            rects: b.full_set(),
            vars: VarSet::Specific(
                BTreeSet::new(),
            ),
        };
        return (b, true);
    }
    (b, false)
}

fn compile_rectangle_sync<
    'a,
    B: ChunkedDataBackendSync,
>(
    rect: &LazyHyperRectangle,
    parent: &GridJoinTreeBuilder<'a, B>,
) -> Result<
    GridJoinTreeBuilder<'a, B>,
    BackendError,
> {
    let (mut b, done) =
        rectangle_preamble(rect, parent);
    if done {
        return Ok(b);
    }
    for (dim, constraint) in rect.dims() {
        match classify(constraint) {
            DimAction::Skip => {}
            DimAction::Empty => {
                b.set_empty();
                return Ok(b);
            }
            DimAction::Resolve(vr, exp) => {
                b.add_constraint(*dim, vr, exp)?;
            }
        }
    }
    Ok(b)
}

async fn compile_rectangle_async<
    'a,
    B: ChunkedDataBackendAsync,
>(
    rect: &LazyHyperRectangle,
    parent: &GridJoinTreeBuilder<'a, B>,
) -> Result<
    GridJoinTreeBuilder<'a, B>,
    BackendError,
> {
    let (mut b, done) =
        rectangle_preamble(rect, parent);
    if done {
        return Ok(b);
    }
    for (dim, constraint) in rect.dims() {
        match classify(constraint) {
            DimAction::Skip => {}
            DimAction::Empty => {
                b.set_empty();
                return Ok(b);
            }
            DimAction::Resolve(vr, exp) => {
                b.add_constraint_async(
                    *dim, vr, exp,
                )
                .await?;
            }
        }
    }
    Ok(b)
}
