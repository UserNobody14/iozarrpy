//! `GridJoinTreeBuilder` — entry point for compiling a Polars [`Expr`] into
//! a [`GridJoinTree`] with resolved per-dimension index ranges.
//!
//! Pipeline:
//! 1. [`crate::chunk_plan::compile_expr`] walks the [`Expr`] and emits an
//!    [`ExprPlan`] whose constraints are already resolved to a concrete
//!    [`RectangleSet`] (resolution happens inline via the
//!    [`crate::chunk_plan::exprs::compile_ctx::CoordResolver`] threaded
//!    through compilation).
//! 2. [`GridJoinTreeBuilder::apply_plan`] folds the resolved plan into the
//!    builder state.
//! 3. [`GridJoinTreeBuilder::finalize`] groups variables by
//!    [`ChunkGridSignature`], projects the global rectangle set onto each
//!    signature's dim subset, wraps [`GridJoinTree::Group`] nodes around
//!    each top-level child of [`ZarrMeta::root`], and returns the
//!    [`GridJoinTree`] consumed by the reader.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::sync::Arc;

use polars::prelude::Expr;
use smallvec::SmallVec;
use zarrs::array::{ArraySubset, ChunkGrid};

use crate::chunk_plan::coord_resolve::{
    CachedResolver, CollectingResolver,
    MemoizingSyncResolver, ResolveKey,
    resolve_value_range_async,
};
use crate::chunk_plan::exprs::compile_ctx::{
    LazyCompileCtx, Universe, lookup_dim_len,
    unknown_dim,
};
use crate::chunk_plan::exprs::expr_plan::{
    ExprPlan, VarSet as ExprVarSet,
};
use crate::chunk_plan::indexing::grid_join_tree::{
    ChunkSubset, GridJoinTree,
    LeafGroup as OwnedGridGroup,
};
use crate::chunk_plan::indexing::index_set::RectangleSet;
use crate::chunk_plan::indexing::types::ChunkGridSignature;
use crate::errors::BackendError;
use crate::meta::ZarrMeta;
use crate::shared::{
    ChunkedDataBackendAsync, ChunkedDataBackendSync,
    IStr,
};

// ============================================================================
// Public types
// ============================================================================

/// Statistics about a single planning run.
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct PlannerStats {
    pub dims_resolved: usize,
    pub vars_referenced: usize,
}

/// Variable selection accumulator.
#[derive(Debug, Clone)]
pub enum VarSet {
    All,
    Specific(BTreeSet<IStr>),
}

// ============================================================================
// Builder state
// ============================================================================

#[derive(Debug, Clone)]
enum BuilderState {
    NoConstraint,
    Empty,
    Active { rects: Box<RectangleSet>, vars: VarSet },
}

// ============================================================================
// GridJoinTreeBuilder
// ============================================================================

/// Accumulator for a [`GridJoinTree`] under construction. The plan is
/// already resolved by the time it reaches the builder, so the builder no
/// longer touches the backend.
pub struct GridJoinTreeBuilder<'a> {
    meta: &'a ZarrMeta,
    universe: &'a Universe,
    state: BuilderState,
}

impl<'a> GridJoinTreeBuilder<'a> {
    pub fn new(
        meta: &'a ZarrMeta,
        universe: &'a Universe,
    ) -> Self {
        Self {
            meta,
            universe,
            state: BuilderState::NoConstraint,
        }
    }

    fn full_set(&self) -> RectangleSet {
        RectangleSet::full(
            self.universe.dims.clone(),
            self.universe.shape.clone(),
        )
    }

    fn add_var(&mut self, name: IStr) {
        match &mut self.state {
            BuilderState::Empty => {}
            BuilderState::NoConstraint => {
                let mut s = BTreeSet::new();
                s.insert(name);
                self.state =
                    BuilderState::Active {
                        rects: Box::new(self.full_set()),
                        vars: VarSet::Specific(s),
                    };
            }
            BuilderState::Active {
                vars, ..
            } => {
                if let VarSet::Specific(s) = vars {
                    s.insert(name);
                }
            }
        }
    }

    fn add_all_vars(&mut self) {
        match &mut self.state {
            BuilderState::Empty => {}
            BuilderState::NoConstraint => {
                self.state =
                    BuilderState::Active {
                        rects: Box::new(self.full_set()),
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

    fn set_empty(&mut self) {
        self.state = BuilderState::Empty;
    }

    /// Intersect a resolved [`RectangleSet`] into the builder's state.
    fn intersect_rects(
        &mut self,
        other: &RectangleSet,
    ) {
        if other.is_empty() {
            self.state = BuilderState::Empty;
            return;
        }
        let me = std::mem::replace(
            &mut self.state,
            BuilderState::Empty,
        );
        self.state = match me {
            BuilderState::Empty => BuilderState::Empty,
            BuilderState::NoConstraint => {
                BuilderState::Active {
                    rects: Box::new(other.clone()),
                    vars: VarSet::Specific(
                        BTreeSet::new(),
                    ),
                }
            }
            BuilderState::Active {
                rects,
                vars,
            } => {
                let r = rects.intersect(other);
                if r.is_empty() {
                    BuilderState::Empty
                } else {
                    BuilderState::Active {
                        rects: Box::new(r),
                        vars,
                    }
                }
            }
        };
    }

    /// Apply a fully-resolved [`ExprPlan`] to the builder.
    pub fn apply_plan(&mut self, plan: &ExprPlan) {
        match plan {
            ExprPlan::NoConstraint => {
                self.add_all_vars();
            }
            ExprPlan::Empty => {
                self.set_empty();
            }
            ExprPlan::Active(p) => {
                apply_vars(self, &p.vars);
                if !p.rects.dims.is_empty() {
                    self.intersect_rects(&p.rects);
                }
            }
        }
    }

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
                            self.universe.shape[i],
                        ) {
                            dim_set.insert(
                                self.universe.dims[i],
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

    pub fn finalize(
        self,
    ) -> Result<Option<GridJoinTree>, BackendError>
    {
        let GridJoinTreeBuilder {
            meta,
            universe,
            state,
        } = self;

        let (rects, vars) = match state {
            BuilderState::Empty => return Ok(None),
            BuilderState::NoConstraint => (
                Box::new(RectangleSet::full(
                    universe.dims.clone(),
                    universe.shape.clone(),
                )),
                VarSet::All,
            ),
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

fn apply_vars(
    builder: &mut GridJoinTreeBuilder<'_>,
    vars: &ExprVarSet,
) {
    match vars {
        ExprVarSet::All => builder.add_all_vars(),
        ExprVarSet::Specific(items) => {
            if items.is_empty() {
                builder.set_empty();
            } else {
                for v in items {
                    builder.add_var(*v);
                }
            }
        }
    }
}

fn is_full_dim(
    ranges: &[Range<u64>],
    dim_len: u64,
) -> bool {
    ranges.len() == 1
        && ranges[0].start == 0
        && ranges[0].end == dim_len
}

// ============================================================================
// Per-signature group construction
// ============================================================================

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
    let chunk_grid: Arc<ChunkGrid> = arr_meta
        .inner_chunk_grid
        .as_ref()
        .map_or_else(
            || Arc::clone(&arr_meta.outer_chunk_grid),
            Arc::clone,
        );
    let array_shape: std::sync::Arc<[u64]> =
        chunk_grid.array_shape().to_vec().into();
    let chunk_shape: Vec<u64> = arr_meta.chunk_shape.to_vec();

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
    let chunk_indices: Vec<std::sync::Arc<[u64]>> =
        seen.into_iter().map(Into::into).collect();
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
    let mut bbox_start: Vec<u64> =
        std::iter::repeat_n(u64::MAX, ndim).collect();
    let mut bbox_end: Vec<u64> =
        std::iter::repeat_n(0u64, ndim).collect();
    for subset in subsets {
        let ranges = subset.to_ranges();
        for d in 0..ndim {
            let inter_start =
                ranges[d].start.max(chunk_start[d]);
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
///
/// Single-pass: a memoizing resolver hits the backend inline on cache
/// miss. For typical 1-3 dim queries this is faster than the dry-run +
/// parallel-resolve dance because we avoid double-walking the expression
/// and the rayon setup cost. Coord chunk loads inside each resolve still
/// parallelize via `MaybeParIter` for multi-chunk arrays.
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
    let universe = Universe::from_meta(meta);
    let resolver =
        MemoizingSyncResolver::new(backend);
    let plan = compile_expr_to_plan(
        expr,
        meta,
        &universe,
        &resolver,
    )?;
    let mut builder =
        GridJoinTreeBuilder::new(meta, &universe);
    builder.apply_plan(&plan);
    let stats = builder.stats();
    let tree = builder.finalize()?;
    Ok((tree, stats))
}

/// Run a pure dry-run compile to collect every `(dim, vr, expansion)`
/// triple the expression would resolve, deduplicated.
fn collect_resolution_keys(
    expr: &Expr,
    meta: &ZarrMeta,
    universe: &Universe,
) -> Vec<ResolveKey> {
    let collector = CollectingResolver::new();
    let _ = compile_expr_to_plan(
        expr,
        meta,
        universe,
        &collector,
    );
    let unique: HashMap<ResolveKey, ()> = collector
        .into_keys()
        .into_iter()
        .map(|k| (k, ()))
        .collect();
    unique.into_keys().collect()
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
    let universe = Universe::from_meta(meta);
    let keys = collect_resolution_keys(
        expr, meta, &universe,
    );

    // For ≤ 1 key, sequential .await avoids try_join_all's BoxFuture and
    // task-scheduler overhead. For larger N, parallel concurrency wins.
    let resolved: Vec<(
        ResolveKey,
        Vec<Range<u64>>,
    )> = if keys.len() <= 1 {
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            let dim_len = lookup_dim_len(
                meta, &key.dim,
            )
            .ok_or_else(|| unknown_dim(key.dim))?;
            let r = resolve_value_range_async(
                backend,
                &key.dim,
                meta,
                dim_len,
                &key.vr,
                key.expansion,
            )
            .await?;
            out.push((key, r));
        }
        out
    } else {
        futures::future::try_join_all(
            keys.into_iter().map(
                |key| async move {
                    let dim_len = lookup_dim_len(
                        meta, &key.dim,
                    )
                    .ok_or_else(|| {
                        unknown_dim(key.dim)
                    })?;
                    let r =
                        resolve_value_range_async(
                            backend,
                            &key.dim,
                            meta,
                            dim_len,
                            &key.vr,
                            key.expansion,
                        )
                        .await?;
                    Ok::<_, BackendError>((key, r))
                },
            ),
        )
        .await?
    };
    let table: HashMap<
        ResolveKey,
        Vec<Range<u64>>,
    > = resolved.into_iter().collect();
    let resolver = CachedResolver::from_table(table);

    let plan = compile_expr_to_plan(
        expr,
        meta,
        &universe,
        &resolver,
    )?;
    let mut builder =
        GridJoinTreeBuilder::new(meta, &universe);
    builder.apply_plan(&plan);
    let stats = builder.stats();
    let tree = builder.finalize()?;
    Ok((tree, stats))
}

fn compile_expr_to_plan(
    expr: &Expr,
    meta: &ZarrMeta,
    universe: &Universe,
    resolver: &dyn crate::chunk_plan::exprs::compile_ctx::CoordResolver,
) -> Result<ExprPlan, BackendError> {
    use crate::chunk_plan::compile_expr;

    let mut ctx = LazyCompileCtx::new(
        meta, universe, resolver,
    );
    compile_expr(expr, &mut ctx)
}
