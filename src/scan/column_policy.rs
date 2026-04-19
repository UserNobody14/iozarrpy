//! Single place for **predicate column refs**, **physical read superset** expansion,
//! **shared eligibility** for 1D variables (chunk group vs post-merge enrichment),
//! and **per-chunk physical read plans** (deduped zarr reads + column materialization).

use std::collections::{BTreeMap, BTreeSet};

use polars::prelude::Expr;

use crate::chunk_plan::collect_column_refs;
use crate::errors::{
    BackendError, BackendResult,
};
use crate::meta::ZarrMeta;
use crate::scan::shared::{
    compute_var_chunk_indices,
    should_include_column,
};
use crate::shared::IStr;
use crate::shared::expand_projection_to_flat_paths;

/// All column names referenced by a Polars predicate expression.
pub(crate) fn predicate_column_refs(
    expr: &Expr,
) -> BTreeSet<IStr> {
    let mut refs: Vec<internment::Intern<str>> =
        Vec::new();
    collect_column_refs(expr, &mut refs);
    refs.into_iter().collect()
}

/// Dimensions touched by the current projection (dim columns + dims of arrays in the set).
fn projection_dims_used(
    expanded: &BTreeSet<IStr>,
    meta: &ZarrMeta,
) -> BTreeSet<IStr> {
    let mut out = BTreeSet::new();
    for col in expanded {
        if meta
            .dim_analysis
            .all_dims
            .iter()
            .any(|d| d == col)
        {
            out.insert(*col);
            continue;
        }
        if let Some(am) = meta.array_by_path(*col)
        {
            for d in am.dims.iter() {
                out.insert(*d);
            }
        }
    }
    out
}

/// Pull in 1D root data variables on any dimension already used by `expanded`.
///
/// Polars may omit auxiliary coords (e.g. lat/lon on `point`) from pushed
/// `with_columns` even when they appear in `.select()`. Once a grid variable
/// implies `point`, we still need those arrays for enrichment and output.
fn expand_1d_aux_on_projection_dims(
    expanded: &mut BTreeSet<IStr>,
    dims_used: &BTreeSet<IStr>,
    meta: &ZarrMeta,
) {
    // Use every on-disk array path, not only `data_vars`. CF auxiliary coords
    // (e.g. lat/lon referenced via ``coordinates``) are stored in `node.arrays`
    // but excluded from `data_vars`, so `all_data_var_paths` would miss them.
    for p in meta.all_zarr_array_paths() {
        if expanded.contains(&p) {
            continue;
        }
        let Some(am) = meta.array_by_path(p)
        else {
            continue;
        };
        if am.dims.len() != 1 {
            continue;
        }
        if dims_used.contains(&am.dims[0]) {
            expanded.insert(p);
        }
    }
}

/// Expand [`expand_projection_to_flat_paths`] after merging predicate column refs
/// and all dataset dimensions, then pull in implied 1D auxiliary arrays.
///
/// Polars `register_io_source` projection pushdown supplies `with_columns` for the
/// final output schema; it may omit columns that appear only in a `.filter(...)`.
/// Those must still be read so chunk DataFrames include coordinate/data columns
/// the predicate references — pass them via `predicate_refs` from
/// [`crate::scan::column_policy::predicate_column_refs`].
///
/// The returned set is the **read/enrichment superset**. Callers must separately
/// **project** each batch to the original Polars `with_columns` list (preserving order).
pub(crate) fn expand_io_source_physical(
    with_columns: Option<BTreeSet<IStr>>,
    predicate_refs: &BTreeSet<IStr>,
    meta: &ZarrMeta,
) -> Option<BTreeSet<IStr>> {
    let mut cols = with_columns?;
    for r in predicate_refs {
        cols.insert(*r);
    }
    // Always include every dataset dimension. Polars may omit index-only dims
    // like `point` when the projection is only filter columns + `.select(...)`,
    // but we need them in each chunk row to join/enrich aux vars (`station_id`
    // on `point`, etc.).
    for d in &meta.dim_analysis.all_dims {
        cols.insert(*d);
    }
    let mut expanded =
        expand_projection_to_flat_paths(
            &cols, meta,
        );
    let dims_used =
        projection_dims_used(&expanded, meta);
    expand_1d_aux_on_projection_dims(
        &mut expanded,
        &dims_used,
        meta,
    );
    Some(expanded)
}

/// Policy built from Polars pushdown + predicate: predicate refs and the expanded
/// physical column set used for streaming reads and enrichment.
pub(crate) struct ResolvedColumnPolicy {
    predicate_refs: BTreeSet<IStr>,
    physical_superset: Option<BTreeSet<IStr>>,
}

impl ResolvedColumnPolicy {
    /// `with_columns_for_physical` should be `Some(...)` built from Polars' list or
    /// [`ZarrMeta::tidy_column_order`] when Polars passes `None`.
    pub(crate) fn new(
        with_columns_for_physical: Option<
            BTreeSet<IStr>,
        >,
        predicate: &Expr,
        meta: &ZarrMeta,
    ) -> Self {
        let predicate_refs =
            predicate_column_refs(predicate);
        let physical_superset =
            expand_io_source_physical(
                with_columns_for_physical,
                &predicate_refs,
                meta,
            );
        Self {
            predicate_refs,
            physical_superset,
        }
    }

    pub(crate) fn predicate_refs(
        &self,
    ) -> &BTreeSet<IStr> {
        &self.predicate_refs
    }

    pub(crate) fn physical_superset(
        &self,
    ) -> Option<&BTreeSet<IStr>> {
        self.physical_superset.as_ref()
    }
}

/// True if this chunk group reads `name` directly, or `name` is 1D on a dimension
/// this group's signature materializes (same rule as enrichment along that dim).
pub(crate) fn group_supplies_array_or_1d_enrichable(
    name: &IStr,
    sig_dims: &BTreeSet<IStr>,
    vars: &BTreeSet<IStr>,
    meta: &ZarrMeta,
) -> bool {
    if vars.contains(name) {
        return true;
    }
    let Some(vm) = meta.array_by_path(name)
    else {
        return false;
    };
    vm.shape.len() == 1
        && sig_dims.contains(&vm.dims[0])
}

// =============================================================================
// Per-chunk physical read plan (unified coords + variables)
// =============================================================================

/// One physical zarr read for the current chunk iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReadSpec {
    /// Contiguous global range along a 1D array (coordinate fast path).
    Slice1d {
        coord_chunk_shape: u64,
        start: u64,
        len: u64,
    },
    /// Single chunk index tuple for an arbitrary-C layout array.
    Chunk { indices: Vec<u64> },
}

#[derive(Debug, Clone)]
pub(crate) enum DimMaterialization {
    /// Integer index: `local + origin` (see [`crate::scan::shared::build_coord_column`]).
    Synthetic,
    /// Values loaded from this on-disk path (1D coord aligned with the dimension).
    FromArray { path: IStr },
}

#[derive(Debug, Clone)]
pub(crate) struct DimStep {
    pub(crate) dim_name: IStr,
    pub(crate) dim_idx: usize,
    pub(crate) mat: DimMaterialization,
}

#[derive(Debug, Clone)]
pub(crate) struct VarStep {
    pub(crate) name: IStr,
    pub(crate) path: IStr,
    pub(crate) var_dims: Vec<IStr>,
    pub(crate) var_chunk_shape: Vec<u64>,
    pub(crate) offsets: Vec<u64>,
}

/// Deduped reads and column build recipe for one primary-grid chunk.
#[derive(Debug, Clone)]
pub(crate) struct ChunkPhysicalPlan {
    pub(crate) dims: Vec<DimStep>,
    pub(crate) vars: Vec<VarStep>,
    /// Sorted by zarr path for stable I/O ordering.
    pub(crate) reads: Vec<(IStr, ReadSpec)>,
}

fn add_var_step(
    meta: &ZarrMeta,
    primary_dims: &[IStr],
    primary_idx: &[u64],
    primary_chunk_shape: &[u64],
    with_columns: Option<&BTreeSet<IStr>>,
    name: IStr,
    require_known: bool,
    reads_acc: &mut BTreeMap<IStr, ReadSpec>,
    vars: &mut Vec<VarStep>,
    seen_names: &mut BTreeSet<IStr>,
) -> BackendResult<()> {
    if seen_names.contains(&name) {
        return Ok(());
    }
    if !should_include_column(&name, with_columns)
    {
        return Ok(());
    }
    if primary_dims.iter().any(|d| d == &name) {
        return Ok(());
    }
    let Some(var_meta) = meta.array_by_path(name)
    else {
        if require_known {
            return Err(
                BackendError::UnknownDataVar {
                    name,
                    available_vars: meta
                        .all_data_var_paths(),
                },
            );
        }
        return Ok(());
    };
    let var_dims: Vec<IStr> =
        var_meta.dims.iter().cloned().collect();
    let (chunk_indices, offsets) =
        compute_var_chunk_indices(
            primary_idx,
            primary_chunk_shape,
            primary_dims,
            &var_dims,
            &var_meta.chunk_shape,
            &var_meta.shape,
        );
    register_read(
        reads_acc,
        var_meta.path,
        ReadSpec::Chunk {
            indices: chunk_indices,
        },
    )?;
    seen_names.insert(name);
    vars.push(VarStep {
        name,
        path: var_meta.path,
        var_dims,
        var_chunk_shape: var_meta
            .chunk_shape
            .to_vec(),
        offsets,
    });
    Ok(())
}

fn register_read(
    reads: &mut BTreeMap<IStr, ReadSpec>,
    path: IStr,
    spec: ReadSpec,
) -> BackendResult<()> {
    use std::collections::btree_map::Entry;
    match reads.entry(path) {
        Entry::Vacant(e) => {
            e.insert(spec);
            Ok(())
        }
        Entry::Occupied(o) => {
            if o.get() == &spec {
                Ok(())
            } else {
                Err(BackendError::Other {
                    msg: format!(
                        "conflicting physical read specs for zarr path {}",
                        o.key()
                    ),
                })
            }
        }
    }
}

/// Build the physical read plan for one chunk of a primary [`crate::chunk_plan::ChunkGridSignature`].
///
/// `with_columns` is the expanded physical superset (predicate + projection); `None` means
/// read every column the group can supply.
pub(crate) fn build_chunk_physical_plan(
    meta: &ZarrMeta,
    primary_dims: &[IStr],
    primary_idx: &[u64],
    primary_chunk_shape: &[u64],
    origin: &[u64],
    group_vars: &[IStr],
    with_columns: Option<&BTreeSet<IStr>>,
) -> BackendResult<ChunkPhysicalPlan> {
    let sig_dims: BTreeSet<IStr> =
        primary_dims.iter().cloned().collect();
    let vars_in_group: BTreeSet<IStr> =
        group_vars.iter().cloned().collect();

    let mut reads_acc: BTreeMap<IStr, ReadSpec> =
        BTreeMap::new();
    let mut dims: Vec<DimStep> = Vec::new();

    for (dim_idx, dim_name) in
        primary_dims.iter().enumerate()
    {
        // Always materialize every primary-grid dimension column: row-major
        // layout and predicates reference these names even when Polars projection
        // omits them from IO `with_columns` (see streaming empty-result tests).
        let mat = match meta
            .array_by_path(*dim_name)
        {
            Some(am) if am.shape.len() == 1 => {
                let start = origin[dim_idx];
                let len =
                    primary_chunk_shape[dim_idx];
                register_read(
                    &mut reads_acc,
                    am.path,
                    ReadSpec::Slice1d {
                        coord_chunk_shape: am
                            .chunk_shape[0],
                        start,
                        len,
                    },
                )?;
                DimMaterialization::FromArray {
                    path: am.path,
                }
            }
            _ => DimMaterialization::Synthetic,
        };
        dims.push(DimStep {
            dim_name: *dim_name,
            dim_idx,
            mat,
        });
    }

    let mut vars: Vec<VarStep> = Vec::new();
    let mut seen_names: BTreeSet<IStr> =
        BTreeSet::new();

    for &name in group_vars {
        add_var_step(
            meta,
            primary_dims,
            primary_idx,
            primary_chunk_shape,
            with_columns,
            name,
            true,
            &mut reads_acc,
            &mut vars,
            &mut seen_names,
        )?;
    }

    if let Some(expanded) = with_columns {
        for name in expanded.iter() {
            if seen_names.contains(name) {
                continue;
            }
            if !group_supplies_array_or_1d_enrichable(
                name,
                &sig_dims,
                &vars_in_group,
                meta,
            ) {
                continue;
            }
            add_var_step(
                meta,
                primary_dims,
                primary_idx,
                primary_chunk_shape,
                with_columns,
                *name,
                false,
                &mut reads_acc,
                &mut vars,
                &mut seen_names,
            )?;
        }
    }

    let reads: Vec<(IStr, ReadSpec)> =
        reads_acc.into_iter().collect();

    Ok(ChunkPhysicalPlan { dims, vars, reads })
}
