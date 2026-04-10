//! Single place for **predicate column refs** and **physical read superset** expansion.

use std::collections::BTreeSet;

use polars::prelude::Expr;

use crate::IStr;
use crate::chunk_plan::collect_column_refs;
use crate::meta::ZarrMeta;
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
            out.insert(col.clone());
            continue;
        }
        if let Some(am) =
            meta.array_by_path(col.clone())
        {
            for d in am.dims.iter() {
                out.insert(d.clone());
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
        let Some(am) =
            meta.array_by_path(p.clone())
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
    let mut cols = match with_columns {
        None => return None,
        Some(c) => c,
    };
    for r in predicate_refs {
        cols.insert(r.clone());
    }
    // Always include every dataset dimension. Polars may omit index-only dims
    // like `point` when the projection is only filter columns + `.select(...)`,
    // but we need them in each chunk row to join/enrich aux vars (`station_id`
    // on `point`, etc.).
    for d in &meta.dim_analysis.all_dims {
        cols.insert(d.clone());
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

/// Policy built from Polars pushdown + predicate: expanded physical column set.
pub(crate) struct ResolvedColumnPolicy {
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
            physical_superset,
        }
    }

    pub(crate) fn physical_superset(
        &self,
    ) -> Option<&BTreeSet<IStr>> {
        self.physical_superset.as_ref()
    }
}
