use std::collections::BTreeMap;
use std::sync::Arc;

use smallvec::SmallVec;
use snafu::ResultExt;
use zarrs::array::{ArraySubset, ChunkGrid};

use super::DatasetSelection;
use super::plan::GroupedChunkPlan;
use super::types::ChunkGridSignature;
use crate::chunk_plan::indexing::selection::ArraySubsetList;
use crate::errors::BackendError;

use crate::meta::ZarrMeta;
use crate::shared::{IntoIStr, IStr, MaybeParIter};

/// Create an ArraySubsetList that covers the entire array shape.
fn all_chunks_subset(
    shape: &[u64],
) -> ArraySubsetList {
    let ranges: Vec<std::ops::Range<u64>> =
        shape.iter().map(|&s| 0..s).collect();
    let subset =
        ArraySubset::new_with_ranges(&ranges);
    let mut list = ArraySubsetList::new();
    list.push(subset);
    list
}

/// Per-variable output of [`selection_to_grouped_chunk_plan_unified_from_meta`]
/// before signature deduplication and insertion into [`GroupedChunkPlan`].
type VarPlanPart = (
    IStr,
    super::types::ChunkGridSignature,
    ArraySubsetList,
    Arc<ChunkGrid>,
);

fn var_plan_part_from_meta(
    var: &str,
    maybe_sel: Option<
        &super::selection::DataArraySelection,
    >,
    meta: &ZarrMeta,
) -> Result<VarPlanPart, BackendError> {
    let Some(var_meta) = meta.array_by_path(var)
    else {
        return Err(BackendError::UnknownZarrArray {
            name: var.istr(),
            available_zarr_arrays: meta
                .all_zarr_array_paths(),
        });
    };

    let all_zeroes_dimensionality =
        vec![0u64; var_meta.shape.len()];
    let outer_chunk_shape: Option<
        SmallVec<[u64; 4]>,
    > = {
        let raw_shape = var_meta
            .outer_chunk_grid
            .chunk_shape(all_zeroes_dimensionality.as_slice())
            .context(
                crate::errors::backend::IncompatibleDimensionalitySnafu {
                    dims: var_meta.dims.clone().to_vec(),
                    shape: var_meta.shape.clone().to_vec(),
                    paths: vec![var.istr()],
                },
            )?;
        raw_shape.map(|v| {
            v.into_iter().map(|n| n.get()).collect()
        })
    };

    let inner_chunk_shape: Option<
        SmallVec<[u64; 4]>,
    > = match var_meta.inner_chunk_grid.as_ref() {
        Some(grid) => {
            let raw_shape = grid
                .chunk_shape(&all_zeroes_dimensionality)
                .context(
                    crate::errors::backend::IncompatibleDimensionalitySnafu {
                        dims: var_meta.dims.clone().to_vec(),
                        shape: var_meta.shape.clone().to_vec(),
                        paths: vec![var.istr()],
                    },
                )?;
            raw_shape.map(|v| {
                v.into_iter().map(|n| n.get()).collect()
            })
        }
        None => None,
    };

    let sig = super::types::ChunkGridSignature::new(
        var_meta.dims.clone(),
        outer_chunk_shape,
        inner_chunk_shape,
    )?;

    let chunk_plan = if let Some(sel) = maybe_sel {
        sel.clone().into()
    } else {
        all_chunks_subset(&var_meta.shape)
    };

    let chunk_grid = match &var_meta.inner_chunk_grid {
        Some(grid) => grid.clone(),
        None => var_meta.outer_chunk_grid.clone(),
    };

    Ok((
        var.istr(),
        sig,
        chunk_plan,
        chunk_grid,
    ))
}

/// Convert a DatasetSelection to a GroupedChunkPlan using metadata only.
///
/// This version doesn't require a store - it creates ChunkGrids from the
/// metadata's shape and chunk_shape information. This enables backends
/// that don't expose a raw store to still compile chunk plans.
pub fn selection_to_grouped_chunk_plan_unified_from_meta(
    selection: &DatasetSelection,
    meta: &ZarrMeta,
) -> Result<GroupedChunkPlan, BackendError> {
    let mut grouped_plan =
        GroupedChunkPlan::new();

    let mut sig_cache: BTreeMap<
        ChunkGridSignature,
        Arc<ChunkGridSignature>,
    > = BTreeMap::new();

    let all_vars;
    let vars_to_process: Vec<(
        &str,
        Option<
            &super::selection::DataArraySelection,
        >,
    )> = match selection {
        DatasetSelection::NoSelectionMade => {
            all_vars = meta.all_data_var_paths();
            all_vars
                .iter()
                .map(|v| (v.as_ref(), None))
                .collect()
        }
        DatasetSelection::Empty => {
            return Ok(grouped_plan);
        }
        DatasetSelection::Selection(
            grouped_sel,
        ) => grouped_sel
            .vars()
            .map(|(v, sel)| (v, Some(sel)))
            .collect(),
    };

    // Below this many variables the rayon scheduling overhead exceeds the
    // gain from parallel signature/chunk-grid construction. Tiny queries
    // (single-variable lookups in particular) regress noticeably when forced
    // through `par_iter`, so fall back to a plain map.
    const PARALLEL_PLAN_VARS: usize = 4;

    let parts: Vec<VarPlanPart> = vars_to_process
        .maybe_par_iter(PARALLEL_PLAN_VARS)
        .map_collect(|(var, maybe_sel)| {
            var_plan_part_from_meta(
                var, *maybe_sel, meta,
            )
        })?;

    for (var, sig, chunk_plan, chunk_grid) in parts
    {
        let sig_arc = sig_cache
            .entry(sig.clone())
            .or_insert_with(|| Arc::new(sig))
            .clone();
        grouped_plan.insert(
            var,
            sig_arc,
            chunk_plan,
            chunk_grid,
        );
    }

    Ok(grouped_plan)
}
