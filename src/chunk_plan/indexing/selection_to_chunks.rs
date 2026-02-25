use std::collections::BTreeMap;
use std::sync::Arc;

use smallvec::SmallVec;
use zarrs::array::ArraySubset;

use super::DatasetSelection;
use super::plan::GroupedChunkPlan;
use super::types::ChunkGridSignature;
use crate::chunk_plan::indexing::selection::ArraySubsetList;
use crate::errors::BackendError;

use crate::IntoIStr;
use crate::meta::ZarrMeta;

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

/// Convert a DatasetSelection to a GroupedChunkPlan using metadata only.
///
/// This version doesn't require a store - it creates ChunkGrids from the
/// metadata's shape and chunk_shape information. This enables backends
/// that don't expose a raw store to still compile chunk plans.
pub(crate) fn selection_to_grouped_chunk_plan_unified_from_meta(
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

    for (var, maybe_sel) in vars_to_process {
        let Some(var_meta) =
            meta.array_by_path(var)
        else {
            return Err(
                BackendError::UnknownZarrArray {
                    name: var.istr(),
                    available_zarr_arrays: meta
                        .all_zarr_array_paths(),
                },
            );
        };

        // Get chunk shape from metadata
        let chunk_shape: SmallVec<[u64; 4]> =
            var_meta
                .chunk_shape
                .iter()
                .copied()
                .collect();

        let sig = ChunkGridSignature::new(
            var_meta.dims.clone(),
            chunk_shape.clone(),
        );
        let sig_arc = sig_cache
            .entry(sig.clone())
            .or_insert_with(|| Arc::new(sig))
            .clone();

        // Create chunk plan
        let chunk_plan =
            if let Some(sel) = maybe_sel {
                sel.clone().into()
            } else {
                all_chunks_subset(&var_meta.shape)
            };

        let chunk_grid =
            var_meta.chunk_grid.clone();
        grouped_plan.insert(
            var.istr(),
            sig_arc,
            chunk_plan,
            chunk_grid,
        );
    }

    Ok(grouped_plan)
}
