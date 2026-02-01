use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use smallvec::SmallVec;
use zarrs::array::Array;
use zarrs::array::chunk_grid::{
    ChunkGrid, RegularChunkGrid,
};
use zarrs::array_subset::ArraySubset;

use super::DatasetSelection;
use super::plan::GroupedChunkPlan;
use super::types::ChunkGridSignature;
use crate::chunk_plan::CompileError;
use crate::chunk_plan::indexing::selection::ArraySubsetList;

use crate::IntoIStr;
use crate::meta::{ZarrDatasetMeta, ZarrMeta};

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

/// Convert a DatasetSelection to a GroupedChunkPlan.
///
/// This function:
/// 1. Groups variables by their chunk grid signature (dims + chunk_shape)
/// 2. Computes chunk indices for each variable based on its selection
/// 3. Returns a GroupedChunkPlan that can iterate chunks per grid
///
/// Variables with the same dimensions AND chunk shape share the same ChunkPlan.
pub(crate) fn selection_to_grouped_chunk_plan(
    selection: &DatasetSelection,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
) -> Result<GroupedChunkPlan, CompileError> {
    let mut grouped_plan =
        GroupedChunkPlan::new();

    // Cache for grid signatures to avoid duplicate Arc allocations
    let mut sig_cache: BTreeMap<
        ChunkGridSignature,
        Arc<ChunkGridSignature>,
    > = BTreeMap::new();

    // For NoSelectionMade, we need to include all data variables with "all chunks"
    let vars_to_process: Vec<(
        &str,
        Option<
            &super::selection::DataArraySelection,
        >,
    )> = match selection {
        DatasetSelection::NoSelectionMade => {
            // Include all data variables with None (meaning "all")
            meta.data_vars
                .iter()
                .map(|v| (v.as_ref(), None))
                .collect()
        }
        DatasetSelection::Empty => {
            // No variables, return empty plan
            return Ok(grouped_plan);
        }
        DatasetSelection::Selection(
            grouped_sel,
        ) => {
            // Include variables from the selection
            grouped_sel
                .vars()
                .map(|(v, sel)| (v, Some(sel)))
                .collect()
        }
    };

    for (var, maybe_sel) in vars_to_process {
        let var_key = var.istr();
        let Some(var_meta) =
            meta.arrays.get(&var_key)
        else {
            continue;
        };

        // Open the array to get chunk grid info
        let arr = Array::open(
            store.clone(),
            var_meta.path.as_ref(),
        )
        .map_err(|e| {
            CompileError::Unsupported(format!(
                "failed to open array '{var}': {e}"
            ))
        })?;

        let zero =
            vec![0u64; arr.dimensionality()];
        let chunk_shape_nz = arr
            .chunk_shape(&zero)
            .map_err(|e| {
                CompileError::Unsupported(
                    e.to_string(),
                )
            })?;
        let chunk_shape: SmallVec<[u64; 4]> =
            chunk_shape_nz
                .iter()
                .map(|nz| nz.get())
                .collect();

        // Create the chunk grid signature
        let sig = ChunkGridSignature::new(
            var_meta.dims.clone(),
            chunk_shape,
        );
        let sig_arc = sig_cache
            .entry(sig.clone())
            .or_insert_with(|| Arc::new(sig))
            .clone();

        // Compute chunk indices based on selection
        // When maybe_sel is None, we select ALL chunks (conservative fallback)
        let chunk_plan =
            if let Some(sel) = maybe_sel {
                sel.clone().into()
            } else {
                all_chunks_subset(arr.shape())
            };

        grouped_plan.insert(
            var_key,
            sig_arc,
            chunk_plan,
            Arc::new(arr.chunk_grid().clone()),
        );
    }

    Ok(grouped_plan)
}

/// Async version: Convert a DatasetSelection to a GroupedChunkPlan.
pub(crate) async fn selection_to_grouped_chunk_plan_async(
    selection: &DatasetSelection,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
) -> Result<GroupedChunkPlan, CompileError> {
    let mut grouped_plan =
        GroupedChunkPlan::new();

    // Cache for grid signatures
    let mut sig_cache: BTreeMap<
        ChunkGridSignature,
        Arc<ChunkGridSignature>,
    > = BTreeMap::new();

    // For NoSelectionMade, include all data variables
    let vars_to_process: Vec<(
        &str,
        Option<
            &super::selection::DataArraySelection,
        >,
    )> = match selection {
        DatasetSelection::NoSelectionMade => meta
            .data_vars
            .iter()
            .map(|v| (v.as_ref(), None))
            .collect(),
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
        let var_key = var.istr();
        let Some(var_meta) =
            meta.arrays.get(&var_key)
        else {
            continue;
        };

        // Open the array async
        let arr = zarrs::array::Array::async_open(
            store.clone(),
            var_meta.path.as_ref(),
        )
        .await
        .map_err(|e| {
            CompileError::Unsupported(format!(
                "failed to open array '{var}': {e}"
            ))
        })?;

        let zero =
            vec![0u64; arr.dimensionality()];
        let chunk_shape_nz = arr
            .chunk_shape(&zero)
            .map_err(|e| {
                CompileError::Unsupported(
                    e.to_string(),
                )
            })?;
        let chunk_shape: SmallVec<[u64; 4]> =
            chunk_shape_nz
                .iter()
                .map(|nz| nz.get())
                .collect();

        let sig = ChunkGridSignature::new(
            var_meta.dims.clone(),
            chunk_shape,
        );
        let sig_arc = sig_cache
            .entry(sig.clone())
            .or_insert_with(|| Arc::new(sig))
            .clone();

        // When maybe_sel is None, we select ALL chunks (conservative fallback)
        let chunk_plan =
            if let Some(sel) = maybe_sel {
                sel.clone().into()
            } else {
                all_chunks_subset(arr.shape())
            };

        grouped_plan.insert(
            var_key,
            sig_arc,
            chunk_plan,
            Arc::new(arr.chunk_grid().clone()),
        );
    }

    Ok(grouped_plan)
}

/// Convert a DatasetSelection to a GroupedChunkPlan (unified ZarrMeta).
pub(crate) fn selection_to_grouped_chunk_plan_unified(
    selection: &DatasetSelection,
    meta: &ZarrMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
) -> Result<GroupedChunkPlan, CompileError> {
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
        let var_key = var.istr();
        let Some(var_meta) =
            meta.array_by_path(var)
        else {
            continue;
        };

        let arr = Array::open(
            store.clone(),
            var_meta.path.as_ref(),
        )
        .map_err(|e| {
            CompileError::Unsupported(format!(
                "failed to open array '{var}': {e}"
            ))
        })?;

        let zero =
            vec![0u64; arr.dimensionality()];
        let chunk_shape_nz = arr
            .chunk_shape(&zero)
            .map_err(|e| {
                CompileError::Unsupported(
                    e.to_string(),
                )
            })?;
        let chunk_shape: SmallVec<[u64; 4]> =
            chunk_shape_nz
                .iter()
                .map(|nz| nz.get())
                .collect();

        let sig = ChunkGridSignature::new(
            var_meta.dims.clone(),
            chunk_shape,
        );
        let sig_arc = sig_cache
            .entry(sig.clone())
            .or_insert_with(|| Arc::new(sig))
            .clone();

        // When maybe_sel is None, we select ALL chunks (conservative fallback)
        let chunk_plan =
            if let Some(sel) = maybe_sel {
                sel.clone().into()
            } else {
                all_chunks_subset(arr.shape())
            };

        grouped_plan.insert(
            var_key,
            sig_arc,
            chunk_plan,
            Arc::new(arr.chunk_grid().clone()),
        );
    }

    Ok(grouped_plan)
}

/// Async version: Convert a DatasetSelection to a GroupedChunkPlan (unified ZarrMeta).
pub(crate) async fn selection_to_grouped_chunk_plan_unified_async(
    selection: &DatasetSelection,
    meta: &ZarrMeta,
    store: zarrs::storage::AsyncReadableWritableListableStorage,
) -> Result<GroupedChunkPlan, CompileError> {
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
        let var_key = var.istr();
        let Some(var_meta) =
            meta.array_by_path(var)
        else {
            continue;
        };

        let arr = zarrs::array::Array::async_open(
            store.clone(),
            var_meta.path.as_ref(),
        )
        .await
        .map_err(|e| {
            CompileError::Unsupported(format!(
                "failed to open array '{var}': {e}"
            ))
        })?;

        let zero =
            vec![0u64; arr.dimensionality()];
        let chunk_shape_nz = arr
            .chunk_shape(&zero)
            .map_err(|e| {
                CompileError::Unsupported(
                    e.to_string(),
                )
            })?;
        let chunk_shape: SmallVec<[u64; 4]> =
            chunk_shape_nz
                .iter()
                .map(|nz| nz.get())
                .collect();

        let sig = ChunkGridSignature::new(
            var_meta.dims.clone(),
            chunk_shape,
        );
        let sig_arc = sig_cache
            .entry(sig.clone())
            .or_insert_with(|| Arc::new(sig))
            .clone();

        // When maybe_sel is None, we select ALL chunks (conservative fallback)
        let chunk_plan =
            if let Some(sel) = maybe_sel {
                sel.clone().into()
            } else {
                all_chunks_subset(arr.shape())
            };

        grouped_plan.insert(
            var_key,
            sig_arc,
            chunk_plan,
            Arc::new(arr.chunk_grid().clone()),
        );
    }

    Ok(grouped_plan)
}

/// Convert a DatasetSelection to a GroupedChunkPlan using metadata only.
///
/// This version doesn't require a store - it creates ChunkGrids from the
/// metadata's shape and chunk_shape information. This enables backends
/// that don't expose a raw store to still compile chunk plans.
pub(crate) fn selection_to_grouped_chunk_plan_unified_from_meta(
    selection: &DatasetSelection,
    meta: &ZarrMeta,
) -> Result<GroupedChunkPlan, CompileError> {
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
        let var_key = var.istr();
        let Some(var_meta) =
            meta.array_by_path(var)
        else {
            continue;
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

        // Create ChunkGrid from array shape and chunk shape
        let chunk_grid =
            create_chunk_grid_from_shapes(
                &var_meta.shape,
                &chunk_shape,
            )?;

        grouped_plan.insert(
            var_key,
            sig_arc,
            chunk_plan,
            Arc::new(chunk_grid),
        );
    }

    Ok(grouped_plan)
}

/// Create a regular ChunkGrid from array shape and chunk shape.
fn create_chunk_grid_from_shapes(
    array_shape: &[u64],
    chunk_shape: &[u64],
) -> Result<ChunkGrid, CompileError> {
    let chunk_shape_nz: Vec<NonZeroU64> =
        chunk_shape
            .iter()
            .map(|&s| {
                NonZeroU64::new(s).unwrap_or(
                    NonZeroU64::new(1).unwrap(),
                )
            })
            .collect();

    let regular_grid = RegularChunkGrid::new(
        array_shape.to_vec(),
        chunk_shape_nz.try_into().map_err(
            |_| {
                CompileError::Unsupported(
                    "invalid chunk shape".into(),
                )
            },
        )?,
    )
    .map_err(|e| {
        CompileError::Unsupported(format!(
            "failed to create chunk grid: {:?}",
            e
        ))
    })?;

    Ok(ChunkGrid::new(regular_grid))
}
