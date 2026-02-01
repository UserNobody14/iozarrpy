use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use smallvec::SmallVec;
use zarrs::array::Array;
use zarrs::array::ArrayShardedExt;
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

        // Use inner_chunk_grid which gives inner chunks for sharded arrays,
        // or regular chunks for non-sharded arrays
        let inner_grid = arr.inner_chunk_grid();

        let zero =
            vec![0u64; arr.dimensionality()];
        let chunk_shape_opt = inner_grid
            .chunk_shape_u64(&zero)
            .map_err(|e| {
                CompileError::Unsupported(
                    e.to_string(),
                )
            })?
            .ok_or_else(|| {
                CompileError::Unsupported(
                    "could not determine chunk shape".to_string(),
                )
            })?;
        let chunk_shape: SmallVec<[u64; 4]> =
            chunk_shape_opt.into_iter().collect();

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
            Arc::new(inner_grid),
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

        // Use inner_chunk_grid which gives inner chunks for sharded arrays,
        // or regular chunks for non-sharded arrays
        let inner_grid = arr.inner_chunk_grid();

        let zero =
            vec![0u64; arr.dimensionality()];
        let chunk_shape_opt = inner_grid
            .chunk_shape_u64(&zero)
            .map_err(|e| {
                CompileError::Unsupported(
                    e.to_string(),
                )
            })?
            .ok_or_else(|| {
                CompileError::Unsupported(
                    "could not determine chunk shape".to_string(),
                )
            })?;
        let chunk_shape: SmallVec<[u64; 4]> =
            chunk_shape_opt.into_iter().collect();

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
            Arc::new(inner_grid),
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
