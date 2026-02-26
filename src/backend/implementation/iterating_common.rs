//! Shared types and utilities for zarr iterators (sync and async).
//!
//! Both `ZarrIterator` and `IcechunkIterator` use the same chunk-planning and
//! batch-postprocessing logic; this module centralizes the common pieces.

use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::*;

use crate::IStr;
use crate::chunk_plan::ChunkSubset;
use crate::chunk_plan::ConsolidatedGridGroup;
use crate::meta::ZarrMeta;
use crate::shared::{
    combine_chunk_dataframes,
    restructure_to_structs,
};

/// Maximum chunks to read per batch iteration (avoids unbounded memory growth).
pub(crate) const CHUNKS_PER_BATCH_LIMIT: usize =
    100;

/// Default batch size in rows when not specified.
pub(crate) const DEFAULT_BATCH_SIZE: usize =
    10_000;

/// Owned version of [`ConsolidatedGridGroup`] for storage in iterator state.
///
/// We need owned data because [`GroupedChunkPlan`] is dropped after
/// initialization, but we continue iterating over the chunk indices.
pub(crate) struct OwnedGridGroup {
    pub sig: Arc<
        crate::chunk_plan::ChunkGridSignature,
    >,
    pub vars: Vec<IStr>,
    pub chunk_indices: Vec<Vec<u64>>,
    pub chunk_subsets: Vec<Option<ChunkSubset>>,
    pub array_shape: Vec<u64>,
}

impl OwnedGridGroup {
    /// Convert from a borrowed [`ConsolidatedGridGroup`] produced by the chunk planner.
    #[inline]
    pub fn from_consolidated(
        group: ConsolidatedGridGroup<'_>,
    ) -> Self {
        Self {
            sig: Arc::new(group.sig.clone()),
            vars: group.vars,
            chunk_indices: group.chunk_indices,
            chunk_subsets: group.chunk_subsets,
            array_shape: group.array_shape,
        }
    }
}

/// Iterator state shared by both sync and async zarr iterators.
pub(crate) struct IteratorState {
    pub grid_groups: Vec<OwnedGridGroup>,
    pub current_group_idx: usize,
    pub current_chunk_idx: usize,

    pub current_batch: Vec<DataFrame>,
    pub current_batch_rows: usize,

    pub total_rows_yielded: usize,
    pub num_rows_limit: Option<usize>,

    pub meta: Arc<ZarrMeta>,
    pub expanded_with_columns:
        Option<BTreeSet<IStr>>,

    pub predicate: Expr,
}

/// Collect the next batch of chunk indices to read from the current group.
///
/// Advances `current_chunk_idx` and returns up to `CHUNKS_PER_BATCH_LIMIT` chunks,
/// or until `current_batch_rows` would exceed `batch_size`, or the group is exhausted.
///
/// Returns `(chunks_to_read, num_chunks_advanced)`.
#[inline]
pub(crate) fn collect_chunks_for_batch(
    group: &OwnedGridGroup,
    current_chunk_idx: &mut usize,
    current_batch_rows: usize,
    batch_size: usize,
) -> Vec<(Vec<u64>, Option<ChunkSubset>)> {
    let mut chunks_to_read = Vec::new();
    while *current_chunk_idx
        < group.chunk_indices.len()
        && current_batch_rows < batch_size
    {
        let ci = *current_chunk_idx;
        chunks_to_read.push((
            group.chunk_indices[ci].clone(),
            group.chunk_subsets[ci].clone(),
        ));
        *current_chunk_idx += 1;

        if chunks_to_read.len()
            >= CHUNKS_PER_BATCH_LIMIT
        {
            break;
        }
    }
    chunks_to_read
}

/// Combine chunk DataFrames, apply predicate filter, row limit, and restructure.
///
/// Shared post-processing for both sync and async iterators.
pub(crate) fn combine_and_postprocess_batch(
    batch_dfs: Vec<DataFrame>,
    state: &mut IteratorState,
) -> Result<DataFrame, pyo3::PyErr> {
    let combined = if batch_dfs.is_empty() {
        let keys: Vec<IStr> =
            state.meta.all_array_paths();
        DataFrame::empty_with_schema(
            &state.meta.tidy_schema(Some(
                keys.as_slice(),
            )),
        )
    } else if batch_dfs.len() == 1 {
        batch_dfs.into_iter().next().unwrap()
    } else {
        combine_chunk_dataframes(
            batch_dfs,
            &state.meta,
        )?
    };

    let result = combined
        .lazy()
        .filter(state.predicate.clone())
        .collect()
        .map_err(|e| {
            pyo3::PyErr::new::<
                pyo3::exceptions::PyRuntimeError,
                _,
            >(format!(
                "Failed to filter data: {}",
                e
            ))
        })?;

    let result = if let Some(limit) =
        state.num_rows_limit
    {
        let remaining = limit.saturating_sub(
            state.total_rows_yielded,
        );
        if remaining < result.height() {
            result.slice(0, remaining)
        } else {
            result
        }
    } else {
        result
    };

    state.total_rows_yielded += result.height();

    let result = if state.meta.is_hierarchical() {
        restructure_to_structs(
            &result,
            &state.meta,
        )?
    } else {
        result
    };

    Ok(result)
}
