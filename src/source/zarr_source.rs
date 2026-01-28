use std::collections::BTreeSet;
use std::sync::Arc;

use polars::prelude::Expr;
use pyo3::prelude::*;
use zarrs::array_subset::ArraySubset;

use crate::chunk_plan::ChunkGridSignature;
use crate::meta::{ZarrDatasetMeta, ZarrMeta};
use crate::IStr;

pub(super) const DEFAULT_BATCH_SIZE: usize =
    10_000;

/// Iteration state for a single chunk grid signature.
///
/// Each grid has its own variables and pending element subsets to read.
#[derive(Debug)]
pub(super) struct GridIterState {
    pub signature: Arc<ChunkGridSignature>,
    pub variables: Vec<IStr>,
    /// Pending array subsets (element ranges) to read
    pub pending_subsets:
        std::collections::VecDeque<ArraySubset>,
    /// Current subset being processed
    pub current_subset: Option<ArraySubset>,
    /// Offset within current subset (for batching)
    pub subset_offset: usize,
}

impl GridIterState {
    pub fn new(
        signature: Arc<ChunkGridSignature>,
        variables: Vec<IStr>,
    ) -> Self {
        Self {
            signature,
            variables,
            pending_subsets:
                std::collections::VecDeque::new(),
            current_subset: None,
            subset_offset: 0,
        }
    }

    /// Add subsets from an ArraySubsetList
    pub fn add_subsets(
        &mut self,
        subsets: impl Iterator<Item = ArraySubset>,
    ) {
        self.pending_subsets.extend(subsets);
    }

    /// Get the next subset to process, advancing if current is exhausted
    pub fn advance(
        &mut self,
    ) -> Option<&ArraySubset> {
        if self.current_subset.is_none() {
            self.current_subset =
                self.pending_subsets.pop_front();
            self.subset_offset = 0;
        }
        self.current_subset.as_ref()
    }

    /// Mark current subset as done, move to next
    pub fn finish_current(&mut self) {
        self.current_subset = None;
        self.subset_offset = 0;
    }

    /// Check if this grid has any work left
    pub fn is_exhausted(&self) -> bool {
        self.current_subset.is_none()
            && self.pending_subsets.is_empty()
    }
}

#[pyclass]
pub struct ZarrSource {
    /// Legacy flat metadata (for backward compatibility)
    pub(super) meta: ZarrDatasetMeta,
    /// Unified hierarchical metadata (when available)
    pub(super) unified_meta: Option<ZarrMeta>,
    pub(super) store:
        zarrs::storage::ReadableWritableListableStorage,

    pub(super) dims: Vec<IStr>,
    pub(super) vars: Vec<IStr>,

    pub(super) batch_size: usize,
    pub(super) n_rows_left: usize,

    pub(super) predicate: Option<Expr>,
    pub(super) with_columns: Option<BTreeSet<IStr>>,

    // Per-grid iteration state
    pub(super) grid_states: Vec<GridIterState>,
    pub(super) current_grid_idx: usize,
    pub(super) done: bool,

    // Optional cap on number of chunks we will read (for debugging / safety).
    pub(super) chunks_left: Option<usize>,

    /// True if this is a hierarchical (DataTree) store
    pub(super) is_hierarchical: bool,
}

pub(super) fn to_py_err<E: std::fmt::Display>(
    e: E,
) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(
        e.to_string(),
    )
}

pub(super) fn panic_to_py_err(
    e: Box<dyn std::any::Any + Send>,
    msg2: &str,
) -> PyErr {
    let msg = if let Some(s) =
        e.downcast_ref::<&str>()
    {
        format!("{msg2}: {}", s.to_string())
    } else if let Some(s) =
        e.downcast_ref::<String>()
    {
        format!("{msg2}: {}", s.clone())
    } else {
        format!("{msg2}: {e:?}")
    };
    PyErr::new::<
        pyo3::exceptions::PyRuntimeError,
        _,
    >(msg)
}

impl ZarrSource {
    pub(super) fn should_emit(
        &self,
        name: &str,
    ) -> bool {
        self.with_columns
            .as_ref()
            .map(|s| {
                s.iter().any(|c| {
                    <IStr as AsRef<str>>::as_ref(
                        c,
                    ) == name
                })
            })
            .unwrap_or(true)
    }
}

// Keep the implementation split into proper submodules, but nest them under
// `zarr_source` so they can access `ZarrSource` private fields (like the old
// `include!` layout allowed).
mod zarr_source_new;
mod zarr_source_next;
mod zarr_source_predicate;
mod zarr_source_pymethods;
