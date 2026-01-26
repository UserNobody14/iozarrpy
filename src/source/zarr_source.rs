use std::collections::BTreeSet;

use polars::prelude::Expr;
use pyo3::prelude::*;

use crate::chunk_plan::ChunkIndexIter;
use crate::meta::ZarrDatasetMeta;
use crate::IStr;

pub(super) const DEFAULT_BATCH_SIZE: usize = 10_000;

#[pyclass]
pub struct ZarrSource {
    meta: ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,

    dims: Vec<IStr>,
    vars: Vec<IStr>,

    batch_size: usize,
    n_rows_left: usize,

    predicate: Option<Expr>,
    with_columns: Option<BTreeSet<IStr>>,

    // Iteration state
    primary_grid_shape: Vec<u64>,
    chunk_iter: ChunkIndexIter,
    current_chunk_indices: Option<Vec<u64>>,
    chunk_offset: usize,
    done: bool,

    // Optional cap on number of chunks we will read (for debugging / safety).
    chunks_left: Option<usize>,
}

pub(super) fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

pub(super) fn panic_to_py_err(e: Box<dyn std::any::Any + Send>, msg2: &str) -> PyErr {
    let msg = if let Some(s) = e.downcast_ref::<&str>() {
        format!("{msg2}: {}", s.to_string())
    } else if let Some(s) = e.downcast_ref::<String>() {
        format!("{msg2}: {}", s.clone())
    } else {
        format!("{msg2}: {e:?}")
    };
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg)
}

impl ZarrSource {
    pub(super) fn should_emit(&self, name: &str) -> bool {
        self.with_columns
            .as_ref()
            .map(|s| s.iter().any(|c| <IStr as AsRef<str>>::as_ref(c) == name))
            .unwrap_or(true)
    }
}

// Keep the implementation split into proper submodules, but nest them under
// `zarr_source` so they can access `ZarrSource` private fields (like the old
// `include!` layout allowed).
mod zarr_source_new;
mod zarr_source_predicate;
mod zarr_source_next;
mod zarr_source_pymethods;

