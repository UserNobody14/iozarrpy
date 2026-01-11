use std::collections::BTreeSet;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PyExpr, PySchema};
use zarrs::array::Array;
// (moved to `reader::retrieve_*`)

use crate::chunk_plan::{compile_expr_to_chunk_plan, ChunkIndexIter, ChunkPlan};
use crate::reader::{
    checked_chunk_len, compute_strides, compute_var_chunk_info, retrieve_1d_subset, retrieve_chunk,
    ColumnData,
};
use crate::meta::ZarrDatasetMeta;

const DEFAULT_BATCH_SIZE: usize = 10_000;

#[pyclass]
pub struct ZarrSource {
    meta: ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,

    dims: Vec<String>,
    vars: Vec<String>,

    batch_size: usize,
    n_rows_left: usize,

    predicate: Option<Expr>,
    with_columns: Option<BTreeSet<String>>,

    // Iteration state
    primary_grid_shape: Vec<u64>,
    chunk_iter: ChunkIndexIter,
    current_chunk_indices: Option<Vec<u64>>,
    chunk_offset: usize,
    done: bool,

    // Optional cap on number of chunks we will read (for debugging / safety).
    chunks_left: Option<usize>,
}

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

fn panic_to_py_err(e: Box<dyn std::any::Any + Send>) -> PyErr {
    let msg = if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic while compiling predicate chunk plan".to_string()
    };
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg)
}

impl ZarrSource {
    fn should_emit(&self, name: &str) -> bool {
        self.with_columns
            .as_ref()
            .map(|s| s.contains(name))
            .unwrap_or(true)
    }
}

