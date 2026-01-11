use std::collections::BTreeSet;
use std::sync::Arc;

use futures::stream::{FuturesUnordered, StreamExt};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use zarrs::array::Array;
// (moved to `reader::retrieve_*`)

use crate::chunk_plan::{compile_expr_to_chunk_plan, ChunkPlan};
use crate::reader::{
    checked_chunk_len, compute_strides, compute_var_chunk_info_async, retrieve_1d_subset_async,
    retrieve_chunk_async, ColumnData,
};
use crate::meta::{load_dataset_meta_from_opened_async, ZarrDatasetMeta};
use crate::store::{open_store, open_store_async};

const DEFAULT_MAX_CONCURRENCY: usize = 16;

fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

