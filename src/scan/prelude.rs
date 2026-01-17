pub(crate) use std::collections::BTreeSet;
pub(crate) use std::sync::Arc;

pub(crate) use futures::stream::{FuturesUnordered, StreamExt};
pub(crate) use polars::prelude::*;
pub(crate) use pyo3::prelude::*;
pub(crate) use pyo3_polars::error::PyPolarsErr;
pub(crate) use zarrs::array::Array;
// (moved to `reader::retrieve_*`)

pub(crate) use crate::chunk_plan::{compile_expr_to_chunk_plan, ChunkPlan};
pub(crate) use crate::reader::{
    checked_chunk_len, compute_strides, compute_var_chunk_info_async, retrieve_1d_subset_async,
    retrieve_chunk_async, ColumnData,
};
pub(crate) use crate::meta::{open_and_load_dataset_meta_async, ZarrDatasetMeta};
pub(crate) use crate::store::open_store;

pub(super) const DEFAULT_MAX_CONCURRENCY: usize = 16;

pub(super) fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

pub(super) fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}
