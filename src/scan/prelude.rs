pub(crate) use std::collections::BTreeSet;
pub(crate) use std::sync::Arc;

pub(crate) use futures::stream::{
    FuturesUnordered, StreamExt,
};
pub(crate) use polars::prelude::*;
pub(crate) use pyo3::prelude::*;
pub(crate) use pyo3_polars::error::PyPolarsErr;
pub(crate) use zarrs::array::Array;

pub(crate) use crate::IStr;

pub(crate) use crate::meta::{
    ZarrDatasetMeta, ZarrMeta,
};
pub(crate) use crate::reader::{
    ColumnData, checked_chunk_len,
    compute_strides,
};

pub(super) fn to_string_err<
    E: std::fmt::Display,
>(
    e: E,
) -> String {
    e.to_string()
}

pub(super) fn to_py_err<E: std::fmt::Display>(
    e: E,
) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(
        e.to_string(),
    )
}
