use std::collections::BTreeSet;
use std::sync::Arc;

use futures::stream::{FuturesUnordered, StreamExt};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PyExpr};
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;

use crate::chunk_plan::{compile_expr_to_chunk_plan, ChunkPlan};
use crate::zarr_meta::{load_dataset_meta_from_opened_async, ZarrDatasetMeta};
use crate::zarr_store::{open_store, open_store_async};

const DEFAULT_MAX_CONCURRENCY: usize = 16;
const DEFAULT_MAX_CHUNK_ELEMS: usize = 50_000_000;

fn max_chunk_elems() -> usize {
    std::env::var("RAINBEAR_MAX_CHUNK_ELEMS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CHUNK_ELEMS)
}

fn checked_chunk_len(shape: &[u64]) -> PyResult<usize> {
    let mut acc: usize = 1;
    for &d in shape {
        let d_usize: usize = d.try_into().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyMemoryError, _>(
                "chunk shape dimension does not fit in usize",
            )
        })?;
        acc = acc.checked_mul(d_usize).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyMemoryError, _>("chunk size overflow")
        })?;
        if acc > max_chunk_elems() {
            return Err(PyErr::new::<pyo3::exceptions::PyMemoryError, _>(
                "refusing to allocate an extremely large chunk; set RAINBEAR_MAX_CHUNK_ELEMS to override",
            ));
        }
    }
    Ok(acc)
}

#[derive(Debug, Clone)]
enum ColumnData {
    Bool(Vec<bool>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl ColumnData {
    fn len(&self) -> usize {
        match self {
            ColumnData::Bool(v) => v.len(),
            ColumnData::I8(v) => v.len(),
            ColumnData::I16(v) => v.len(),
            ColumnData::I32(v) => v.len(),
            ColumnData::I64(v) => v.len(),
            ColumnData::U8(v) => v.len(),
            ColumnData::U16(v) => v.len(),
            ColumnData::U32(v) => v.len(),
            ColumnData::U64(v) => v.len(),
            ColumnData::F32(v) => v.len(),
            ColumnData::F64(v) => v.len(),
        }
    }

    fn take_indices(&self, indices: &[usize]) -> ColumnData {
        match self {
            ColumnData::Bool(v) => ColumnData::Bool(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::I8(v) => ColumnData::I8(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::I16(v) => ColumnData::I16(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::I32(v) => ColumnData::I32(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::I64(v) => ColumnData::I64(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::U8(v) => ColumnData::U8(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::U16(v) => ColumnData::U16(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::U32(v) => ColumnData::U32(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::U64(v) => ColumnData::U64(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::F32(v) => ColumnData::F32(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::F64(v) => ColumnData::F64(indices.iter().map(|&i| v[i]).collect()),
        }
    }

    fn get_f64(&self, idx: usize) -> Option<f64> {
        match self {
            ColumnData::F64(v) => Some(v[idx]),
            ColumnData::F32(v) => Some(v[idx] as f64),
            ColumnData::I64(v) => Some(v[idx] as f64),
            ColumnData::I32(v) => Some(v[idx] as f64),
            ColumnData::I16(v) => Some(v[idx] as f64),
            ColumnData::I8(v) => Some(v[idx] as f64),
            ColumnData::U64(v) => Some(v[idx] as f64),
            ColumnData::U32(v) => Some(v[idx] as f64),
            ColumnData::U16(v) => Some(v[idx] as f64),
            ColumnData::U8(v) => Some(v[idx] as f64),
            ColumnData::Bool(_) => None,
        }
    }

    fn get_i64(&self, idx: usize) -> Option<i64> {
        match self {
            ColumnData::I64(v) => Some(v[idx]),
            ColumnData::I32(v) => Some(v[idx] as i64),
            ColumnData::I16(v) => Some(v[idx] as i64),
            ColumnData::I8(v) => Some(v[idx] as i64),
            ColumnData::U64(v) => Some(v[idx] as i64),
            ColumnData::U32(v) => Some(v[idx] as i64),
            ColumnData::U16(v) => Some(v[idx] as i64),
            ColumnData::U8(v) => Some(v[idx] as i64),
            ColumnData::F32(v) => Some(v[idx] as i64),
            ColumnData::F64(v) => Some(v[idx] as i64),
            ColumnData::Bool(v) => Some(i64::from(v[idx])),
        }
    }

    fn is_float(&self) -> bool {
        matches!(self, ColumnData::F32(_) | ColumnData::F64(_))
    }

    fn into_series(self, name: &str) -> Series {
        match self {
            ColumnData::Bool(v) => Series::new(name.into(), v),
            ColumnData::I8(v) => Series::new(name.into(), v),
            ColumnData::I16(v) => Series::new(name.into(), v),
            ColumnData::I32(v) => Series::new(name.into(), v),
            ColumnData::I64(v) => Series::new(name.into(), v),
            ColumnData::U8(v) => Series::new(name.into(), v),
            ColumnData::U16(v) => Series::new(name.into(), v),
            ColumnData::U32(v) => Series::new(name.into(), v),
            ColumnData::U64(v) => Series::new(name.into(), v),
            ColumnData::F32(v) => Series::new(name.into(), v),
            ColumnData::F64(v) => Series::new(name.into(), v),
        }
    }
}

async fn retrieve_chunk_async(
    array: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    chunk: &[u64],
) -> Result<ColumnData, String> {
    let id = array.data_type().identifier();
    match id {
        "bool" => Ok(ColumnData::Bool(
            array
                .async_retrieve_chunk::<Vec<bool>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        "int8" => Ok(ColumnData::I8(
            array
                .async_retrieve_chunk::<Vec<i8>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        "int16" => Ok(ColumnData::I16(
            array
                .async_retrieve_chunk::<Vec<i16>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        "int32" => Ok(ColumnData::I32(
            array
                .async_retrieve_chunk::<Vec<i32>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        "int64" => Ok(ColumnData::I64(
            array
                .async_retrieve_chunk::<Vec<i64>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        "uint8" => Ok(ColumnData::U8(
            array
                .async_retrieve_chunk::<Vec<u8>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        "uint16" => Ok(ColumnData::U16(
            array
                .async_retrieve_chunk::<Vec<u16>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        "uint32" => Ok(ColumnData::U32(
            array
                .async_retrieve_chunk::<Vec<u32>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        "uint64" => Ok(ColumnData::U64(
            array
                .async_retrieve_chunk::<Vec<u64>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        "float32" => Ok(ColumnData::F32(
            array
                .async_retrieve_chunk::<Vec<f32>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        "float64" => Ok(ColumnData::F64(
            array
                .async_retrieve_chunk::<Vec<f64>>(chunk)
                .await
                .map_err(to_string_err)?,
        )),
        other => Err(format!("unsupported zarr dtype: {other}")),
    }
}

async fn retrieve_1d_subset_async(
    array: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
    start: u64,
    len: u64,
) -> Result<ColumnData, String> {
    if len as u128 > max_chunk_elems() as u128 {
        return Err("refusing to allocate extremely large coordinate subset; set RAINBEAR_MAX_CHUNK_ELEMS to override".to_string());
    }
    let subset = ArraySubset::new_with_ranges(&[start..(start + len)]);
    let id = array.data_type().identifier();
    match id {
        "bool" => Ok(ColumnData::Bool(
            array
                .async_retrieve_array_subset::<Vec<bool>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        "int8" => Ok(ColumnData::I8(
            array
                .async_retrieve_array_subset::<Vec<i8>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        "int16" => Ok(ColumnData::I16(
            array
                .async_retrieve_array_subset::<Vec<i16>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        "int32" => Ok(ColumnData::I32(
            array
                .async_retrieve_array_subset::<Vec<i32>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        "int64" => Ok(ColumnData::I64(
            array
                .async_retrieve_array_subset::<Vec<i64>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        "uint8" => Ok(ColumnData::U8(
            array
                .async_retrieve_array_subset::<Vec<u8>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        "uint16" => Ok(ColumnData::U16(
            array
                .async_retrieve_array_subset::<Vec<u16>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        "uint32" => Ok(ColumnData::U32(
            array
                .async_retrieve_array_subset::<Vec<u32>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        "uint64" => Ok(ColumnData::U64(
            array
                .async_retrieve_array_subset::<Vec<u64>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        "float32" => Ok(ColumnData::F32(
            array
                .async_retrieve_array_subset::<Vec<f32>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        "float64" => Ok(ColumnData::F64(
            array
                .async_retrieve_array_subset::<Vec<f64>>(&subset)
                .await
                .map_err(to_string_err)?,
        )),
        other => Err(format!("unsupported zarr dtype: {other}")),
    }
}

fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

fn compute_strides(chunk_shape: &[u64]) -> Vec<u64> {
    let mut strides = vec![1u64; chunk_shape.len()];
    for i in (0..chunk_shape.len()).rev() {
        if i + 1 < chunk_shape.len() {
            strides[i] = strides[i + 1] * chunk_shape[i + 1];
        }
    }
    strides
}

fn compute_var_chunk_info(
    primary_chunk_indices: &[u64],
    primary_chunk_shape: &[u64],
    primary_dims: &[String],
    var_dims: &[String],
    var_array: &Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let mut var_chunk_indices = Vec::with_capacity(var_dims.len());
    let mut offsets = Vec::with_capacity(var_dims.len());

    for var_dim in var_dims {
        if let Some(primary_d) = primary_dims.iter().position(|pd| pd == var_dim) {
            let primary_chunk_idx = primary_chunk_indices[primary_d];
            let primary_chunk_size = primary_chunk_shape[primary_d];
            let global_start = primary_chunk_idx * primary_chunk_size;

            let var_dim_idx = var_dims.iter().position(|vd| vd == var_dim).unwrap();

            let zero_indices: Vec<u64> = vec![0; var_dims.len()];
            let var_regular_chunk_shape = var_array
                .chunk_shape(&zero_indices)
                .map_err(to_string_err)?;
            let var_chunk_size = var_regular_chunk_shape[var_dim_idx].get();

            let var_chunk_idx = global_start / var_chunk_size;
            let offset = global_start % var_chunk_size;

            var_chunk_indices.push(var_chunk_idx);
            offsets.push(offset);
        } else {
            return Err(format!("variable dimension {var_dim} not found in primary dims"));
        }
    }

    Ok((var_chunk_indices, offsets))
}

async fn open_arrays_async(
    store: zarrs::storage::AsyncReadableWritableListableStorage,
    meta: &ZarrDatasetMeta,
    vars: &[String],
    dims: &[String],
) -> Result<
    (
        Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>,
        Vec<(String, Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>)>,
        Vec<(String, Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>)>,
    ),
    String,
> {
    let primary_path = meta
        .arrays
        .get(&vars[0])
        .ok_or_else(|| "unknown primary variable".to_string())?
        .path
        .clone();

    let primary = Array::async_open(store.clone(), &primary_path)
        .await
        .map_err(to_string_err)?;
    let primary = Arc::new(primary);

    // Open coord arrays (dims) and variable arrays in parallel.
    let mut coord_futs = FuturesUnordered::new();
    for d in dims {
        if let Some(m) = meta.arrays.get(d) {
            let path = m.path.clone();
            let d_name = d.clone();
            let st = store.clone();
            coord_futs.push(async move {
                let arr = Array::async_open(st, &path).await.map_err(to_string_err)?;
                Ok::<_, String>((d_name, Arc::new(arr)))
            });
        }
    }

    let mut var_futs = FuturesUnordered::new();
    for v in vars {
        let Some(m) = meta.arrays.get(v) else {
            continue;
        };
        let path = m.path.clone();
        let v_name = v.clone();
        let st = store.clone();
        var_futs.push(async move {
            let arr = Array::async_open(st, &path).await.map_err(to_string_err)?;
            Ok::<_, String>((v_name, Arc::new(arr)))
        });
    }

    let mut coords = Vec::new();
    while let Some(r) = coord_futs.next().await {
        coords.push(r?);
    }
    let mut vars_out = Vec::new();
    while let Some(r) = var_futs.next().await {
        vars_out.push(r?);
    }

    Ok((primary, vars_out, coords))
}

async fn chunk_to_df(
    idx: Vec<u64>,
    primary: Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>,
    meta: Arc<ZarrDatasetMeta>,
    dims: Arc<Vec<String>>,
    _vars: Arc<Vec<String>>,
    var_arrays: Arc<Vec<(String, Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>)>>,
    coord_arrays: Arc<Vec<(String, Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>)>>,
    with_columns: Arc<Option<BTreeSet<String>>>,
) -> Result<DataFrame, PyErr> {
    // Compute primary chunk geometry.
    let chunk_shape_nz = primary.chunk_shape(&idx).map_err(to_py_err)?;
    let chunk_shape: Vec<u64> = chunk_shape_nz.iter().map(|x| x.get()).collect();
    let chunk_len = checked_chunk_len(&chunk_shape)?;

    let array_shape = primary.shape().to_vec();
    let origin = primary
        .chunk_grid()
        .chunk_origin(&idx)
        .map_err(to_py_err)?
        .unwrap_or_else(|| vec![0; chunk_shape.len()]);
    let strides = compute_strides(&chunk_shape);

    // In-bounds mask.
    let mut keep: Vec<usize> = Vec::with_capacity(chunk_len);
    for row in 0..chunk_len {
        let mut ok = true;
        for d in 0..chunk_shape.len() {
            let local = (row as u64 / strides[d]) % chunk_shape[d];
            let global = origin[d] + local;
            if global >= array_shape[d] {
                ok = false;
                break;
            }
        }
        if ok {
            keep.push(row);
        }
    }

    // Coord reads (per dim) in parallel.
    let mut coord_reads = FuturesUnordered::new();
    for (d, dim_name) in dims.iter().enumerate() {
        if !with_columns
            .as_ref()
            .as_ref()
            .map(|s| s.contains(dim_name))
            .unwrap_or(true)
        {
            continue;
        }
        let Some((_, arr)) = coord_arrays.iter().find(|(n, _)| n == dim_name) else {
            continue;
        };
        let dim_start = origin[d];
        let dim_len = chunk_shape[d];
        let arr = Arc::clone(arr);
        let dim_name = dim_name.clone();
        coord_reads.push(async move {
            let coord = retrieve_1d_subset_async(&arr, dim_start, dim_len).await;
            (dim_name, coord)
        });
    }
    let mut coord_slices: std::collections::BTreeMap<String, ColumnData> = Default::default();
    while let Some((name, res)) = coord_reads.next().await {
        coord_slices.insert(name, res.map_err(to_py_err)?);
    }

    // Var chunk reads in parallel.
    let mut var_reads = FuturesUnordered::new();
    for (name, arr) in var_arrays.iter() {
        if !with_columns
            .as_ref()
            .as_ref()
            .map(|s| s.contains(name))
            .unwrap_or(true)
        {
            continue;
        }
        let var_meta = meta
            .arrays
            .get(name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown variable"))?
            .clone();
        let arr = Arc::clone(arr);
        let name = name.clone();
        let dims = Arc::clone(&dims);
        let idx = idx.clone();
        let chunk_shape = chunk_shape.clone();
        var_reads.push(async move {
            let (var_chunk_indices, var_offsets) = if var_meta.dims.len() == dims.len()
                && var_meta.dims == *dims
            {
                (idx.clone(), vec![0; dims.len()])
            } else {
                compute_var_chunk_info(&idx, &chunk_shape, &dims, &var_meta.dims, &arr)
                    .map_err(to_py_err)?
            };

            let var_chunk_shape: Vec<u64> = if var_chunk_indices.is_empty() {
                vec![]
            } else {
                arr.chunk_shape(&var_chunk_indices)
                    .map_err(to_py_err)?
                    .iter()
                    .map(|x| x.get())
                    .collect()
            };
            if !var_chunk_shape.is_empty() {
                let _ = checked_chunk_len(&var_chunk_shape)?;
            }

            let data = retrieve_chunk_async(&arr, &var_chunk_indices).await.map_err(to_py_err)?;
            Ok::<_, PyErr>((name, data, var_meta.dims, var_chunk_shape, var_offsets))
        });
    }

    let mut var_chunks: Vec<(String, ColumnData, Vec<String>, Vec<u64>, Vec<u64>)> = Vec::new();
    while let Some(r) = var_reads.next().await {
        var_chunks.push(r?);
    }

    let mut cols: Vec<Column> = Vec::new();

    // Coord columns.
    for (d, dim_name) in dims.iter().enumerate() {
        if !with_columns
            .as_ref()
            .as_ref()
            .map(|s| s.contains(dim_name))
            .unwrap_or(true)
        {
            continue;
        }

        let time_encoding = meta
            .arrays
            .get(dim_name)
            .and_then(|m| m.time_encoding.as_ref());

        if let Some(te) = time_encoding {
            let mut out_i64: Vec<i64> = Vec::with_capacity(keep.len());
            for &row in &keep {
                let local = (row as u64 / strides[d]) % chunk_shape[d];
                let raw_value = coord_slices
                    .get(dim_name)
                    .and_then(|c| c.get_i64(local as usize))
                    .unwrap_or((origin[d] + local) as i64);
                let ns = if te.is_duration {
                    raw_value.saturating_mul(te.unit_ns)
                } else {
                    raw_value
                        .saturating_mul(te.unit_ns)
                        .saturating_add(te.epoch_ns)
                };
                out_i64.push(ns);
            }
            let series = if te.is_duration {
                Series::new(dim_name.into(), &out_i64)
                    .cast(&DataType::Duration(TimeUnit::Nanoseconds))
                    .unwrap_or_else(|_| Series::new(dim_name.into(), out_i64))
            } else {
                Series::new(dim_name.into(), &out_i64)
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .unwrap_or_else(|_| Series::new(dim_name.into(), out_i64))
            };
            cols.push(series.into());
        } else if let Some(coord) = coord_slices.get(dim_name)
            && coord.is_float()
        {
            let mut out_f64: Vec<f64> = Vec::with_capacity(keep.len());
            for &row in &keep {
                let local = (row as u64 / strides[d]) % chunk_shape[d];
                out_f64.push(coord.get_f64(local as usize).unwrap());
            }
            cols.push(Series::new(dim_name.into(), out_f64).into());
        } else {
            let mut out_i64: Vec<i64> = Vec::with_capacity(keep.len());
            for &row in &keep {
                let local = (row as u64 / strides[d]) % chunk_shape[d];
                if let Some(coord) = coord_slices.get(dim_name) {
                    if let Some(v) = coord.get_i64(local as usize) {
                        out_i64.push(v);
                    } else {
                        out_i64.push((origin[d] + local) as i64);
                    }
                } else {
                    out_i64.push((origin[d] + local) as i64);
                }
            }
            cols.push(Series::new(dim_name.into(), out_i64).into());
        }
    }

    // Variable columns.
    for (name, data, var_dims, var_chunk_shape, var_offsets) in var_chunks {
        if var_dims.len() == dims.len() && var_dims == *dims && var_offsets.iter().all(|&o| o == 0)
        {
            cols.push(data.take_indices(&keep).into_series(&name).into());
        } else {
            let dim_mapping: Vec<Option<usize>> = dims
                .iter()
                .map(|pd| var_dims.iter().position(|vd| vd == pd))
                .collect();
            let var_strides = compute_strides(&var_chunk_shape);
            let indices: Vec<usize> = keep
                .iter()
                .map(|&row| {
                    let mut var_idx: u64 = 0;
                    for (primary_d, maybe_var_d) in dim_mapping.iter().enumerate() {
                        if let Some(var_d) = *maybe_var_d {
                            let local = (row as u64 / strides[primary_d]) % chunk_shape[primary_d];
                            let local_with_offset = local + var_offsets[var_d];
                            var_idx += local_with_offset * var_strides[var_d];
                        }
                    }
                    var_idx as usize
                })
                .collect();
            cols.push(data.take_indices(&indices).into_series(&name).into());
        }
    }

    Ok(DataFrame::new(cols).map_err(PyPolarsErr::from)?)
}

#[pyfunction]
#[pyo3(signature = (zarr_url, predicate, variables=None, max_concurrency=None, with_columns=None))]
pub fn scan_zarr_async(
    py: Python<'_>,
    zarr_url: String,
    predicate: &Bound<'_, PyAny>,
    variables: Option<Vec<String>>,
    max_concurrency: Option<usize>,
    with_columns: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    // Extract Expr under the GIL (and guard against panics during conversion).
    let expr = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let pyexpr: PyExpr = predicate.extract()?;
        Ok::<Expr, PyErr>(pyexpr.0.clone())
    }))
    .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("panic while converting predicate Expr"))??;

    let with_columns: Option<BTreeSet<String>> =
        with_columns.map(|v| v.into_iter().collect::<BTreeSet<_>>());

    let awaitable = future_into_py(py, async move {
        // Async open + async meta traversal.
        let opened_async = open_store_async(&zarr_url).map_err(to_py_err)?;
        let meta = load_dataset_meta_from_opened_async(&opened_async)
            .await
            .map_err(to_py_err)?;

        let vars = variables.unwrap_or_else(|| meta.data_vars.clone());
        if vars.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "no variables found/selected",
            ));
        }

        let primary_var = &vars[0];
        let primary_meta = meta.arrays.get(primary_var).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown primary variable")
        })?;
        let dims = if !primary_meta.dims.is_empty() {
            primary_meta.dims.clone()
        } else {
            meta.dims.clone()
        };

        // Open arrays once (async) for reading.
        let (primary, var_arrays, coord_arrays) =
            open_arrays_async(opened_async.store.clone(), &meta, &vars, &dims).await.map_err(to_py_err)?;

        // Compile chunk plan off-thread using the existing (sync) planner.
        // This keeps behavior identical to the current predicate pushdown planning.
        let zarr_url_plan = zarr_url.clone();
        let meta_plan = meta.clone();
        let expr_plan = expr.clone();
        let primary_var_plan = primary_var.to_string();
        let dims_for_fallback = dims.clone();
        let (plan, _stats) = tokio::task::spawn_blocking(move || -> Result<(ChunkPlan, crate::chunk_plan::PlannerStats), PyErr> {
            let opened_sync = open_store(&zarr_url_plan).map_err(to_py_err)?;
            match compile_expr_to_chunk_plan(
                &expr_plan,
                &meta_plan,
                opened_sync.store.clone(),
                &primary_var_plan,
            ) {
                Ok(x) => Ok(x),
                Err(_) => {
                    // Fall back to scanning all chunks if planning fails.
                    let arr = Array::open(
                        opened_sync.store.clone(),
                        &meta_plan.arrays[&primary_var_plan].path,
                    )
                    .map_err(to_py_err)?;
                    let grid_shape = arr.chunk_grid().grid_shape().to_vec();
                    let zero = vec![0u64; arr.dimensionality()];
                    let regular_chunk_shape = arr
                        .chunk_shape(&zero)
                        .map_err(to_py_err)?
                        .iter()
                        .map(|x| x.get())
                        .collect::<Vec<u64>>();
                    Ok((
                        ChunkPlan::all(dims_for_fallback.clone(), grid_shape, regular_chunk_shape),
                        crate::chunk_plan::PlannerStats { coord_reads: 0 },
                    ))
                }
            }
        })
        .await
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("chunk planner join error: {e}")))??;

        let indices: Vec<Vec<u64>> = plan.into_index_iter().collect();

        let max_conc = max_concurrency
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_MAX_CONCURRENCY);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_conc));

        let meta = Arc::new(meta);
        let dims = Arc::new(dims);
        let vars = Arc::new(vars);
        let var_arrays = Arc::new(var_arrays);
        let coord_arrays = Arc::new(coord_arrays);
        let with_columns = Arc::new(with_columns);

        let mut futs = FuturesUnordered::new();
        for idx in indices {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let primary = primary.clone();
            let meta = Arc::clone(&meta);
            let dims = Arc::clone(&dims);
            let vars = Arc::clone(&vars);
            let var_arrays = Arc::clone(&var_arrays);
            let coord_arrays = Arc::clone(&coord_arrays);
            let with_columns = Arc::clone(&with_columns);
            futs.push(async move {
                let _permit = permit;
                chunk_to_df(idx, primary, meta, dims, vars, var_arrays, coord_arrays, with_columns).await
            });
        }

        let mut out: Option<DataFrame> = None;
        while let Some(r) = futs.next().await {
            let df = r?;
            if let Some(acc) = &mut out {
                acc.vstack_mut(&df).map_err(PyPolarsErr::from)?;
            } else {
                out = Some(df);
            }
        }

        let df = out.unwrap_or_else(|| DataFrame::new(vec![]).unwrap());
        Python::attach(|py| Ok(PyDataFrame(df).into_pyobject(py)?.unbind()))
    })?;
    Ok(awaitable.unbind())
}


