use std::collections::BTreeSet;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PyExpr, PySchema};
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;

use crate::zarr_meta::{load_dataset_meta_from_opened, TimeEncoding, ZarrDatasetMeta};
use crate::zarr_store::open_store;

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

    fn slice(&self, start: usize, len: usize) -> ColumnData {
        let end = start + len;
        match self {
            ColumnData::Bool(v) => ColumnData::Bool(v[start..end].to_vec()),
            ColumnData::I8(v) => ColumnData::I8(v[start..end].to_vec()),
            ColumnData::I16(v) => ColumnData::I16(v[start..end].to_vec()),
            ColumnData::I32(v) => ColumnData::I32(v[start..end].to_vec()),
            ColumnData::I64(v) => ColumnData::I64(v[start..end].to_vec()),
            ColumnData::U8(v) => ColumnData::U8(v[start..end].to_vec()),
            ColumnData::U16(v) => ColumnData::U16(v[start..end].to_vec()),
            ColumnData::U32(v) => ColumnData::U32(v[start..end].to_vec()),
            ColumnData::U64(v) => ColumnData::U64(v[start..end].to_vec()),
            ColumnData::F32(v) => ColumnData::F32(v[start..end].to_vec()),
            ColumnData::F64(v) => ColumnData::F64(v[start..end].to_vec()),
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

fn retrieve_chunk(array: &Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>, chunk: &[u64]) -> Result<ColumnData, String> {
    let id = array.data_type().identifier();
    match id {
        "bool" => Ok(ColumnData::Bool(array.retrieve_chunk::<Vec<bool>>(chunk).map_err(to_string_err)?)),
        "int8" => Ok(ColumnData::I8(array.retrieve_chunk::<Vec<i8>>(chunk).map_err(to_string_err)?)),
        "int16" => Ok(ColumnData::I16(array.retrieve_chunk::<Vec<i16>>(chunk).map_err(to_string_err)?)),
        "int32" => Ok(ColumnData::I32(array.retrieve_chunk::<Vec<i32>>(chunk).map_err(to_string_err)?)),
        "int64" => Ok(ColumnData::I64(array.retrieve_chunk::<Vec<i64>>(chunk).map_err(to_string_err)?)),
        "uint8" => Ok(ColumnData::U8(array.retrieve_chunk::<Vec<u8>>(chunk).map_err(to_string_err)?)),
        "uint16" => Ok(ColumnData::U16(array.retrieve_chunk::<Vec<u16>>(chunk).map_err(to_string_err)?)),
        "uint32" => Ok(ColumnData::U32(array.retrieve_chunk::<Vec<u32>>(chunk).map_err(to_string_err)?)),
        "uint64" => Ok(ColumnData::U64(array.retrieve_chunk::<Vec<u64>>(chunk).map_err(to_string_err)?)),
        "float32" => Ok(ColumnData::F32(array.retrieve_chunk::<Vec<f32>>(chunk).map_err(to_string_err)?)),
        "float64" => Ok(ColumnData::F64(array.retrieve_chunk::<Vec<f64>>(chunk).map_err(to_string_err)?)),
        other => Err(format!("unsupported zarr dtype: {other}")),
    }
}

fn retrieve_1d_subset(
    array: &Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
    start: u64,
    len: u64,
) -> Result<ColumnData, String> {
    let subset = ArraySubset::new_with_ranges(&[start..(start + len)]);
    let id = array.data_type().identifier();
    match id {
        "bool" => Ok(ColumnData::Bool(
            array
                .retrieve_array_subset::<Vec<bool>>(&subset)
                .map_err(to_string_err)?,
        )),
        "int8" => Ok(ColumnData::I8(
            array
                .retrieve_array_subset::<Vec<i8>>(&subset)
                .map_err(to_string_err)?,
        )),
        "int16" => Ok(ColumnData::I16(
            array
                .retrieve_array_subset::<Vec<i16>>(&subset)
                .map_err(to_string_err)?,
        )),
        "int32" => Ok(ColumnData::I32(
            array
                .retrieve_array_subset::<Vec<i32>>(&subset)
                .map_err(to_string_err)?,
        )),
        "int64" => Ok(ColumnData::I64(
            array
                .retrieve_array_subset::<Vec<i64>>(&subset)
                .map_err(to_string_err)?,
        )),
        "uint8" => Ok(ColumnData::U8(
            array
                .retrieve_array_subset::<Vec<u8>>(&subset)
                .map_err(to_string_err)?,
        )),
        "uint16" => Ok(ColumnData::U16(
            array
                .retrieve_array_subset::<Vec<u16>>(&subset)
                .map_err(to_string_err)?,
        )),
        "uint32" => Ok(ColumnData::U32(
            array
                .retrieve_array_subset::<Vec<u32>>(&subset)
                .map_err(to_string_err)?,
        )),
        "uint64" => Ok(ColumnData::U64(
            array
                .retrieve_array_subset::<Vec<u64>>(&subset)
                .map_err(to_string_err)?,
        )),
        "float32" => Ok(ColumnData::F32(
            array
                .retrieve_array_subset::<Vec<f32>>(&subset)
                .map_err(to_string_err)?,
        )),
        "float64" => Ok(ColumnData::F64(
            array
                .retrieve_array_subset::<Vec<f64>>(&subset)
                .map_err(to_string_err)?,
        )),
        other => Err(format!("unsupported zarr dtype: {other}")),
    }
}

fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

#[pyclass]
pub struct ZarrSource {
    meta: ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,

    dims: Vec<String>,
    vars: Vec<String>,

    batch_size: usize,
    n_rows_left: usize,

    predicate: Option<Expr>,
    constraints: Option<Vec<(String, DimConstraint)>>,
    with_columns: Option<BTreeSet<String>>,

    // Iteration state
    grid_shape: Vec<u64>,
    chunk_indices: Vec<u64>,
    chunk_offset: usize,
    done: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct DimConstraint {
    eq: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

#[pymethods]
impl ZarrSource {
    #[new]
    #[pyo3(signature = (zarr_url, batch_size, n_rows, variables=None))]
    fn new(
        zarr_url: String,
        batch_size: Option<usize>,
        n_rows: Option<usize>,
        variables: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let opened = open_store(&zarr_url).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let meta = load_dataset_meta_from_opened(&opened)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let store = opened.store.clone();

        let vars = if let Some(v) = variables {
            v
        } else {
            meta.data_vars.clone()
        };

        if vars.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "no variables found/selected",
            ));
        }

        // Primary var defines the chunk iteration.
        let primary_path = meta
            .arrays
            .get(&vars[0])
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("unknown variable"))?
            .path
            .clone();
        let primary = Array::open(store.clone(), &primary_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })?;

        let grid_shape = primary.chunk_grid().grid_shape().to_vec();
        let chunk_indices = vec![0; primary.dimensionality()];

        let n_rows_left = n_rows.unwrap_or(usize::MAX);
        let batch_size = batch_size.unwrap_or(10_000);

        let dims = meta
            .arrays
            .get(&vars[0])
            .map(|m| m.dims.clone())
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| (0..primary.dimensionality()).map(|i| format!("dim_{i}")).collect());

        Ok(Self {
            meta,
            store,
            dims,
            vars,
            batch_size,
            n_rows_left,
            predicate: None,
            constraints: None,
            with_columns: None,
            grid_shape,
            chunk_indices,
            chunk_offset: 0,
            done: primary.dimensionality() == 0 && false,
        })
    }

    fn schema(&self) -> PySchema {
        let schema = self.meta.tidy_schema(Some(&self.vars));
        PySchema(Arc::new(schema))
    }

    fn try_set_predicate(&mut self, predicate: PyExpr) -> PyResult<()> {
        // The PyExpr has already been extracted/deserialized at this point.
        // Clone the expression to avoid any lifetime issues.
        let expr = predicate.0.clone();
        
        // Catch any panics during constraint compilation
        let meta_ref = &self.meta;
        let constraints = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compile_dim_constraints(&expr, meta_ref)
        }));
        
        match constraints {
            Ok(c) => {
                self.constraints = c;
                self.predicate = Some(expr);
                Ok(())
            }
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic in constraint compilation".to_string()
                };
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            }
        }
    }

    fn set_with_columns(&mut self, columns: Vec<String>) {
        self.with_columns = Some(columns.into_iter().collect());
    }

    fn next(&mut self) -> PyResult<Option<PyDataFrame>> {
        if self.n_rows_left == 0 || self.done {
            return Ok(None);
        }

        // Open arrays (cheap-ish metadata, but still avoid reopening per element).
        let primary_path = self.meta.arrays[&self.vars[0]].path.clone();
        let primary = Array::open(self.store.clone(), &primary_path).map_err(to_py_err)?;
        let array_shape = primary.shape().to_vec();

        // Handle scalar arrays as a single “chunk”
        if primary.dimensionality() == 0 {
            // For now: scalar => one row
            if self.n_rows_left == 0 {
                return Ok(None);
            }

            let mut cols: Vec<Column> = Vec::new();
            for v in &self.vars {
                if !self.should_emit(v) {
                    continue;
                }
                let path = &self.meta.arrays[v].path;
                let arr = Array::open(self.store.clone(), path).map_err(to_py_err)?;
                let data = retrieve_chunk(&arr, &[]).map_err(to_py_err)?;
                cols.push(data.slice(0, 1).into_series(v).into());
            }
            let df = DataFrame::new(cols).map_err(PyPolarsErr::from)?;
            self.n_rows_left = self.n_rows_left.saturating_sub(1);

            self.done = true;
            return Ok(Some(PyDataFrame(df)));
        }

        // Ensure we are on a chunk that satisfies constraints (chunk pruning).
        loop {
            let chunk_shape_nz = primary.chunk_shape(&self.chunk_indices).map_err(to_py_err)?;
            let chunk_shape: Vec<u64> = chunk_shape_nz.iter().map(|x| x.get()).collect();
            let chunk_len: usize = chunk_shape.iter().product::<u64>() as usize;

            if self.chunk_offset >= chunk_len {
                self.chunk_offset = 0;
                if !advance_chunk_indices(&mut self.chunk_indices, &self.grid_shape) {
                    self.done = true;
                    return Ok(None);
                }
                continue;
            }

            // Evaluate chunk-level constraints at chunk boundaries only.
            if self.chunk_offset == 0 {
                if let Some(constraints) = &self.constraints {
                    let origin = primary
                        .chunk_grid()
                        .chunk_origin(&self.chunk_indices)
                        .map_err(to_py_err)?
                        .unwrap_or_else(|| vec![0; chunk_shape.len()]);

                    if !chunk_satisfies_constraints(
                        self,
                        &origin,
                        &chunk_shape,
                        constraints,
                    )? {
                        self.chunk_offset = chunk_len; // force advance
                        continue;
                    }
                }
            }

            break;
        }

        // We may need to skip “empty” batches after trimming out-of-bounds rows (e.g. sharded edges).
        loop {
            let chunk_shape_nz = primary.chunk_shape(&self.chunk_indices).map_err(to_py_err)?;
            let chunk_shape: Vec<u64> = chunk_shape_nz.iter().map(|x| x.get()).collect();
            let chunk_len: usize = chunk_shape.iter().product::<u64>() as usize;

            let start = self.chunk_offset;
            let max_len = std::cmp::min(self.batch_size, self.n_rows_left);
            let len = std::cmp::min(chunk_len - start, max_len);

            // Origin for this chunk for dims/coords.
            let origin = primary
                .chunk_grid()
                .chunk_origin(&self.chunk_indices)
                .map_err(to_py_err)?
                .unwrap_or_else(|| vec![0; chunk_shape.len()]);

            let strides = compute_strides(&chunk_shape);

            // Identify in-bounds rows (important for sharded stores where shards extend beyond array shape).
            let mut keep: Vec<usize> = Vec::with_capacity(len);
            for r in 0..len {
                let row = start + r;
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
                    keep.push(r);
                }
            }

            // If everything in this slice is out-of-bounds, advance within the chunk and continue.
            if keep.is_empty() {
                self.chunk_offset += len;
                if self.chunk_offset >= chunk_len {
                    self.chunk_offset = chunk_len;
                }
                if self.chunk_offset >= chunk_len {
                    self.chunk_offset = 0;
                    if !advance_chunk_indices(&mut self.chunk_indices, &self.grid_shape) {
                        self.done = true;
                        return Ok(None);
                    }
                }
                continue;
            }

            // Preload coordinate slices for this chunk range (per dim).
            let mut coord_slices: Vec<Option<ColumnData>> = Vec::with_capacity(self.dims.len());
            for (d, dim_name) in self.dims.iter().enumerate() {
                if let Some(coord_meta) = self.meta.arrays.get(dim_name) {
                    let coord_arr =
                        Array::open(self.store.clone(), &coord_meta.path).map_err(to_py_err)?;
                    let dim_start = origin[d];
                    let dim_len = chunk_shape[d];
                    let coord =
                        retrieve_1d_subset(&coord_arr, dim_start, dim_len).map_err(to_py_err)?;
                    coord_slices.push(Some(coord));
                } else {
                    coord_slices.push(None);
                }
            }

            // Load chunk data for each requested var once.
            let mut var_chunks: Vec<(String, ColumnData)> = Vec::new();
            for v in &self.vars {
                if !self.should_emit(v) {
                    continue;
                }
                let path = &self.meta.arrays[v].path;
                let arr = Array::open(self.store.clone(), path).map_err(to_py_err)?;
                let data = retrieve_chunk(&arr, &self.chunk_indices).map_err(to_py_err)?;
                var_chunks.push((v.clone(), data));
            }

            // Build output columns.
            let mut cols: Vec<Column> = Vec::new();

            // Dim/coord columns.
            for (d, dim_name) in self.dims.iter().enumerate() {
                if !self.should_emit(dim_name) {
                    continue;
                }

                // Check for time encoding on this coordinate
                let time_encoding = self.meta.arrays.get(dim_name).and_then(|m| m.time_encoding.as_ref());

                if let Some(te) = time_encoding {
                    // Build datetime or duration column
                    let mut out_i64: Vec<i64> = Vec::with_capacity(keep.len());
                    for &r in &keep {
                        let row = start + r;
                        let local = (row as u64 / strides[d]) % chunk_shape[d];
                        let raw_value = if let Some(coord) = &coord_slices[d] {
                            coord.get_i64(local as usize).unwrap_or((origin[d] + local) as i64)
                        } else {
                            (origin[d] + local) as i64
                        };
                        // Convert to nanoseconds
                        let ns = if te.is_duration {
                            raw_value.saturating_mul(te.unit_ns)
                        } else {
                            raw_value.saturating_mul(te.unit_ns).saturating_add(te.epoch_ns)
                        };
                        out_i64.push(ns);
                    }

                    let series = if te.is_duration {
                        Series::new(dim_name.into(), &out_i64)
                            .cast(&polars::prelude::DataType::Duration(polars::prelude::TimeUnit::Nanoseconds))
                            .unwrap_or_else(|_| Series::new(dim_name.into(), out_i64))
                    } else {
                        Series::new(dim_name.into(), &out_i64)
                            .cast(&polars::prelude::DataType::Datetime(polars::prelude::TimeUnit::Nanoseconds, None))
                            .unwrap_or_else(|_| Series::new(dim_name.into(), out_i64))
                    };
                    cols.push(series.into());
                } else if let Some(coord) = &coord_slices[d] && coord.is_float() {
                    let mut out_f64: Vec<f64> = Vec::with_capacity(keep.len());
                    for &r in &keep {
                        let row = start + r;
                        let local = (row as u64 / strides[d]) % chunk_shape[d];
                        out_f64.push(coord.get_f64(local as usize).unwrap());
                    }
                    cols.push(Series::new(dim_name.into(), out_f64).into());
                } else {
                    let mut out_i64: Vec<i64> = Vec::with_capacity(keep.len());
                    for &r in &keep {
                        let row = start + r;
                        let local = (row as u64 / strides[d]) % chunk_shape[d];
                        if let Some(coord) = &coord_slices[d] {
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
            for (name, data) in var_chunks {
                let sliced = data.slice(start, len);
                cols.push(sliced.take_indices(&keep).into_series(&name).into());
            }

            let df = DataFrame::new(cols).map_err(PyPolarsErr::from)?;
            self.chunk_offset += len;
            self.n_rows_left = self.n_rows_left.saturating_sub(keep.len());

            // if let Some(predicate) = &self.predicate {
            //     df = df
            //         .lazy()
            //         .filter(predicate.clone())
            //         ._with_eager(true)
            //         .collect()
            //         .map_err(PyPolarsErr::from)?;
            // }
            // Note: We don't apply the predicate here anymore - this caused type inference 
            // issues with datetime/duration columns in the Polars lazy filter.
            // The constraints are used for chunk pruning, and the Python layer handles
            // final row filtering if needed.

            return Ok(Some(PyDataFrame(df)));
        }
    }
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

fn advance_chunk_indices(indices: &mut [u64], grid_shape: &[u64]) -> bool {
    if indices.is_empty() {
        return false;
    }
    for i in (0..indices.len()).rev() {
        indices[i] += 1;
        if indices[i] < grid_shape[i] {
            return true;
        }
        indices[i] = 0;
    }
    false
}

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

impl ZarrSource {
    fn should_emit(&self, name: &str) -> bool {
        self.with_columns
            .as_ref()
            .map(|s| s.contains(name))
            .unwrap_or(true)
    }
}

fn compile_dim_constraints(expr: &Expr, meta: &ZarrDatasetMeta) -> Option<Vec<(String, DimConstraint)>> {
    let mut out: Vec<(String, DimConstraint)> = Vec::new();
    if !collect_constraints(expr, meta, &mut out) {
        return None;
    }
    // Merge constraints per column.
    let mut merged: std::collections::BTreeMap<String, DimConstraint> = std::collections::BTreeMap::new();
    for (col, c) in out {
        let entry = merged.entry(col).or_default();
        entry.eq = entry.eq.or(c.eq);
        entry.min = match (entry.min, c.min) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (None, Some(b)) => Some(b),
            (Some(a), None) => Some(a),
            (None, None) => None,
        };
        entry.max = match (entry.max, c.max) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (None, Some(b)) => Some(b),
            (Some(a), None) => Some(a),
            (None, None) => None,
        };
    }
    Some(merged.into_iter().collect())
}

fn collect_constraints(expr: &Expr, meta: &ZarrDatasetMeta, out: &mut Vec<(String, DimConstraint)>) -> bool {
    match expr {
        Expr::Alias(inner, _) => collect_constraints(inner, meta, out),
        Expr::BinaryExpr { left, op, right } => {
            use polars::prelude::Operator;
            match op {
                Operator::And | Operator::LogicalAnd => {
                    collect_constraints(left, meta, out) && collect_constraints(right, meta, out)
                }
                Operator::Eq | Operator::GtEq | Operator::Gt | Operator::LtEq | Operator::Lt => {
                    if let Some((col, lit)) = col_lit(left, right).or_else(|| col_lit(right, left))
                    {
                        // Get time encoding for this column if it exists
                        let time_encoding = meta.arrays.get(&col).and_then(|a| a.time_encoding.as_ref());
                        
                        if let Some(v) = literal_to_f64(&lit, time_encoding) {
                            let mut c = DimConstraint::default();
                            match op {
                                Operator::Eq => c.eq = Some(v),
                                // We don't do strict bound nudging here; we still apply the full
                                // predicate after materializing batches, so correctness is preserved.
                                Operator::Gt => c.min = Some(v),
                                Operator::GtEq => c.min = Some(v),
                                Operator::Lt => c.max = Some(v),
                                Operator::LtEq => c.max = Some(v),
                                _ => {}
                            }
                            out.push((col, c));
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn col_lit(col_side: &Expr, lit_side: &Expr) -> Option<(String, LiteralValue)> {
    match (col_side, lit_side) {
        (Expr::Column(name), Expr::Literal(lit)) => Some((name.to_string(), lit.clone())),
        _ => None,
    }
}

fn literal_to_f64(lit: &LiteralValue, time_encoding: Option<&TimeEncoding>) -> Option<f64> {
    match lit {
        LiteralValue::Scalar(s) => match s.clone().into_value() {
            AnyValue::Int64(v) => Some(v as f64),
            AnyValue::Int32(v) => Some(v as f64),
            AnyValue::Int16(v) => Some(v as f64),
            AnyValue::Int8(v) => Some(v as f64),
            AnyValue::UInt64(v) => Some(v as f64),
            AnyValue::UInt32(v) => Some(v as f64),
            AnyValue::UInt16(v) => Some(v as f64),
            AnyValue::UInt8(v) => Some(v as f64),
            AnyValue::Float64(v) => Some(v),
            AnyValue::Float32(v) => Some(v as f64),
            // Handle datetime - convert time_unit to ns, then to raw encoding if available
            AnyValue::Datetime(value, time_unit, _) => {
                // Convert to nanoseconds based on time_unit
                let ns = match time_unit {
                    polars::prelude::TimeUnit::Nanoseconds => value,
                    polars::prelude::TimeUnit::Microseconds => value * 1_000,
                    polars::prelude::TimeUnit::Milliseconds => value * 1_000_000,
                };
                if let Some(enc) = time_encoding {
                    Some(enc.encode(ns) as f64)
                } else {
                    Some(ns as f64)
                }
            }
            AnyValue::Date(days) => {
                let ns = days as i64 * 86400 * 1_000_000_000;
                if let Some(enc) = time_encoding {
                    Some(enc.encode(ns) as f64)
                } else {
                    Some(ns as f64)
                }
            }
            // Handle duration - convert time_unit to ns, then to raw encoding if available
            AnyValue::Duration(value, time_unit) => {
                // Convert to nanoseconds based on time_unit
                let ns = match time_unit {
                    polars::prelude::TimeUnit::Nanoseconds => value,
                    polars::prelude::TimeUnit::Microseconds => value * 1_000,
                    polars::prelude::TimeUnit::Milliseconds => value * 1_000_000,
                };
                if let Some(enc) = time_encoding {
                    Some(enc.encode(ns) as f64)
                } else {
                    Some(ns as f64)
                }
            }
            _ => None,
        },
        _ => None,
    }
}

fn chunk_satisfies_constraints(
    src: &ZarrSource,
    origin: &[u64],
    chunk_shape: &[u64],
    constraints: &[(String, DimConstraint)],
) -> PyResult<bool> {
    for (col, c) in constraints {
        let Some(dim_idx) = src.dims.iter().position(|d| d == col) else {
            // constraint on non-dim column -> cannot prune safely
            return Ok(true);
        };

        let (min_v, max_v) = if let Some(coord_meta) = src.meta.arrays.get(col) {
            // Use coordinate values, if present (numeric only).
            let coord_arr = Array::open(src.store.clone(), &coord_meta.path).map_err(to_py_err)?;
            let start = origin[dim_idx];
            let len = chunk_shape[dim_idx];
            let data = retrieve_1d_subset(&coord_arr, start, len).map_err(to_py_err)?;
            let (mn, mx) = min_max_f64(&data).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "unsupported coord dtype for predicate pushdown",
                )
            })?;
            (mn, mx)
        } else {
            let mn = origin[dim_idx] as f64;
            let mx = (origin[dim_idx] + chunk_shape[dim_idx].saturating_sub(1)) as f64;
            (mn, mx)
        };

        if let Some(eq) = c.eq {
            if eq < min_v || eq > max_v {
                return Ok(false);
            }
        }
        if let Some(min) = c.min {
            if max_v < min {
                return Ok(false);
            }
        }
        if let Some(max) = c.max {
            if min_v > max {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn min_max_f64(data: &ColumnData) -> Option<(f64, f64)> {
    let mut mn: Option<f64> = None;
    let mut mx: Option<f64> = None;
    for i in 0..data.len() {
        let v = data.get_f64(i)?;
        mn = Some(mn.map(|x| x.min(v)).unwrap_or(v));
        mx = Some(mx.map(|x| x.max(v)).unwrap_or(v));
    }
    Some((mn?, mx?))
}
