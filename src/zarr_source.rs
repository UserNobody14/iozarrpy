use std::collections::BTreeSet;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PyExpr, PySchema};
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;

use crate::chunk_plan::{compile_expr_to_chunk_plan, ChunkIndexIter, ChunkPlan};
use crate::zarr_meta::{load_dataset_meta_from_opened, ZarrDatasetMeta};
use crate::zarr_store::open_store;

const DEFAULT_BATCH_SIZE: usize = 10_000;
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
    // Guard against pathological coord arrays / unchunked giant reads which would OOM-abort.
    if len as u128 > max_chunk_elems() as u128 {
        return Err("refusing to allocate extremely large coordinate subset; set RAINBEAR_MAX_CHUNK_ELEMS to override".to_string());
    }
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

#[pymethods]
impl ZarrSource {
    #[new]
    #[pyo3(signature = (zarr_url, batch_size, n_rows, variables=None, max_chunks_to_read=None))]
    fn new(
        zarr_url: String,
        batch_size: Option<usize>,
        n_rows: Option<usize>,
        variables: Option<Vec<String>>,
        max_chunks_to_read: Option<usize>,
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

        let primary_grid_shape = primary.chunk_grid().grid_shape().to_vec();
        let zero = vec![0u64; primary.dimensionality()];
        let regular_chunk_shape = primary
            .chunk_shape(&zero)
            .map_err(to_py_err)?
            .iter()
            .map(|x| x.get())
            .collect::<Vec<u64>>();

        // Polars may pass 0 to mean "unspecified"; interpret it as the default.
        let n_rows_left = match n_rows {
            None | Some(0) => usize::MAX,
            Some(n) => n,
        };
        let batch_size = match batch_size {
            None | Some(0) => DEFAULT_BATCH_SIZE,
            Some(n) => n,
        };

        let dims = meta
            .arrays
            .get(&vars[0])
            .map(|m| m.dims.clone())
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| (0..primary.dimensionality()).map(|i| format!("dim_{i}")).collect());
        let chunk_iter =
            ChunkPlan::all(dims.clone(), primary_grid_shape.clone(), regular_chunk_shape)
                .into_index_iter();

        Ok(Self {
            meta,
            store,
            dims,
            vars,
            batch_size,
            n_rows_left,
            predicate: None,
            with_columns: None,
            primary_grid_shape,
            chunk_iter,
            current_chunk_indices: None,
            chunk_offset: 0,
            done: primary.dimensionality() == 0 && false,
            chunks_left: max_chunks_to_read,
        })
    }

    fn consume_chunk_budget(&mut self) -> PyResult<()> {
        if let Some(left) = self.chunks_left {
            if left == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "max_chunks_to_read limit hit",
                ));
            }
            self.chunks_left = Some(left - 1);
        }
        Ok(())
    }

    fn schema(&self) -> PySchema {
        let schema = self.meta.tidy_schema(Some(&self.vars));
        PySchema(Arc::new(schema))
    }

    fn try_set_predicate(&mut self, predicate: &Bound<'_, PyAny>) -> PyResult<()> {
        // IMPORTANT: Taking `PyExpr` directly in the signature can abort the whole
        // Python process if the Python->Rust Expr conversion panics.
        //
        // By accepting `PyAny` and extracting inside a `catch_unwind`, we can turn
        // those panics into a normal Python exception which the Python wrapper can
        // safely ignore (disabling pushdown but keeping correctness).
        let expr = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let pyexpr: PyExpr = predicate.extract()?;
            Ok::<Expr, PyErr>(pyexpr.0.clone())
        }))
        .map_err(|e| panic_to_py_err(e))??;

        // Compile Expr -> candidate-chunk plan (no full-grid scans).
        let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compile_expr_to_chunk_plan(&expr, &self.meta, self.store.clone(), &self.vars[0])
        }))
        .map_err(|e| panic_to_py_err(e))?;

        match compiled {
            Ok((plan, _stats)) => {
                self.primary_grid_shape = plan.grid_shape().to_vec();
                self.chunk_iter = plan.into_index_iter();
                self.current_chunk_indices = None;
                self.chunk_offset = 0;
            }
            Err(_) => {
                // Fall back to scanning all chunks if planning fails.
                let primary_path = self.meta.arrays[&self.vars[0]].path.clone();
                let primary = Array::open(self.store.clone(), &primary_path).map_err(to_py_err)?;
                let grid_shape = primary.chunk_grid().grid_shape().to_vec();
                let zero = vec![0u64; primary.dimensionality()];
                let regular_chunk_shape = primary
                    .chunk_shape(&zero)
                    .map_err(to_py_err)?
                    .iter()
                    .map(|x| x.get())
                    .collect::<Vec<u64>>();
                self.primary_grid_shape = grid_shape.clone();
                self.chunk_iter = ChunkPlan::all(self.dims.clone(), grid_shape, regular_chunk_shape).into_index_iter();
                self.current_chunk_indices = None;
                self.chunk_offset = 0;
            }
        }

        self.predicate = Some(expr);
        Ok(())
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
            self.consume_chunk_budget()?;

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

        // Advance to the next candidate chunk (planned from the predicate) as needed.
        loop {
            if self.current_chunk_indices.is_none() {
                if let Some(next_idx) = self.chunk_iter.next() {
                    self.consume_chunk_budget()?;
                    self.current_chunk_indices = Some(next_idx);
                    self.chunk_offset = 0;
                } else {
                    self.done = true;
                    return Ok(None);
                }
            }
            let idx = self.current_chunk_indices.as_ref().unwrap();
            let chunk_shape_nz = primary.chunk_shape(idx).map_err(to_py_err)?;
            let chunk_shape: Vec<u64> = chunk_shape_nz.iter().map(|x| x.get()).collect();
            let chunk_len: usize = checked_chunk_len(&chunk_shape)?;
            if self.chunk_offset >= chunk_len {
                self.current_chunk_indices = None;
                self.chunk_offset = 0;
                continue;
            }
            break;
        }

        // We may need to skip “empty” batches after trimming out-of-bounds rows (e.g. sharded edges).
        loop {
            let idx = self.current_chunk_indices.as_ref().unwrap();
            let chunk_shape_nz = primary.chunk_shape(idx).map_err(to_py_err)?;
            let chunk_shape: Vec<u64> = chunk_shape_nz.iter().map(|x| x.get()).collect();
            let chunk_len: usize = chunk_shape.iter().product::<u64>() as usize;

            let start = self.chunk_offset;
            let max_len = std::cmp::min(self.batch_size, self.n_rows_left);
            let len = std::cmp::min(chunk_len - start, max_len);

            // Origin for this chunk for dims/coords.
            let origin = primary
                .chunk_grid()
                .chunk_origin(idx)
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
                    self.current_chunk_indices = None;
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
                    let coord = retrieve_1d_subset(&coord_arr, dim_start, dim_len).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "error reading coord dim={dim_name} path={} start={dim_start} len={dim_len}: {e}",
                            coord_meta.path
                        ))
                    })?;
                    coord_slices.push(Some(coord));
                } else {
                    coord_slices.push(None);
                }
            }

            // Load chunk data for each requested var once.
            // Store the variable metadata along with the data for proper indexing.
            // For variables with different chunk sizes, we also store the offset within the var chunk.
            let mut var_chunks: Vec<(String, ColumnData, Vec<String>, Vec<u64>, Vec<u64>)> = Vec::new();
            for v in &self.vars {
                if !self.should_emit(v) {
                    continue;
                }
                let var_meta = &self.meta.arrays[v];
                let path = &var_meta.path;
                let arr = Array::open(self.store.clone(), path).map_err(to_py_err)?;
                
                // For variables with different dimensionality or chunk sizes,
                // compute the correct chunk indices and offsets.
                let (var_chunk_indices, var_offsets) = if var_meta.dims.len() == self.dims.len()
                    && var_meta.dims == self.dims
                {
                    // Same dims - use primary chunk indices directly
                    (idx.clone(), vec![0; self.dims.len()])
                } else {
                    // Different dims or chunk sizes - compute proper mapping
                    compute_var_chunk_info(
                        idx,
                        &chunk_shape,
                        &self.dims,
                        &var_meta.dims,
                        &arr,
                    ).map_err(to_py_err)?
                };
                
                // Get the chunk shape for this variable's chunk
                let var_chunk_shape: Vec<u64> = if var_chunk_indices.is_empty() {
                    vec![] // scalar
                } else {
                    arr.chunk_shape(&var_chunk_indices)
                        .map_err(to_py_err)?
                        .iter()
                        .map(|x| x.get())
                        .collect()
                };
                
                // Prevent OOM-abort on absurdly large chunks.
                if !var_chunk_shape.is_empty() {
                    let _ = checked_chunk_len(&var_chunk_shape)?;
                }
                let data = retrieve_chunk(&arr, &var_chunk_indices).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "error reading var={v} path={path} chunk_indices={var_chunk_indices:?} chunk_shape={var_chunk_shape:?} offsets={var_offsets:?}: {e}"
                    ))
                })?;
                var_chunks.push((v.clone(), data, var_meta.dims.clone(), var_chunk_shape, var_offsets));
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
            // For variables with the same dimensionality as the primary, we can slice directly.
            // For variables with fewer dimensions or different chunk sizes, we need to compute
            // the correct index by projecting from the primary iteration space.
            for (name, data, var_dims, var_chunk_shape, var_offsets) in var_chunks {
                if var_dims.len() == self.dims.len() && var_dims == self.dims && var_offsets.iter().all(|&o| o == 0) {
                    // Same dimensionality and order with zero offsets - direct slice
                    let sliced = data.slice(start, len);
                    cols.push(sliced.take_indices(&keep).into_series(&name).into());
                } else {
                    // Different dimensionality or non-zero offsets - need to map indices
                    // Build a mapping from primary dim index to variable dim index
                    let dim_mapping: Vec<Option<usize>> = self.dims.iter()
                        .map(|pd| var_dims.iter().position(|vd| vd == pd))
                        .collect();
                    
                    let var_strides = compute_strides(&var_chunk_shape);
                    
                    // For each row, compute the index in the variable's chunk
                    let indices: Vec<usize> = keep.iter().map(|&r| {
                        let row = start + r;
                        let mut var_idx: u64 = 0;
                        for (primary_d, maybe_var_d) in dim_mapping.iter().enumerate() {
                            if let Some(var_d) = *maybe_var_d {
                                // Get local position in primary dimension
                                let local = (row as u64 / strides[primary_d]) % chunk_shape[primary_d];
                                // Add the offset within the variable's chunk and contribution to index
                                let local_with_offset = local + var_offsets[var_d];
                                var_idx += local_with_offset * var_strides[var_d];
                            }
                        }
                        var_idx as usize
                    }).collect();
                    
                    cols.push(data.take_indices(&indices).into_series(&name).into());
                }
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
            // The chunk plan is used for chunk pruning, and the Python layer handles
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

/// Compute the variable chunk indices and within-chunk offset for a variable
/// with potentially different chunk sizes than the primary array.
/// 
/// Returns (var_chunk_indices, offsets) where:
/// - var_chunk_indices: the chunk indices to use when reading the variable's chunk
/// - offsets: the offset within each dimension of the variable's chunk to start reading
fn compute_var_chunk_info(
    primary_chunk_indices: &[u64],
    primary_chunk_shape: &[u64],
    primary_dims: &[String],
    var_dims: &[String],
    var_array: &Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let _var_grid_shape = var_array.chunk_grid().grid_shape();
    
    let mut var_chunk_indices = Vec::with_capacity(var_dims.len());
    let mut offsets = Vec::with_capacity(var_dims.len());
    
    for var_dim in var_dims {
        // Find this dimension in the primary array
        if let Some(primary_d) = primary_dims.iter().position(|pd| pd == var_dim) {
            // Calculate global position for start of primary chunk in this dimension
            let primary_chunk_idx = primary_chunk_indices[primary_d];
            let primary_chunk_size = primary_chunk_shape[primary_d];
            let global_start = primary_chunk_idx * primary_chunk_size;
            
            // Find which variable chunk contains this global position
            // Get the variable's chunk shape for this dimension
            let var_dim_idx = var_dims.iter().position(|vd| vd == var_dim).unwrap();
            
            // We need the chunk shape to compute which chunk index this falls into
            // Use chunk index 0 to get the regular chunk shape
            let zero_indices: Vec<u64> = vec![0; var_dims.len()];
            let var_regular_chunk_shape = var_array
                .chunk_shape(&zero_indices)
                .map_err(|e| e.to_string())?;
            let var_chunk_size = var_regular_chunk_shape[var_dim_idx].get();
            
            // Compute the variable chunk index and offset
            let var_chunk_idx = global_start / var_chunk_size;
            let offset = global_start % var_chunk_size;
            
            var_chunk_indices.push(var_chunk_idx);
            offsets.push(offset);
        } else {
            // This dimension doesn't exist in primary - shouldn't happen for well-formed data
            return Err(format!("variable dimension {} not found in primary dims", var_dim));
        }
    }
    
    Ok((var_chunk_indices, offsets))
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

