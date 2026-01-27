use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::PyDataFrame;
use smallvec::SmallVec;
use zarrs::array::Array;

use crate::reader::{
    checked_chunk_len, compute_strides, compute_var_chunk_info, retrieve_1d_subset, retrieve_chunk,
    ColumnData,
};
use crate::{IStr, IntoIStr};

use super::{to_py_err, ZarrSource};

impl ZarrSource {
    pub(super) fn next_impl(&mut self) -> PyResult<Option<PyDataFrame>> {
        if self.n_rows_left == 0 || self.done {
            return Ok(None);
        }

        // Open arrays (cheap-ish metadata, but still avoid reopening per element).
        let primary_path = self.meta.arrays[&self.vars[0]].path.clone();
        let primary = Array::open(self.store.clone(), primary_path.as_ref()).map_err(to_py_err)?;
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
                if !self.should_emit(v.as_ref()) {
                    continue;
                }
                let path = &self.meta.arrays[v].path;
                let arr = Array::open(self.store.clone(), path.as_ref()).map_err(to_py_err)?;
                let data = retrieve_chunk(&arr, &[]).map_err(to_py_err)?;
                cols.push(data.slice(0, 1).into_series(v.as_ref()).into());
            }
            let df = DataFrame::new(1, cols).map_err(PyPolarsErr::from)?;
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
                        Array::open(self.store.clone(), coord_meta.path.as_ref()).map_err(to_py_err)?;
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
            let mut var_chunks: Vec<(IStr, ColumnData, SmallVec<[IStr; 4]>, Vec<u64>, Vec<u64>)> =
                Vec::new();
            for v in &self.vars {
                if !self.should_emit(v.as_ref()) {
                    continue;
                }
                let var_meta = &self.meta.arrays[v];
                let path = &var_meta.path;
                let arr = Array::open(self.store.clone(), path.as_ref()).map_err(to_py_err)?;

                // For variables with different dimensionality or chunk sizes,
                // compute the correct chunk indices and offsets.
                let (var_chunk_indices, var_offsets) = if var_meta.dims.len() == self.dims.len()
                    && var_meta.dims.iter().zip(self.dims.iter()).all(|(a, b)| a == b)
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
                    )
                    .map_err(to_py_err)?
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
                var_chunks.push((
                    v.clone(),
                    data,
                    var_meta.dims.clone(),
                    var_chunk_shape,
                    var_offsets,
                ));
            }

            // Build output columns.
            let mut cols: Vec<Column> = Vec::new();

            // Dim/coord columns.
            for (d, dim_name) in self.dims.iter().enumerate() {
                if !self.should_emit(dim_name.as_ref()) {
                    continue;
                }

                // Check for time encoding on this coordinate
                let time_encoding = self
                    .meta
                    .arrays
                    .get(dim_name)
                    .and_then(|m| m.time_encoding.as_ref());

                if let Some(te) = time_encoding {
                    // Build datetime or duration column
                    let mut out_i64: Vec<i64> = Vec::with_capacity(keep.len());
                    for &r in &keep {
                        let row = start + r;
                        let local = (row as u64 / strides[d]) % chunk_shape[d];
                        let raw_value = if let Some(coord) = &coord_slices[d] {
                            coord.get_i64(local as usize)
                                .unwrap_or((origin[d] + local) as i64)
                        } else {
                            (origin[d] + local) as i64
                        };
                        // Convert to nanoseconds
                        let ns = if te.is_duration {
                            raw_value.saturating_mul(te.unit_ns)
                        } else {
                            raw_value
                                .saturating_mul(te.unit_ns)
                                .saturating_add(te.epoch_ns)
                        };
                        out_i64.push(ns);
                    }

                    let dim_name_str: &str = dim_name.as_ref();
                    let series = if te.is_duration {
                        Series::new(dim_name_str.into(), &out_i64)
                            .cast(&polars::prelude::DataType::Duration(
                                polars::prelude::TimeUnit::Nanoseconds,
                            ))
                            .unwrap_or_else(|_| Series::new(dim_name_str.into(), out_i64))
                    } else {
                        Series::new(dim_name_str.into(), &out_i64)
                            .cast(&polars::prelude::DataType::Datetime(
                                polars::prelude::TimeUnit::Nanoseconds,
                                None,
                            ))
                            .unwrap_or_else(|_| Series::new(dim_name_str.into(), out_i64))
                    };
                    cols.push(series.into());
                } else if let Some(coord) = &coord_slices[d]
                    && coord.is_float()
                {
                    let mut out_f64: Vec<f64> = Vec::with_capacity(keep.len());
                    for &r in &keep {
                        let row = start + r;
                        let local = (row as u64 / strides[d]) % chunk_shape[d];
                        out_f64.push(coord.get_f64(local as usize).unwrap());
                    }
                    cols.push(Series::new(<crate::IStr as AsRef<str>>::as_ref(dim_name).into(), out_f64).into());
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
                    cols.push(Series::new(<crate::IStr as AsRef<str>>::as_ref(dim_name).into(), out_i64).into());
                }
            }

            // Variable columns.
            // For variables with the same dimensionality as the primary, we can slice directly.
            // For variables with fewer dimensions or different chunk sizes, we need to compute
            // the correct index by projecting from the primary iteration space.
            for (name, data, var_dims, var_chunk_shape, var_offsets) in var_chunks {
                let var_dims_vec: Vec<IStr> = var_dims.iter().cloned().collect();
                if var_dims_vec.len() == self.dims.len()
                    && var_dims_vec == self.dims
                    && var_offsets.iter().all(|&o| o == 0)
                {
                    // Same dimensionality and order with zero offsets - direct slice
                    let sliced = data.slice(start, len);
                    cols.push(sliced.take_indices(&keep).into_series(name.as_ref()).into());
                } else {
                    // Different dimensionality or non-zero offsets - need to map indices
                    // Build a mapping from primary dim index to variable dim index
                    let dim_mapping: Vec<Option<usize>> = self
                        .dims
                        .iter()
                        .map(|pd| var_dims.iter().position(|vd| vd == pd))
                        .collect();

                    let var_strides = compute_strides(&var_chunk_shape);

                    // For each row, compute the index in the variable's chunk
                    let indices: Vec<usize> = keep
                        .iter()
                        .map(|&r| {
                            let row = start + r;
                            let mut var_idx: u64 = 0;
                            for (primary_d, maybe_var_d) in dim_mapping.iter().enumerate() {
                                if let Some(var_d) = *maybe_var_d {
                                    // Get local position in primary dimension
                                    let local =
                                        (row as u64 / strides[primary_d]) % chunk_shape[primary_d];
                                    // Add the offset within the variable's chunk and contribution to index
                                    let local_with_offset = local + var_offsets[var_d];
                                    var_idx += local_with_offset * var_strides[var_d];
                                }
                            }
                            var_idx as usize
                        })
                        .collect();

                    cols.push(data.take_indices(&indices).into_series(name.as_ref()).into());
                }
            }

            // Build struct columns for child groups (hierarchical stores only)
            if self.is_hierarchical {
                if let Some(ref unified) = self.unified_meta {
                    for (child_name, child_node) in &unified.root.children {
                        let child_name_str: &str = child_name.as_ref();
                        
                        // Check if this group should be emitted
                        let should_emit = self.with_columns.as_ref().map_or(true, |cols| {
                            let child_name_istr: &str = child_name.as_ref();
                            cols.contains(&child_name_istr.istr())
                        });
                        if !should_emit {
                            continue;
                        }

                        let mut field_series: Vec<Series> = Vec::new();

                        // Load each data variable in this child group
                        for var_name in &child_node.data_vars {
                            let var_name_str: &str = var_name.as_ref();
                            
                            // Build the path to find this variable
                            let child_path_str: &str = child_node.path.as_ref();
                            let full_path = format!(
                                "{}{}",
                                child_path_str,
                                if child_path_str.ends_with('/') || var_name_str.starts_with('/') {
                                    var_name_str.to_string()
                                } else {
                                    format!("/{}", var_name_str)
                                }
                            );

                            // Try to find and load the array
                            if let Some(arr_meta) = unified.path_to_array.get(&full_path.istr()) {
                                let arr = Array::open(self.store.clone(), arr_meta.path.as_ref())
                                    .map_err(to_py_err)?;

                                // Compute chunk indices for this variable
                                let var_dims: Vec<IStr> = arr_meta.dims.iter().cloned().collect();
                                let (var_chunk_indices, var_offsets) = if var_dims == self.dims {
                                    (idx.clone(), vec![0; self.dims.len()])
                                } else {
                                    compute_var_chunk_info(
                                        idx,
                                        &chunk_shape,
                                        &self.dims,
                                        &var_dims,
                                        &arr,
                                    )
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

                                let data = retrieve_chunk(&arr, &var_chunk_indices).map_err(to_py_err)?;

                                // Build series with broadcasting if needed
                                let series = if var_dims == self.dims && var_offsets.iter().all(|&o| o == 0) {
                                    data.slice(start, len).take_indices(&keep).into_series(var_name_str)
                                } else {
                                    let dim_mapping: Vec<Option<usize>> = self
                                        .dims
                                        .iter()
                                        .map(|pd| var_dims.iter().position(|vd| vd == pd))
                                        .collect();
                                    let var_strides = compute_strides(&var_chunk_shape);

                                    let indices: Vec<usize> = keep
                                        .iter()
                                        .map(|&r| {
                                            let row = start + r;
                                            let mut var_idx: u64 = 0;
                                            for (primary_d, maybe_var_d) in dim_mapping.iter().enumerate() {
                                                if let Some(var_d) = *maybe_var_d {
                                                    let local = (row as u64 / strides[primary_d]) % chunk_shape[primary_d];
                                                    let local_with_offset = local + var_offsets.get(var_d).copied().unwrap_or(0);
                                                    if var_d < var_strides.len() {
                                                        var_idx += local_with_offset * var_strides[var_d];
                                                    }
                                                }
                                            }
                                            var_idx as usize
                                        })
                                        .collect();

                                    data.take_indices(&indices).into_series(var_name_str)
                                };

                                field_series.push(series);
                            } else {
                                // Variable not found - create null series
                                let null_series = Series::new_null(var_name_str.into(), keep.len());
                                field_series.push(null_series);
                            }
                        }

                        // Create struct column from field series
                        if !field_series.is_empty() {
                            let struct_chunked = StructChunked::from_series(
                                child_name_str.into(),
                                keep.len(),  // Row length, not field count
                                field_series.iter(),
                            )
                            .map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
                            })?;
                            cols.push(struct_chunked.into_series().into());
                        }
                    }
                }
            }

            let df = DataFrame::new(keep.len(), cols).map_err(PyPolarsErr::from)?;
            self.chunk_offset += len;
            self.n_rows_left = self.n_rows_left.saturating_sub(keep.len());
            return Ok(Some(PyDataFrame(df)));
        }
    }
}

