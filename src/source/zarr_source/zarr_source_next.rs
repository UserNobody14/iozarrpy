//! Next-batch iteration logic for ZarrSource.
//!
//! This module handles the per-batch iteration for the Polars scan interface.
//! It reads from multiple chunk grids and joins them on dimension columns.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3_polars::error::PyPolarsErr;
use zarrs::array::Array;

use crate::reader::{
    ColumnData, compute_strides,
    retrieve_1d_subset, retrieve_chunk,
};
use crate::{IStr, IntoIStr};
use crate::meta::ZarrNode;

use super::{ZarrSource, to_py_err};

impl ZarrSource {
    pub(super) fn next_impl(
        &mut self,
    ) -> PyResult<Option<PyDataFrame>> {
        if self.n_rows_left == 0 || self.done {
            return Ok(None);
        }

        // Handle scalar case (0-dimensional arrays)
        if self.dims.is_empty() {
            return self.next_scalar_impl();
        }

        // Read from all grids and join
        let df = self.read_joined_batch()?;

        if let Some(df) = df {
            let n_rows = df.height();
            self.n_rows_left = self
                .n_rows_left
                .saturating_sub(n_rows);
            Ok(Some(PyDataFrame(df)))
        } else {
            self.done = true;
            Ok(None)
        }
    }

    /// Handle scalar (0-dimensional) arrays
    fn next_scalar_impl(
        &mut self,
    ) -> PyResult<Option<PyDataFrame>> {
        if self.n_rows_left == 0 {
            return Ok(None);
        }
        self.consume_chunk_budget()?;

        let mut cols: Vec<Column> = Vec::new();
        let vars = if self.is_hierarchical {
            self.unified_meta
                .as_ref()
                .map(|m| m.root.data_vars.as_slice())
                .unwrap_or(&[])
        } else {
            self.vars.as_slice()
        };
        for v in vars {
            if !self.should_emit(v.as_ref()) {
                continue;
            }
            let path = &self.meta.arrays[v].path;
            let arr = Array::open(
                self.store.clone(),
                path.as_ref(),
            )
            .map_err(to_py_err)?;
            let data =
                crate::reader::retrieve_chunk(
                    &arr,
                    &[],
                )
                .map_err(to_py_err)?;
            cols.push(
                data.slice(0, 1)
                    .into_series(v.as_ref())
                    .into(),
            );
        }
        let df = DataFrame::new(1, cols)
            .map_err(PyPolarsErr::from)?;
        self.n_rows_left =
            self.n_rows_left.saturating_sub(1);

        self.done = true;
        Ok(Some(PyDataFrame(df)))
    }

    /// Read a batch from the current grid, including ALL variables
    /// with consistent columns.
    ///
    /// This handles heterogeneous chunk grids by using the current grid
    /// to define the coordinate range and filling other grids with nulls.
    fn read_joined_batch(
        &mut self,
    ) -> PyResult<Option<DataFrame>> {
        // Find the first grid with pending work to define the batch coordinate range
        let primary_grid_idx = loop {
            if self.current_grid_idx
                >= self.grid_states.len()
            {
                return Ok(None);
            }

            let grid_state = &mut self
                .grid_states
                [self.current_grid_idx];
            if grid_state.advance().is_some() {
                break self.current_grid_idx;
            }

            // This grid is exhausted, move to next
            self.current_grid_idx += 1;
        };

        self.consume_chunk_budget()?;

        // Get the subset from the primary grid (defines coordinate range for this batch)
        let primary_subset = match &self
            .grid_states[primary_grid_idx]
            .current_subset
        {
            Some(s) => s.clone(),
            None => return Ok(None),
        };

        let subset_shape: Vec<u64> =
            primary_subset.shape().to_vec();
        let subset_origin: Vec<u64> =
            primary_subset.start().to_vec();
        let subset_len: usize =
            primary_subset.num_elements_usize();

        if subset_len == 0 {
            self.grid_states[primary_grid_idx]
                .finish_current();
            return self.read_joined_batch(); // Try next subset
        }

        // Calculate batch slice within subset
        let start = self.grid_states
            [primary_grid_idx]
            .subset_offset;
        let max_len = std::cmp::min(
            self.batch_size,
            self.n_rows_left,
        );
        let len = std::cmp::min(
            subset_len - start,
            max_len,
        );

        if len == 0 {
            self.grid_states[primary_grid_idx]
                .finish_current();
            return self.read_joined_batch(); // Try next subset
        }

        let strides =
            compute_strides(&subset_shape);

        // Read a full batch with ALL columns (consistent schema)
        let result = self.read_full_batch(
            primary_grid_idx,
            &primary_subset,
            &subset_shape,
            &subset_origin,
            &strides,
            start,
            len,
        )?;

        // Update the primary grid's offset
        self.grid_states[primary_grid_idx]
            .subset_offset += len;
        if self.grid_states[primary_grid_idx]
            .subset_offset
            >= subset_len
        {
            self.grid_states[primary_grid_idx]
                .finish_current();
        }

        // Build struct columns for hierarchical stores
        let result = if self.is_hierarchical {
            self.add_struct_columns_to_df(
                result,
                &primary_subset,
                &subset_shape,
                &subset_origin,
                &strides,
                start,
                len,
            )?
        } else {
            result
        };

        Ok(Some(result))
    }

    /// Read a batch that includes ALL requested variables (with consistent column order).
    /// Variables not in the current grid are filled with nulls.
    fn read_full_batch(
        &mut self,
        primary_grid_idx: usize,
        subset: &zarrs::array_subset::ArraySubset,
        subset_shape: &[u64],
        subset_origin: &[u64],
        strides: &[u64],
        start: usize,
        len: usize,
    ) -> PyResult<DataFrame> {
        let dim_names: std::collections::HashSet<
            IStr,
        > = self.dims.iter().cloned().collect();

        // Build dimension coordinate columns
        let mut cols: Vec<Column> = Vec::new();

        for (d, dim_name) in
            self.dims.iter().enumerate()
        {
            if !self
                .should_emit(dim_name.as_ref())
            {
                continue;
            }

            let dim_start = subset_origin
                .get(d)
                .copied()
                .unwrap_or(0);
            let dim_len = subset_shape
                .get(d)
                .copied()
                .unwrap_or(1);

            let coord_data =
                if let Some(coord_meta) =
                    self.meta.arrays.get(dim_name)
                {
                    let coord_arr = Array::open(
                        self.store.clone(),
                        coord_meta.path.as_ref(),
                    )
                    .map_err(to_py_err)?;
                    Some(
                        retrieve_1d_subset(
                            &coord_arr,
                            dim_start, dim_len,
                        )
                        .map_err(to_py_err)?,
                    )
                } else {
                    None
                };

            let time_encoding = self
                .meta
                .arrays
                .get(dim_name)
                .and_then(|m| {
                    m.time_encoding.as_ref()
                });

            let col = build_coord_column(
                dim_name,
                d,
                strides,
                subset_shape,
                subset_origin,
                start,
                len,
                coord_data.as_ref(),
                time_encoding,
            )?;
            cols.push(col);
        }

        let primary_dims = self.grid_states
            [primary_grid_idx]
            .signature
            .dims()
            .to_vec();

        // Add variable columns in consistent order
        let vars: Vec<IStr> = if self.is_hierarchical {
            self.unified_meta
                .as_ref()
                .map(|m| m.root.data_vars.clone())
                .unwrap_or_default()
        } else {
            self.vars.clone()
        };
        for var_name in &vars {
            if dim_names.contains(var_name) {
                continue; // Skip dimension coordinates
            }
            if !self
                .should_emit(var_name.as_ref())
            {
                continue;
            }

            let var_meta = match self
                .meta
                .arrays
                .get(var_name)
            {
                Some(m) => m,
                None => continue,
            };

            let same_dims =
                var_meta.dims.as_slice() == primary_dims;

            if same_dims {
                let arr = Array::open(
                    self.store.clone(),
                    var_meta.path.as_ref(),
                )
                .map_err(to_py_err)?;
                let data = gather_subset_data(
                    self,
                    &arr,
                    subset,
                    subset_shape,
                    subset_origin,
                )?;
                let sliced =
                    data.slice(start, len);
                cols.push(
                    sliced
                        .into_series(
                            var_name.as_ref(),
                        )
                        .into(),
                );
            } else {
                // Variable has different dims - add null column with correct dtype
                let var_name_str: &str =
                    var_name.as_ref();
                let dtype =
                    var_meta.polars_dtype.clone();
                let null_series =
                    Series::new_null(
                        var_name_str.into(),
                        len,
                    )
                    .cast(&dtype)
                    .unwrap_or_else(|_| {
                        Series::new_null(
                            var_name_str.into(),
                            len,
                        )
                    });
                cols.push(null_series.into());
            }
        }

        DataFrame::new(len, cols).map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e.to_string())
        })
    }

    /// Add struct columns for hierarchical child groups.
    fn add_struct_columns_to_df(
        &mut self,
        mut df: DataFrame,
        subset: &zarrs::array_subset::ArraySubset,
        subset_shape: &[u64],
        subset_origin: &[u64],
        _strides: &[u64],
        start: usize,
        len: usize,
    ) -> PyResult<DataFrame> {
        let Some(unified) = self.unified_meta.as_ref()
        else {
            return Ok(df);
        };
        let child_nodes: Vec<(IStr, ZarrNode)> =
            unified
                .root
                .children
                .iter()
                .map(|(name, node)| {
                    (name.clone(), node.clone())
                })
                .collect();
        for (child_name, child_node) in
            child_nodes
        {
            if !self.should_emit_group_path(
                child_node.path.as_ref(),
            ) {
                continue;
            }

            let struct_series =
                self.build_struct_series_for_node(
                    &child_name,
                    &child_node,
                    subset,
                    subset_shape,
                    subset_origin,
                    start,
                    len,
                )?;

            df = df
                .hstack(&[struct_series.into()])
                .map_err(|e| {
                    PyErr::new::<
                        pyo3::exceptions::PyValueError,
                        _,
                    >(
                        e.to_string()
                    )
                })?;
        }

        Ok(df)
    }

    fn should_emit_group_path(
        &self,
        group_path: &str,
    ) -> bool {
        let name = group_path.trim_start_matches('/');
        self.with_columns
            .as_ref()
            .map(|cols| {
                cols.iter().any(|c| {
                    let c_str: &str = c.as_ref();
                    c_str == name
                        || c_str
                            .starts_with(&format!(
                                "{}/",
                                name
                            ))
                })
            })
            .unwrap_or(true)
    }

    fn build_struct_series_for_node(
        &mut self,
        group_name: &IStr,
        node: &ZarrNode,
        subset: &zarrs::array_subset::ArraySubset,
        subset_shape: &[u64],
        subset_origin: &[u64],
        start: usize,
        len: usize,
    ) -> PyResult<Series> {
        if self.unified_meta.is_none() {
            let group_name_str: &str = group_name.as_ref();
            return Ok(Series::new_null(
                group_name_str.into(),
                len,
            ));
        }

        let group_name_str: &str = group_name.as_ref();
        let mut field_series: Vec<Series> =
            Vec::new();
        let primary_ndim = subset_shape.len();

        for var_name in &node.data_vars {
            let var_name_str: &str =
                var_name.as_ref();
            let var_meta =
                node.arrays.get(var_name);
            let dtype = var_meta
                .map(|m| m.polars_dtype.clone())
                .unwrap_or(DataType::Null);

            let same_dims = var_meta
                .map(|m| m.dims.len() == primary_ndim)
                .unwrap_or(false);

            if same_dims {
                let node_path_str: &str =
                    node.path.as_ref();
                let full_path = format!(
                    "{}/{}",
                    node_path_str.trim_end_matches('/'),
                    var_name_str
                );

                let arr_path = self
                    .unified_meta
                    .as_ref()
                    .and_then(|meta| {
                        meta.path_to_array
                            .get(&full_path.istr())
                            .map(|arr| arr.path.clone())
                    });
                if let Some(arr_path) = arr_path {
                    let arr = Array::open(
                        self.store.clone(),
                        arr_path.as_ref(),
                    )
                    .map_err(to_py_err)?;

                    let data =
                        gather_subset_data(
                            self,
                            &arr,
                            subset,
                            subset_shape,
                            subset_origin,
                        )?;
                    let series = data
                        .slice(start, len)
                        .into_series(
                            var_name_str,
                        );
                    field_series.push(series);
                    continue;
                }
            }

            let null_series =
                Series::new_null(
                    var_name_str.into(),
                    len,
                )
                .cast(&dtype)
                .unwrap_or_else(|_| {
                    Series::new_null(
                        var_name_str.into(),
                        len,
                    )
                });
            field_series.push(null_series);
        }

        for (child_name, child_node) in
            &node.children
        {
            if !self.should_emit_group_path(
                child_node.path.as_ref(),
            ) {
                continue;
            }
            let child_series =
                self.build_struct_series_for_node(
                    child_name,
                    child_node,
                    subset,
                    subset_shape,
                    subset_origin,
                    start,
                    len,
                )?;
            field_series.push(child_series);
        }

        if field_series.is_empty() {
            return Ok(Series::new_null(
                group_name_str.into(),
                len,
            ));
        }

        let struct_chunked =
            StructChunked::from_series(
                group_name_str.into(),
                len,
                field_series.iter(),
            )
            .map_err(|e| {
                PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >(e.to_string())
            })?;
        Ok(struct_chunked.into_series())
    }
}

fn gather_subset_data(
    source: &mut ZarrSource,
    arr: &Array<
        dyn zarrs::storage::ReadableWritableListableStorageTraits,
    >,
    subset: &zarrs::array_subset::ArraySubset,
    subset_shape: &[u64],
    subset_origin: &[u64],
) -> PyResult<ColumnData> {
    let mut data = ColumnData::empty_for_dtype(
        arr.data_type().identifier(),
    )
    .ok_or_else(|| {
        to_py_err(format!(
            "unsupported zarr dtype: {}",
            arr.data_type().identifier()
        ))
    })?;

    let chunks = arr
        .chunks_in_array_subset(subset)
        .map_err(to_py_err)?;
    let Some(chunks) = chunks else {
        return Ok(data);
    };

    let mut chunk_indices: Vec<Vec<u64>> = chunks
        .indices()
        .iter()
        .map(|chunk| {
            chunk.iter().copied().collect()
        })
        .collect();
    chunk_indices.sort();
    let array_shape = arr.shape().to_vec();
    let subset_strides =
        compute_strides(subset_shape);
    let mut global_indices: Vec<usize> =
        Vec::new();

    for chunk in chunk_indices {
        source.consume_chunk_budget()?;
        let chunk_data =
            retrieve_chunk(arr, &chunk)
                .map_err(to_py_err)?;
        let chunk_shape = arr
            .chunk_shape(&chunk)
            .map_err(to_py_err)?;
        let chunk_shape: Vec<u64> = chunk_shape
            .iter()
            .map(|v| v.get())
            .collect();
        let chunk_len = chunk_shape
            .iter()
            .try_fold(1usize, |acc, v| {
                acc.checked_mul(*v as usize)
            })
            .ok_or_else(|| {
                to_py_err(
                    "chunk size overflow"
                        .to_string(),
                )
            })?;
        let origin = arr
            .chunk_grid()
            .chunk_origin(&chunk)
            .map_err(to_py_err)?
            .unwrap_or_else(|| {
                vec![0; chunk_shape.len()]
            });
        let strides =
            compute_strides(&chunk_shape);

        let mut keep: Vec<usize> =
            Vec::with_capacity(chunk_len);
        let mut keep_global: Vec<usize> =
            Vec::with_capacity(chunk_len);
        for row in 0..chunk_len {
            let mut ok = true;
            let mut global_idx: u64 = 0;
            for d in 0..chunk_shape.len() {
                let local = (row as u64
                    / strides[d])
                    % chunk_shape[d];
                let global = origin[d] + local;
                let array_len = array_shape
                    .get(d)
                    .copied()
                    .unwrap_or(0);
                let sub_start = subset_origin
                    .get(d)
                    .copied()
                    .unwrap_or(0);
                let sub_len = subset_shape
                    .get(d)
                    .copied()
                    .unwrap_or(1);
                if global >= array_len
                    || global < sub_start
                    || global
                        >= sub_start + sub_len
                {
                    ok = false;
                    break;
                }
                let local_in_subset =
                    global - sub_start;
                let stride = subset_strides
                    .get(d)
                    .copied()
                    .unwrap_or(1);
                global_idx = global_idx
                    .saturating_add(
                        local_in_subset * stride,
                    );
            }
            if ok {
                keep.push(row);
                keep_global
                    .push(global_idx as usize);
            }
        }

        if !keep.is_empty() {
            if keep.len() == chunk_len {
                data.extend(chunk_data);
                global_indices.extend(
                    keep_global.into_iter(),
                );
            } else {
                data.extend(
                    chunk_data
                        .take_indices(&keep),
                );
                global_indices.extend(
                    keep_global.into_iter(),
                );
            }
        }
    }

    if !global_indices.is_empty() {
        let mut order: Vec<usize> =
            (0..global_indices.len()).collect();
        order.sort_by_key(|&i| global_indices[i]);
        data = data.take_indices(&order);
    }

    Ok(data)
}

/// Build a coordinate column for a dimension
fn build_coord_column(
    dim_name: &IStr,
    dim_idx: usize,
    strides: &[u64],
    subset_shape: &[u64],
    subset_origin: &[u64],
    start: usize,
    len: usize,
    coord_data: Option<&ColumnData>,
    time_encoding: Option<
        &crate::meta::TimeEncoding,
    >,
) -> PyResult<Column> {
    let dim_name_str: &str = dim_name.as_ref();
    let dim_size = subset_shape
        .get(dim_idx)
        .copied()
        .unwrap_or(1);
    let dim_origin = subset_origin
        .get(dim_idx)
        .copied()
        .unwrap_or(0);
    let stride = strides
        .get(dim_idx)
        .copied()
        .unwrap_or(1);

    if let Some(te) = time_encoding {
        // Build datetime or duration column
        let mut out_i64: Vec<i64> =
            Vec::with_capacity(len);
        for i in 0..len {
            let row = start + i;
            let local =
                (row as u64 / stride) % dim_size;
            let raw_value =
                if let Some(coord) = coord_data {
                    coord
                        .get_i64(local as usize)
                        .unwrap_or(
                            (dim_origin + local)
                                as i64,
                        )
                } else {
                    (dim_origin + local) as i64
                };
            let ns = if te.is_duration {
                raw_value
                    .saturating_mul(te.unit_ns)
            } else {
                raw_value
                    .saturating_mul(te.unit_ns)
                    .saturating_add(te.epoch_ns)
            };
            out_i64.push(ns);
        }

        let series = if te.is_duration {
            Series::new(
                dim_name_str.into(),
                &out_i64,
            )
            .cast(&DataType::Duration(
                TimeUnit::Nanoseconds,
            ))
            .unwrap_or_else(|_| {
                Series::new(
                    dim_name_str.into(),
                    out_i64,
                )
            })
        } else {
            Series::new(
                dim_name_str.into(),
                &out_i64,
            )
            .cast(&DataType::Datetime(
                TimeUnit::Nanoseconds,
                None,
            ))
            .unwrap_or_else(|_| {
                Series::new(
                    dim_name_str.into(),
                    out_i64,
                )
            })
        };
        Ok(series.into())
    } else if let Some(coord) = coord_data {
        if coord.is_float() {
            let mut out_f64: Vec<f64> =
                Vec::with_capacity(len);
            for i in 0..len {
                let row = start + i;
                let local = (row as u64 / stride)
                    % dim_size;
                out_f64.push(
                    coord
                        .get_f64(local as usize)
                        .unwrap_or(0.0),
                );
            }
            Ok(Series::new(
                dim_name_str.into(),
                out_f64,
            )
            .into())
        } else {
            let mut out_i64: Vec<i64> =
                Vec::with_capacity(len);
            for i in 0..len {
                let row = start + i;
                let local = (row as u64 / stride)
                    % dim_size;
                let val = coord
                    .get_i64(local as usize)
                    .unwrap_or(
                        (dim_origin + local)
                            as i64,
                    );
                out_i64.push(val);
            }
            Ok(Series::new(
                dim_name_str.into(),
                out_i64,
            )
            .into())
        }
    } else {
        // No coordinate array - use index values
        let mut out_i64: Vec<i64> =
            Vec::with_capacity(len);
        for i in 0..len {
            let row = start + i;
            let local =
                (row as u64 / stride) % dim_size;
            out_i64.push(
                (dim_origin + local) as i64,
            );
        }
        Ok(Series::new(
            dim_name_str.into(),
            out_i64,
        )
        .into())
    }
}
