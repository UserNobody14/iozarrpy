//! Next-batch iteration logic for ZarrSource.
//!
//! This module handles the per-batch iteration for the Polars scan interface.
//! It reads from multiple chunk grids and joins them on dimension columns.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::PyDataFrame;
use zarrs::array::Array;

use crate::reader::{
    compute_strides, retrieve_1d_subset,
    retrieve_array_subset, ColumnData,
};
use crate::{IStr, IntoIStr};

use super::{to_py_err, ZarrSource};

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
        for v in &self.vars {
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

    /// Read a batch from the current grid, including ALL variables with consistent columns.
    ///
    /// This is the core method for handling heterogeneous chunk grids:
    /// 1. Find a grid with pending work to define the coordinate range
    /// 2. Read ALL variables that share the same dimensions (data for some, reading actual values)
    /// 3. Variables with different dimensions get null columns
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

    /// Read a batch for a specific grid, producing a DataFrame with dims + this grid's variables.
    /// This is used when reading partial DataFrames to be joined.
    fn read_single_grid_batch(
        &self,
        grid_idx: usize,
        subset: &zarrs::array_subset::ArraySubset,
        start: usize,
        len: usize,
        dim_cols: &[Column],
    ) -> PyResult<DataFrame> {
        let grid_state =
            &self.grid_states[grid_idx];
        let dim_names: std::collections::HashSet<
            &IStr,
        > = self.dims.iter().collect();

        let mut cols: Vec<Column> =
            dim_cols.to_vec();

        // Read this grid's variables
        for var_name in &grid_state.variables {
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

            let arr = Array::open(
                self.store.clone(),
                var_meta.path.as_ref(),
            )
            .map_err(to_py_err)?;

            // Read the subset directly
            let data = retrieve_array_subset(
                &arr, subset,
            )
            .map_err(to_py_err)?;

            // Slice for batching
            let sliced = data.slice(start, len);
            cols.push(
                sliced
                    .into_series(
                        var_name.as_ref(),
                    )
                    .into(),
            );
        }

        DataFrame::new(len, cols).map_err(|e| {
            PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(e.to_string())
        })
    }

    /// Read a batch that includes ALL requested variables (with consistent column order).
    /// Variables not in the current grid are read from their respective grids using the same
    /// coordinate range.
    fn read_full_batch(
        &self,
        primary_grid_idx: usize,
        subset: &zarrs::array_subset::ArraySubset,
        subset_shape: &[u64],
        subset_origin: &[u64],
        strides: &[u64],
        start: usize,
        len: usize,
    ) -> PyResult<DataFrame> {
        let dim_names: std::collections::HashSet<
            &IStr,
        > = self.dims.iter().collect();
        let primary_dims = self.grid_states
            [primary_grid_idx]
            .signature
            .dims();

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

        // Build a map of which grid each variable belongs to
        let mut var_to_grid: std::collections::HashMap<
            &IStr,
            usize,
        > = std::collections::HashMap::new();
        for (idx, grid_state) in
            self.grid_states.iter().enumerate()
        {
            for var in &grid_state.variables {
                var_to_grid.insert(var, idx);
            }
        }

        // Add variable columns in self.vars order (consistent across batches)
        for var_name in &self.vars {
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

            // Check if this variable's grid has the same dimensions as the primary grid
            let var_grid_idx =
                var_to_grid.get(var_name);
            let var_dims =
                var_grid_idx.map(|&idx| {
                    self.grid_states[idx]
                        .signature
                        .dims()
                        .to_vec()
                });
            let same_dims = var_dims
                .map_or(false, |d| {
                    d == primary_dims
                });

            if same_dims {
                // Variable shares dims with primary grid - read actual data
                let arr = Array::open(
                    self.store.clone(),
                    var_meta.path.as_ref(),
                )
                .map_err(to_py_err)?;
                let data = retrieve_array_subset(
                    &arr, subset,
                )
                .map_err(to_py_err)?;
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
        &self,
        mut df: DataFrame,
        subset: &zarrs::array_subset::ArraySubset,
        _subset_shape: &[u64],
        _subset_origin: &[u64],
        _strides: &[u64],
        start: usize,
        len: usize,
    ) -> PyResult<DataFrame> {
        let Some(ref unified) = self.unified_meta
        else {
            return Ok(df);
        };

        for (child_name, child_node) in
            &unified.root.children
        {
            let child_name_str: &str =
                child_name.as_ref();

            let should_emit = self
                .with_columns
                .as_ref()
                .map_or(true, |cols| {
                    cols.contains(
                        &child_name_str.istr(),
                    )
                });
            if !should_emit {
                continue;
            }

            let mut field_series: Vec<Series> =
                Vec::new();

            for var_name in &child_node.data_vars
            {
                let var_name_str: &str =
                    var_name.as_ref();

                let child_path_str: &str =
                    child_node.path.as_ref();
                let full_path = format!(
                    "{}{}",
                    child_path_str,
                    if child_path_str
                        .ends_with('/')
                        || var_name_str
                            .starts_with('/')
                    {
                        var_name_str.to_string()
                    } else {
                        format!(
                            "/{}",
                            var_name_str
                        )
                    }
                );

                if let Some(arr_meta) = unified
                    .path_to_array
                    .get(&full_path.istr())
                {
                    let arr = Array::open(
                        self.store.clone(),
                        arr_meta.path.as_ref(),
                    )
                    .map_err(to_py_err)?;

                    let data =
                        retrieve_array_subset(
                            &arr, subset,
                        )
                        .map_err(to_py_err)?;
                    let series = data
                        .slice(start, len)
                        .into_series(
                            var_name_str,
                        );
                    field_series.push(series);
                } else {
                    let null_series =
                        Series::new_null(
                            var_name_str.into(),
                            len,
                        );
                    field_series
                        .push(null_series);
                }
            }

            if !field_series.is_empty() {
                let struct_chunked =
                    StructChunked::from_series(
                        child_name_str.into(),
                        len,
                        field_series.iter(),
                    )
                    .map_err(|e| {
                        PyErr::new::<
                            pyo3::exceptions::PyValueError,
                            _,
                        >(
                            e.to_string()
                        )
                    })?;
                df = df
                    .hstack(&[struct_chunked
                        .into_series()
                        .into()])
                    .map_err(|e| {
                        PyErr::new::<
                            pyo3::exceptions::PyValueError,
                            _,
                        >(
                            e.to_string()
                        )
                    })?;
            }
        }

        Ok(df)
    }
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
