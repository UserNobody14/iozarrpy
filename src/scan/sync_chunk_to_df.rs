//////////
use crate::IStr;
/// Sync equivalents of async functions.
///
///
use crate::chunk_plan::ChunkGridSignature;
use crate::meta::ZarrDatasetMeta;
use crate::reader::compute_var_chunk_info;
use crate::reader::{
    retrieve_1d_subset, retrieve_chunk,
};
use crate::scan::prelude::*;
use polars::prelude::DataFrame;
use std::collections::BTreeSet;
use std::sync::Arc;
use zarrs::array::Array;
use zarrs::array::ChunkGrid;

pub(crate) fn chunk_to_df(
    idx: Vec<u64>,
    meta: Arc<ZarrDatasetMeta>,
    chunk_shape: &[u64],
    chunk_len: usize,
    origin: &[u64],
    array_shape: &[u64],
    dims: Arc<Vec<IStr>>,
    _vars: Arc<Vec<IStr>>,
    var_arrays: Arc<Vec<(IStr, Arc<Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>>)>>,
    coord_arrays: Arc<Vec<(IStr, Arc<Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>>)>>,
    with_columns: Arc<Option<BTreeSet<IStr>>>,
) -> Result<DataFrame, PyErr> {
    let strides = compute_strides(&chunk_shape);

    // In-bounds mask.
    let mut keep: Vec<usize> =
        Vec::with_capacity(chunk_len);
    for row in 0..chunk_len {
        let mut ok = true;
        for d in 0..chunk_shape.len() {
            let local = (row as u64 / strides[d])
                % chunk_shape[d];
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

    // Coord reads (per dim).
    let mut coord_slices: std::collections::BTreeMap<
        IStr,
        ColumnData,
    > = Default::default();
    for (d, dim_name) in dims.iter().enumerate() {
        if !with_columns
            .as_ref()
            .as_ref()
            .map(|s| s.contains(dim_name))
            .unwrap_or(true)
        {
            continue;
        }
        let Some((_, arr)) = coord_arrays
            .iter()
            .find(|(n, _)| n == dim_name)
        else {
            continue;
        };
        let dim_start = origin[d];
        let dim_len = chunk_shape[d];
        let dim_len_usize: usize =
            dim_len.try_into().map_err(|_| {
                PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >("dim len overflow")
            })?;
        let coord = retrieve_1d_subset(
            arr, dim_start, dim_len,
        )
        .map_err(to_py_err)?;
        if coord.len() != dim_len_usize {
            return Err(PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(format!(
                "coord '{}' length mismatch: expected {}, got {}",
                dim_name,
                dim_len_usize,
                coord.len()
            )));
        }
        coord_slices
            .insert(dim_name.clone(), coord);
    }

    // Var chunk reads.
    let mut var_chunks: Vec<(
        IStr,
        ColumnData,
        SmallVec<[IStr; 4]>,
        Vec<u64>,
        Vec<u64>,
    )> = Vec::new();
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
            .ok_or_else(|| {
                PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >("unknown variable")
            })?
            .clone();
        let var_meta_dims: Vec<IStr> = var_meta
            .dims
            .iter()
            .cloned()
            .collect();

        // Get the variable's chunk grid shape to validate indices
        let var_grid_shape: Vec<u64> = arr
            .chunk_grid()
            .grid_shape()
            .to_vec();

        let (var_chunk_indices, var_offsets) =
            if var_meta_dims.len() == dims.len()
                && var_meta_dims == *dims
            {
                // Same dimensions - but check if chunk index is valid for this variable's grid
                let idx_valid = idx.len()
                    == var_grid_shape.len()
                    && idx
                        .iter()
                        .zip(
                            var_grid_shape.iter(),
                        )
                        .all(|(i, g)| *i < *g);

                if idx_valid {
                    (
                        idx.clone(),
                        vec![0; dims.len()],
                    )
                } else {
                    // Chunk index out of range for this variable - clamp to valid range
                    let clamped: Vec<u64> = idx
                        .iter()
                        .zip(
                            var_grid_shape.iter(),
                        )
                        .map(|(i, g)| {
                            (*i).min(
                                g.saturating_sub(
                                    1,
                                ),
                            )
                        })
                        .collect();
                    (clamped, vec![0; dims.len()])
                }
            } else {
                compute_var_chunk_info(
                    &idx,
                    chunk_shape,
                    &dims,
                    &var_meta_dims,
                    arr,
                )
                .map_err(to_py_err)?
            };

        let var_chunk_shape: Vec<u64> =
            if var_chunk_indices.is_empty() {
                vec![]
            } else {
                arr.chunk_shape(
                    &var_chunk_indices,
                )
                .map_err(to_py_err)?
                .iter()
                .map(|x| x.get())
                .collect()
            };
        if !var_chunk_shape.is_empty() {
            let _ = checked_chunk_len(
                &var_chunk_shape,
            )?;
        }

        let data = retrieve_chunk(
            arr,
            &var_chunk_indices,
        )
        .map_err(to_py_err)?;
        var_chunks.push((
            name.clone(),
            data,
            var_meta.dims,
            var_chunk_shape,
            var_offsets,
        ));
    }

    let mut cols: Vec<Column> = Vec::new();
    let height = keep.len();

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

        let time_encoding =
            meta.arrays.get(dim_name).and_then(
                |m| m.time_encoding.as_ref(),
            );

        let dim_name_str: &str =
            dim_name.as_ref();
        if let Some(te) = time_encoding {
            let mut out_i64: Vec<i64> =
                Vec::with_capacity(keep.len());
            for &row in &keep {
                let local = (row as u64
                    / strides[d])
                    % chunk_shape[d];
                let raw_value = coord_slices
                    .get(dim_name)
                    .and_then(|c| {
                        c.get_i64(local as usize)
                    })
                    .unwrap_or(
                        (origin[d] + local)
                            as i64,
                    );
                let ns = if te.is_duration {
                    raw_value.saturating_mul(
                        te.unit_ns,
                    )
                } else {
                    raw_value
                        .saturating_mul(
                            te.unit_ns,
                        )
                        .saturating_add(
                            te.epoch_ns,
                        )
                };
                out_i64.push(ns);
            }
            let series =
                if te.is_duration {
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
            cols.push(series.into());
        } else if let Some(coord) =
            coord_slices.get(dim_name)
            && coord.is_float()
        {
            let mut out_f64: Vec<f64> =
                Vec::with_capacity(keep.len());
            for &row in &keep {
                let local = (row as u64
                    / strides[d])
                    % chunk_shape[d];
                out_f64.push(
                    coord
                        .get_f64(local as usize)
                        .unwrap(),
                );
            }
            cols.push(
                Series::new(
                    dim_name_str.into(),
                    out_f64,
                )
                .into(),
            );
        } else {
            let mut out_i64: Vec<i64> =
                Vec::with_capacity(keep.len());
            for &row in &keep {
                let local = (row as u64
                    / strides[d])
                    % chunk_shape[d];
                if let Some(coord) =
                    coord_slices.get(dim_name)
                {
                    if let Some(v) = coord
                        .get_i64(local as usize)
                    {
                        out_i64.push(v);
                    } else {
                        out_i64.push(
                            (origin[d] + local)
                                as i64,
                        );
                    }
                } else {
                    out_i64.push(
                        (origin[d] + local)
                            as i64,
                    );
                }
            }
            cols.push(
                Series::new(
                    dim_name_str.into(),
                    out_i64,
                )
                .into(),
            );
        }
    }

    // Variable columns.
    for (
        name,
        data,
        var_dims,
        var_chunk_shape,
        var_offsets,
    ) in var_chunks
    {
        let var_dims_vec: Vec<IStr> =
            var_dims.iter().cloned().collect();

        // Check if we can use direct indexing:
        // - Same dimensions in same order
        // - Same chunk shape (important! different chunking means different data layout)
        // - Zero offsets
        let same_dims = var_dims_vec.len()
            == dims.len()
            && var_dims_vec == *dims;
        let same_chunk_shape =
            var_chunk_shape == chunk_shape;
        let zero_offsets =
            var_offsets.iter().all(|&o| o == 0);

        if same_dims
            && same_chunk_shape
            && zero_offsets
        {
            // Fast path: direct index mapping
            cols.push(
                data.take_indices(&keep)
                    .into_series(name.as_ref())
                    .into(),
            );
        } else {
            // Slow path: map indices through dimension/chunk shape differences
            let dim_mapping: Vec<Option<usize>> =
                dims.iter()
                    .map(|pd| {
                        var_dims.iter().position(
                            |vd| vd == pd,
                        )
                    })
                    .collect();
            let var_strides =
                compute_strides(&var_chunk_shape);
            let var_data_len = data.len();

            let indices: Vec<usize> = keep
                .iter()
                .map(|&row| {
                    let mut var_idx: u64 = 0;
                    for (primary_d, maybe_var_d) in
                        dim_mapping.iter().enumerate()
                    {
                        if let Some(var_d) = *maybe_var_d {
                            let local = (row as u64
                                / strides[primary_d])
                                % chunk_shape[primary_d];
                            // When chunk shapes differ, map local to the var's chunk shape
                            let var_local = if same_dims
                                && var_chunk_shape.len()
                                    > var_d
                            {
                                // Same dims but different chunk shape: clamp to var's chunk bounds
                                local.min(
                                    var_chunk_shape[var_d]
                                        .saturating_sub(1),
                                )
                            } else {
                                local
                            };
                            let local_with_offset =
                                var_local
                                    + var_offsets[var_d];
                            var_idx += local_with_offset
                                * var_strides[var_d];
                        }
                    }
                    // Clamp to data bounds to prevent panics
                    (var_idx as usize)
                        .min(var_data_len.saturating_sub(1))
                })
                .collect();
            cols.push(
                data.take_indices(&indices)
                    .into_series(name.as_ref())
                    .into(),
            );
        }
    }

    Ok(DataFrame::new(height, cols)
        .map_err(PyPolarsErr::from)?)
}

pub fn chunk_to_df_from_grid(
    idx: Vec<u64>,
    sig: ChunkGridSignature,
    grid: Arc<ChunkGrid>,
    meta: Arc<ZarrDatasetMeta>,
    dims: Arc<Vec<IStr>>,
    _vars: Arc<Vec<IStr>>,
    var_arrays: Arc<Vec<(IStr, Arc<Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>>)>>,
    coord_arrays: Arc<Vec<(IStr, Arc<Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>>)>>,
) -> Result<DataFrame, PyErr> {
    let chunk_shape = sig.chunk_shape();

    let chunk_len =
        checked_chunk_len(&chunk_shape)?;

    let array_shape = grid.array_shape().to_vec();
    let origin = grid
        .chunk_origin(&idx)
        .map_err(to_py_err)?
        .unwrap_or_else(|| {
            vec![0; chunk_shape.len()]
        });
    return Ok(chunk_to_df(
        idx,
        meta,
        &chunk_shape,
        chunk_len,
        &origin,
        &array_shape,
        dims,
        _vars,
        var_arrays,
        coord_arrays,
        Arc::new(None),
    )?);
}
