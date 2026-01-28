use super::prelude::*;
use crate::IntoIStr;
use crate::meta::dims::dims_for_array;
use futures::future::BoxFuture;

pub(crate) async fn chunk_to_df(
    idx: Vec<u64>,
    primary: Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>,
    meta: Arc<ZarrDatasetMeta>,
    _dims: Arc<Vec<IStr>>,
    _vars: Arc<Vec<IStr>>,
    var_arrays: Arc<Vec<(IStr, Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>)>>,
    coord_arrays: Arc<Vec<(IStr, Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>)>>,
    with_columns: Arc<Option<BTreeSet<IStr>>>,
) -> Result<DataFrame, PyErr> {
    // Get dimension names from the primary array itself, not from global meta.dims
    // This ensures correct ordering for this specific array/grid
    let dims: Arc<Vec<IStr>> = Arc::new(
        dims_for_array(primary.as_ref())
            .map(|sv| {
                sv.into_iter()
                    .collect::<Vec<IStr>>()
            })
            .unwrap_or_else(|| {
                // Fallback to dim_0, dim_1, etc.
                (0..primary.dimensionality())
                    .map(|i| {
                        format!("dim_{i}").istr()
                    })
                    .collect()
            }),
    );

    // Compute primary chunk geometry.
    let chunk_shape_nz = primary
        .chunk_shape(&idx)
        .map_err(to_py_err)?;
    let chunk_shape: Vec<u64> = chunk_shape_nz
        .iter()
        .map(|x| x.get())
        .collect();
    let chunk_len =
        checked_chunk_len(&chunk_shape)?;

    let array_shape = primary.shape().to_vec();
    let origin = primary
        .chunk_grid()
        .chunk_origin(&idx)
        .map_err(to_py_err)?
        .unwrap_or_else(|| {
            vec![0; chunk_shape.len()]
        });
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
        let arr = Arc::clone(arr);
        let dim_name = dim_name.clone();
        coord_reads.push(async move {
            let coord = retrieve_1d_subset_async(
                &arr, dim_start, dim_len,
            )
            .await;
            (dim_name, dim_len_usize, coord)
        });
    }
    let mut coord_slices: std::collections::BTreeMap<
        IStr,
        ColumnData,
    > = Default::default();
    while let Some((name, expected_len, res)) =
        coord_reads.next().await
    {
        let coord = res.map_err(to_py_err)?;
        if coord.len() != expected_len {
            return Err(PyErr::new::<
                pyo3::exceptions::PyValueError,
                _,
            >(format!(
                "coord '{name}' length mismatch: expected {expected_len}, got {}",
                coord.len()
            )));
        }
        coord_slices.insert(name, coord);
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
            .ok_or_else(|| {
                PyErr::new::<
                    pyo3::exceptions::PyValueError,
                    _,
                >("unknown variable")
            })?
            .clone();
        let arr = Arc::clone(arr);
        let name = name.clone();
        let dims = Arc::clone(&dims);
        let idx = idx.clone();
        let chunk_shape = chunk_shape.clone();
        var_reads.push(async move {
            let var_meta_dims: Vec<IStr> =
                var_meta.dims.iter().cloned().collect();

            // Get the variable's chunk grid shape to validate indices
            let var_grid_shape: Vec<u64> =
                arr.chunk_grid().grid_shape().to_vec();

            let (var_chunk_indices, var_offsets) =
                if var_meta_dims.len() == dims.len()
                    && var_meta_dims == *dims
                {
                    // Same dimensions - but check if chunk index is valid for this variable's grid
                    let idx_valid = idx.len()
                        == var_grid_shape.len()
                        && idx
                            .iter()
                            .zip(var_grid_shape.iter())
                            .all(|(i, g)| *i < *g);

                    if idx_valid {
                        (idx.clone(), vec![0; dims.len()])
                    } else {
                        // Chunk index out of range for this variable - clamp to valid range
                        let clamped: Vec<u64> = idx
                            .iter()
                            .zip(var_grid_shape.iter())
                            .map(|(i, g)| {
                                (*i).min(
                                    g.saturating_sub(1),
                                )
                            })
                            .collect();
                        (clamped, vec![0; dims.len()])
                    }
                } else {
                    compute_var_chunk_info_async(
                        &idx,
                        &chunk_shape,
                        &dims,
                        &var_meta_dims,
                        &arr,
                    )
                    .map_err(to_py_err)?
                };

            let var_chunk_shape: Vec<u64> =
                if var_chunk_indices.is_empty() {
                    vec![]
                } else {
                    arr.chunk_shape(&var_chunk_indices)
                        .map_err(to_py_err)?
                        .iter()
                        .map(|x| x.get())
                        .collect()
                };
            if !var_chunk_shape.is_empty() {
                let _ =
                    checked_chunk_len(&var_chunk_shape)?;
            }

            let data = retrieve_chunk_async(
                &arr,
                &var_chunk_indices,
            )
            .await
            .map_err(to_py_err)?;
            Ok::<_, PyErr>((
                name,
                data,
                var_meta.dims,
                var_chunk_shape,
                var_offsets,
            ))
        });
    }

    let mut var_chunks: Vec<(
        IStr,
        ColumnData,
        SmallVec<[IStr; 4]>,
        Vec<u64>,
        Vec<u64>,
    )> = Vec::new();
    while let Some(r) = var_reads.next().await {
        var_chunks.push(r?);
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

// =============================================================================
// Hierarchical DataTree Chunk Loading
// =============================================================================

type AsyncArray = Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>;

/// Load a chunk from a hierarchical zarr store and return a DataFrame with struct columns.
///
/// For flat stores (no children), this behaves identically to chunk_to_df.
/// For hierarchical stores, child groups become struct columns in the output.
pub(crate) async fn chunk_to_df_tree(
    idx: Vec<u64>,
    primary: Arc<AsyncArray>,
    meta: Arc<ZarrMeta>,
    var_arrays: Arc<Vec<(IStr, Arc<AsyncArray>)>>,
    coord_arrays: Arc<
        Vec<(IStr, Arc<AsyncArray>)>,
    >,
    with_columns: Arc<Option<BTreeSet<IStr>>>,
) -> Result<DataFrame, PyErr> {
    // If not hierarchical, delegate to simpler flat implementation
    if !meta.is_hierarchical() {
        // Convert to legacy format and use existing implementation
        let legacy_meta =
            ZarrDatasetMeta::from(&*meta);
        let dims = Arc::new(
            meta.dim_analysis.all_dims.clone(),
        );
        let vars =
            Arc::new(meta.root.data_vars.clone());
        return chunk_to_df(
            idx,
            primary,
            Arc::new(legacy_meta),
            dims,
            vars,
            var_arrays,
            coord_arrays,
            with_columns,
        )
        .await;
    }

    // For hierarchical stores, we need to:
    // 1. Compute the combined output grid based on all dimensions
    // 2. Build dimension columns
    // 3. Build root data variable columns (with broadcasting if needed)
    // 4. Build child group struct columns

    let dim_analysis = &meta.dim_analysis;
    let output_dims = &dim_analysis.all_dims;

    // Compute primary chunk geometry
    let chunk_shape_nz = primary
        .chunk_shape(&idx)
        .map_err(to_py_err)?;
    let chunk_shape: Vec<u64> = chunk_shape_nz
        .iter()
        .map(|x| x.get())
        .collect();
    let chunk_len =
        checked_chunk_len(&chunk_shape)?;

    let array_shape = primary.shape().to_vec();
    let origin = primary
        .chunk_grid()
        .chunk_origin(&idx)
        .map_err(to_py_err)?
        .unwrap_or_else(|| {
            vec![0; chunk_shape.len()]
        });
    let strides = compute_strides(&chunk_shape);

    // In-bounds mask
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

    let height = keep.len();
    let mut cols: Vec<Column> = Vec::new();

    // 1. Build dimension columns (same as flat case)
    for (d, dim_name) in
        output_dims.iter().enumerate()
    {
        if !should_emit_col(
            dim_name,
            &with_columns,
        ) {
            continue;
        }

        let time_encoding = meta
            .path_to_array
            .get(dim_name)
            .and_then(|m| {
                m.time_encoding.as_ref()
            });

        let dim_name_str: &str =
            dim_name.as_ref();

        // Try to find coord array
        let coord_data = if let Some((_, arr)) =
            coord_arrays
                .iter()
                .find(|(n, _)| n == dim_name)
        {
            if d < origin.len() {
                let dim_start = origin[d];
                let dim_len = chunk_shape[d];
                retrieve_1d_subset_async(
                    arr, dim_start, dim_len,
                )
                .await
                .ok()
            } else {
                None
            }
        } else {
            None
        };

        let col = build_dim_column_internal(
            dim_name_str,
            d,
            &keep,
            &strides,
            &chunk_shape,
            &origin,
            coord_data.as_ref(),
            time_encoding,
        );
        cols.push(col);
    }

    // 2. Build root data variable columns
    for var_name in &meta.root.data_vars {
        if !should_emit_col(
            var_name,
            &with_columns,
        ) {
            continue;
        }

        let Some((_, arr)) = var_arrays
            .iter()
            .find(|(n, _)| n == var_name)
        else {
            continue;
        };

        let Some(var_meta) =
            meta.root.arrays.get(var_name)
        else {
            continue;
        };

        let data =
            retrieve_chunk_async(arr, &idx)
                .await
                .map_err(to_py_err)?;
        let var_dims: Vec<IStr> = var_meta
            .dims
            .iter()
            .cloned()
            .collect();
        let var_chunk_shape: Vec<u64> = arr
            .chunk_shape(&idx)
            .map_err(to_py_err)?
            .iter()
            .map(|x| x.get())
            .collect();

        let col = build_var_column_with_broadcast(
            var_name.as_ref(),
            &data,
            &var_dims,
            output_dims,
            &strides,
            &chunk_shape,
            &var_chunk_shape,
            &keep,
        );
        cols.push(col);
    }

    // 3. Build child group struct columns
    for (child_name, child_node) in
        &meta.root.children
    {
        if !should_emit_group(
            child_name,
            &with_columns,
        ) {
            continue;
        }

        let struct_series =
            build_group_struct_series(
                child_name,
                child_node,
                &idx,
                output_dims,
                &strides,
                &chunk_shape,
                &keep,
                &var_arrays,
                &meta,
            )
            .await?;

        cols.push(struct_series.into());
    }

    Ok(DataFrame::new(height, cols)
        .map_err(PyPolarsErr::from)?)
}

fn should_emit_col(
    name: &IStr,
    with_columns: &Option<BTreeSet<IStr>>,
) -> bool {
    with_columns
        .as_ref()
        .map(|s| s.contains(name))
        .unwrap_or(true)
}

fn should_emit_group(
    name: &IStr,
    with_columns: &Option<BTreeSet<IStr>>,
) -> bool {
    // Include group if either the group name is in selection, or any of its vars are
    with_columns
        .as_ref()
        .map(|s| {
            s.contains(name)
                || s.iter().any(|v| {
                    let v_str: &str = v.as_ref();
                    {
                        let name_str: &str =
                            name.as_ref();
                        v_str.starts_with(
                            &format!(
                                "{}/",
                                name_str
                            ),
                        )
                    }
                })
        })
        .unwrap_or(true)
}

fn build_dim_column_internal(
    dim_name: &str,
    dim_idx: usize,
    keep: &[usize],
    strides: &[u64],
    chunk_shape: &[u64],
    origin: &[u64],
    coord_data: Option<&ColumnData>,
    time_encoding: Option<
        &crate::meta::TimeEncoding,
    >,
) -> Column {
    if let Some(te) = time_encoding {
        let mut out_i64: Vec<i64> =
            Vec::with_capacity(keep.len());
        for &row in keep {
            let local = (row as u64
                / strides[dim_idx])
                % chunk_shape[dim_idx];
            let raw_value = coord_data
                .and_then(|c| {
                    c.get_i64(local as usize)
                })
                .unwrap_or(
                    (origin[dim_idx] + local)
                        as i64,
                );
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
            Series::new(dim_name.into(), &out_i64)
                .cast(&DataType::Duration(
                    TimeUnit::Nanoseconds,
                ))
                .unwrap_or_else(|_| {
                    Series::new(
                        dim_name.into(),
                        out_i64,
                    )
                })
        } else {
            Series::new(dim_name.into(), &out_i64)
                .cast(&DataType::Datetime(
                    TimeUnit::Nanoseconds,
                    None,
                ))
                .unwrap_or_else(|_| {
                    Series::new(
                        dim_name.into(),
                        out_i64,
                    )
                })
        };
        series.into()
    } else if let Some(coord) = coord_data {
        if coord.is_float() {
            let mut out_f64: Vec<f64> =
                Vec::with_capacity(keep.len());
            for &row in keep {
                let local = (row as u64
                    / strides[dim_idx])
                    % chunk_shape[dim_idx];
                out_f64.push(
                    coord
                        .get_f64(local as usize)
                        .unwrap_or(0.0),
                );
            }
            Series::new(dim_name.into(), out_f64)
                .into()
        } else {
            let mut out_i64: Vec<i64> =
                Vec::with_capacity(keep.len());
            for &row in keep {
                let local = (row as u64
                    / strides[dim_idx])
                    % chunk_shape[dim_idx];
                out_i64.push(
                    coord
                        .get_i64(local as usize)
                        .unwrap_or(
                            (origin[dim_idx]
                                + local)
                                as i64,
                        ),
                );
            }
            Series::new(dim_name.into(), out_i64)
                .into()
        }
    } else {
        let mut out_i64: Vec<i64> =
            Vec::with_capacity(keep.len());
        for &row in keep {
            let local = (row as u64
                / strides[dim_idx])
                % chunk_shape[dim_idx];
            out_i64.push(
                (origin[dim_idx] + local) as i64,
            );
        }
        Series::new(dim_name.into(), out_i64)
            .into()
    }
}

fn build_var_column_with_broadcast(
    var_name: &str,
    data: &ColumnData,
    var_dims: &[IStr],
    output_dims: &[IStr],
    output_strides: &[u64],
    output_chunk_shape: &[u64],
    var_chunk_shape: &[u64],
    keep: &[usize],
) -> Column {
    // Check if variable has same dimensions as output (no broadcasting needed)
    let same_dims = var_dims.len()
        == output_dims.len()
        && var_dims
            .iter()
            .zip(output_dims.iter())
            .all(|(a, b)| a == b);

    if same_dims {
        data.take_indices(keep)
            .into_series(var_name)
            .into()
    } else {
        // Need to broadcast: map output indices to variable indices
        let dim_mapping: Vec<Option<usize>> =
            output_dims
                .iter()
                .map(|od| {
                    var_dims
                        .iter()
                        .position(|vd| vd == od)
                })
                .collect();
        let var_strides =
            compute_strides(var_chunk_shape);

        let indices: Vec<usize> = keep
            .iter()
            .map(|&row| {
                let mut var_idx: u64 = 0;
                for (out_d, maybe_var_d) in
                    dim_mapping.iter().enumerate()
                {
                    if let Some(var_d) =
                        *maybe_var_d
                    {
                        let local = (row as u64
                            / output_strides
                                [out_d])
                            % output_chunk_shape
                                [out_d];
                        if var_d
                            < var_strides.len()
                        {
                            var_idx += local
                                * var_strides
                                    [var_d];
                        }
                    }
                }
                var_idx as usize
            })
            .collect();

        data.take_indices(&indices)
            .into_series(var_name)
            .into()
    }
}

fn build_group_struct_series<'a>(
    group_name: &'a IStr,
    node: &'a ZarrNode,
    idx: &'a [u64],
    output_dims: &'a [IStr],
    output_strides: &'a [u64],
    output_chunk_shape: &'a [u64],
    keep: &'a [usize],
    var_arrays: &'a [(IStr, Arc<AsyncArray>)],
    meta: &'a ZarrMeta,
) -> BoxFuture<'a, Result<Series, PyErr>> {
    Box::pin(async move {
        let group_name_str: &str =
            group_name.as_ref();
        let mut field_series: Vec<Series> =
            Vec::new();

        // Load each data variable in this child group
        for var_name in &node.data_vars {
            // Build the path to find this variable in var_arrays
            let node_path_str: &str =
                node.path.as_ref();
            let var_path = format!(
                "{}/{}",
                node_path_str
                    .trim_start_matches('/'),
                var_name
            );

            let arr_opt = var_arrays.iter().find(
                |(n, _)| {
                    let n_str: &str = n.as_ref();
                    let var_str: &str =
                        var_name.as_ref();
                    n_str == var_path
                        || n_str == var_str
                },
            );

            if let Some((_, arr)) = arr_opt {
                let var_meta =
                    node.arrays.get(var_name);

                // Retrieve chunk data
                let data = retrieve_chunk_async(
                    arr, idx,
                )
                .await
                .map_err(to_py_err)?;

                let var_dims: Vec<IStr> =
                    var_meta
                        .map(|m| {
                            m.dims
                                .iter()
                                .cloned()
                                .collect()
                        })
                        .unwrap_or_default();

                let var_chunk_shape: Vec<u64> =
                    arr.chunk_shape(idx)
                        .map(|s| {
                            s.iter()
                                .map(|x| x.get())
                                .collect()
                        })
                        .unwrap_or_default();

                // Build the series with broadcasting
                let series =
                build_var_series_with_broadcast(
                    var_name.as_ref(),
                    &data,
                    &var_dims,
                    output_dims,
                    output_strides,
                    output_chunk_shape,
                    &var_chunk_shape,
                    keep,
                );

                field_series.push(series);
            } else {
                // Variable not found - create null series
                let var_name_str: &str =
                    var_name.as_ref();
                let null_series =
                    Series::new_null(
                        var_name_str.into(),
                        keep.len(),
                    );
                field_series.push(null_series);
            }
        }

        // Recursively add nested child groups as struct fields
        for (child_name, child_node) in
            &node.children
        {
            let child_series =
                build_group_struct_series(
                    child_name,
                    child_node,
                    idx,
                    output_dims,
                    output_strides,
                    output_chunk_shape,
                    keep,
                    var_arrays,
                    meta,
                )
                .await?;
            field_series.push(child_series);
        }

        // Create struct column from the field series
        if field_series.is_empty() {
            // Empty struct
            let null_series = Series::new_null(
                group_name_str.into(),
                keep.len(),
            );
            Ok(null_series)
        } else {
            let struct_chunked = StructChunked::from_series(
                group_name_str.into(),
                field_series.len(),
                field_series.iter(),
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    e.to_string(),
                )
            })?;
            Ok(struct_chunked.into_series())
        }
    })
}

fn build_var_series_with_broadcast(
    var_name: &str,
    data: &ColumnData,
    var_dims: &[IStr],
    output_dims: &[IStr],
    output_strides: &[u64],
    output_chunk_shape: &[u64],
    var_chunk_shape: &[u64],
    keep: &[usize],
) -> Series {
    // Check if variable has same dimensions as output (no broadcasting needed)
    let same_dims = var_dims.len()
        == output_dims.len()
        && var_dims
            .iter()
            .zip(output_dims.iter())
            .all(|(a, b)| a == b);

    if same_dims {
        data.take_indices(keep)
            .into_series(var_name)
    } else {
        // Need to broadcast: map output indices to variable indices
        let dim_mapping: Vec<Option<usize>> =
            output_dims
                .iter()
                .map(|od| {
                    var_dims
                        .iter()
                        .position(|vd| vd == od)
                })
                .collect();
        let var_strides =
            compute_strides(var_chunk_shape);

        let indices: Vec<usize> = keep
            .iter()
            .map(|&row| {
                let mut var_idx: u64 = 0;
                for (out_d, maybe_var_d) in
                    dim_mapping.iter().enumerate()
                {
                    if let Some(var_d) =
                        *maybe_var_d
                    {
                        let local = (row as u64
                            / output_strides
                                [out_d])
                            % output_chunk_shape
                                [out_d];
                        if var_d
                            < var_strides.len()
                        {
                            var_idx += local
                                * var_strides
                                    [var_d];
                        }
                    }
                }
                var_idx as usize
            })
            .collect();

        data.take_indices(&indices)
            .into_series(var_name)
    }
}
