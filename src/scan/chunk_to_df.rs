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
        let dim_len_usize: usize = dim_len
            .try_into()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("dim len overflow"))?;
        let arr = Arc::clone(arr);
        let dim_name = dim_name.clone();
        coord_reads.push(async move {
            let coord = retrieve_1d_subset_async(&arr, dim_start, dim_len).await;
            (dim_name, dim_len_usize, coord)
        });
    }
    let mut coord_slices: std::collections::BTreeMap<String, ColumnData> = Default::default();
    while let Some((name, expected_len, res)) = coord_reads.next().await {
        let coord = res.map_err(to_py_err)?;
        if coord.len() != expected_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
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
                compute_var_chunk_info_async(&idx, &chunk_shape, &dims, &var_meta.dims, &arr)
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

