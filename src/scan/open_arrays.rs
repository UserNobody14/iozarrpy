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

