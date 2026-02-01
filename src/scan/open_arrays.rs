use super::prelude::*;
use crate::meta::ZarrMeta;

/// Open variable and coordinate arrays for reading.
///
/// Returns (var_arrays, coord_arrays) - the caller is responsible for
/// selecting a reference array for chunk iteration based on their needs.
pub(crate) async fn open_arrays_async(
    store: zarrs::storage::AsyncReadableWritableListableStorage,
    meta: &ZarrDatasetMeta,
    vars: &[IStr],
    dims: &[IStr],
) -> Result<
    (
        Vec<(IStr, Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>)>,
        Vec<(IStr, Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>)>,
    ),
    String,
>{
    // Open coord arrays (dims) and variable arrays in parallel.
    let mut coord_futs = FuturesUnordered::new();
    for d in dims {
        if let Some(m) = meta.arrays.get(d) {
            let path = m.path.clone();
            let d_name = d.clone();
            let st = store.clone();
            coord_futs.push(async move {
                let arr = Array::async_open(
                    st,
                    path.as_ref(),
                )
                .await
                .map_err(to_string_err)?;
                Ok::<_, String>((
                    d_name,
                    Arc::new(arr),
                ))
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
            let arr = Array::async_open(
                st,
                path.as_ref(),
            )
            .await
            .map_err(to_string_err)?;
            Ok::<_, String>((
                v_name,
                Arc::new(arr),
            ))
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

    Ok((vars_out, coords))
}

/// Open variable and coordinate arrays for reading (unified ZarrMeta).
pub(crate) async fn open_arrays_async_unified(
    store: zarrs::storage::AsyncReadableWritableListableStorage,
    meta: &ZarrMeta,
    vars: &[IStr],
    dims: &[IStr],
) -> Result<
    (
        Vec<(IStr, Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>)>,
        Vec<(IStr, Arc<Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>>)>,
    ),
    String,
>{
    let mut coord_futs = FuturesUnordered::new();
    for d in dims {
        if let Some(m) =
            meta.array_by_path(d.as_ref())
        {
            let path = m.path.clone();
            let d_name = d.clone();
            let st = store.clone();
            coord_futs.push(async move {
                let arr = Array::async_open(
                    st,
                    path.as_ref(),
                )
                .await
                .map_err(to_string_err)?;
                Ok::<_, String>((
                    d_name,
                    Arc::new(arr),
                ))
            });
        }
    }

    let mut var_futs = FuturesUnordered::new();
    for v in vars {
        let Some(m) =
            meta.array_by_path(v.as_ref())
        else {
            continue;
        };
        let path = m.path.clone();
        let v_name = v.clone();
        let st = store.clone();
        var_futs.push(async move {
            let arr = Array::async_open(
                st,
                path.as_ref(),
            )
            .await
            .map_err(to_string_err)?;
            Ok::<_, String>((
                v_name,
                Arc::new(arr),
            ))
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

    Ok((vars_out, coords))
}

/// Open variable and coordinate arrays for reading (unified ZarrMeta).
pub(crate) fn open_arrays_sync_unified(
    store: zarrs::storage::ReadableWritableListableStorage,
    meta: &ZarrMeta,
    vars: &[IStr],
    dims: &[IStr],
) -> Result<
    (
        Vec<(IStr, Arc<Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>>)>,
        Vec<(IStr, Arc<Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>>)>,
    ),
    String,
>{
    let mut coords = Vec::new();
    for d in dims {
        if let Some(m) =
            meta.array_by_path(d.as_ref())
        {
            let path = m.path.clone();
            let d_name = d.clone();
            let st = store.clone();
            let arr =
                Array::open(st, path.as_ref())
                    .map_err(to_string_err)?;
            coords.push((d_name, Arc::new(arr)));
        }
    }

    let mut vars_out = Vec::new();
    for v in vars {
        let Some(m) =
            meta.array_by_path(v.as_ref())
        else {
            continue;
        };
        let path = m.path.clone();
        let v_name = v.clone();
        let st = store.clone();
        let arr = Array::open(st, path.as_ref())
            .map_err(to_string_err)?;
        vars_out.push((v_name, Arc::new(arr)));
    }

    Ok((vars_out, coords))
}
