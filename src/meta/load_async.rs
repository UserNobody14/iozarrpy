use std::collections::{BTreeMap, BTreeSet};

use zarrs::array::Array;
use zarrs::hierarchy::NodeMetadata;

use crate::meta::dims::{default_dims, dims_for_array, leaf_name};
use crate::meta::dtype::zarr_dtype_to_polars;
use crate::meta::time_encoding::extract_time_encoding;
use crate::meta::types::{ZarrArrayMeta, ZarrDatasetMeta};
use crate::store::{open_store_async, AsyncOpenedStore};

pub async fn open_and_load_dataset_meta_async(
    zarr_url: &str,
) -> Result<(AsyncOpenedStore, ZarrDatasetMeta), String> {
    let opened = open_store_async(zarr_url)?;
    let meta = load_dataset_meta_from_opened_async(&opened).await?;
    Ok((opened, meta))
}

pub async fn load_dataset_meta_from_opened_async(
    opened: &AsyncOpenedStore,
) -> Result<ZarrDatasetMeta, String> {
    let store = opened.store.clone();
    let root = opened.root.clone();

    let group = zarrs::group::Group::async_open(store.clone(), &root)
        .await
        .map_err(to_string_err)?;
    let nodes = group.async_traverse().await.map_err(to_string_err)?;

    let mut arrays: BTreeMap<String, ZarrArrayMeta> = BTreeMap::new();
    let mut seen_names: BTreeMap<String, usize> = BTreeMap::new();
    let mut dims_seen: BTreeSet<String> = BTreeSet::new();
    let mut dims_ordered: Vec<String> = Vec::new();
    let mut coord_candidates = BTreeMap::new();
    let mut primary_dims: Option<Vec<String>> = None;
    let mut max_ndim = 0usize;

    for (path, md) in nodes {
        let NodeMetadata::Array(array_md) = md else {
            continue;
        };

        let path_str = path.as_str().to_string();
        let leaf = leaf_name(&path_str);

        let array = Array::new_with_metadata(store.clone(), &path_str, array_md.clone())
            .map_err(to_string_err)?;

        let shape = array.shape().to_vec();
        let dims = dims_for_array(&array).unwrap_or_else(|| default_dims(shape.len()));

        for d in &dims {
            if dims_seen.insert(d.clone()) {
                dims_ordered.push(d.clone());
            }
        }
        if dims.len() > max_ndim {
            max_ndim = dims.len();
            primary_dims = Some(dims.clone());
        }

        let time_encoding = extract_time_encoding(&array);

        if shape.len() == 1 && dims.len() == 1 && leaf == dims[0] {
            let dt = zarr_dtype_to_polars(array.data_type().identifier(), time_encoding.as_ref());
            coord_candidates.insert(leaf.clone(), (shape.clone(), dt));
        }

        let polars_dtype =
            zarr_dtype_to_polars(array.data_type().identifier(), time_encoding.as_ref());

        let name = match seen_names.get_mut(&leaf) {
            None => {
                seen_names.insert(leaf.clone(), 1);
                leaf.clone()
            }
            Some(n) => {
                *n += 1;
                format!("{leaf}__{n}")
            }
        };

        arrays.insert(
            name.clone(),
            ZarrArrayMeta {
                path: path_str,
                shape,
                dims,
                polars_dtype,
                time_encoding,
            },
        );
    }

    let dims: Vec<String> = primary_dims.unwrap_or(dims_ordered);

    let mut coords: Vec<String> = Vec::new();
    for dim in &dims {
        if let Some((shape, _dtype)) = coord_candidates.get(dim)
            && shape.len() == 1
        {
            coords.push(dim.clone());
        }
    }

    let coord_set: BTreeSet<&str> = coords.iter().map(|s| s.as_str()).collect();
    let data_vars: Vec<String> = arrays
        .keys()
        .filter(|k| !coord_set.contains(k.as_str()))
        .cloned()
        .collect();

    Ok(ZarrDatasetMeta { arrays, dims, data_vars })
}

fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

