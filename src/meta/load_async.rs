use std::collections::{BTreeMap, BTreeSet};

use smallvec::SmallVec;
use zarrs::array::Array;
use zarrs::hierarchy::NodeMetadata;

use crate::meta::dims::{default_dims, dims_for_array, leaf_name};
use crate::meta::dtype::zarr_dtype_to_polars;
use crate::meta::time_encoding::extract_time_encoding;
use crate::meta::types::{ZarrArrayMeta, ZarrDatasetMeta};
use crate::store::{open_store_async, AsyncOpenedStore, StoreInput};
use crate::{IStr, IntoIStr};

pub async fn open_and_load_dataset_meta_async(
    zarr_url: &str,
) -> Result<(AsyncOpenedStore, ZarrDatasetMeta), String> {
    let opened = open_store_async(zarr_url)?;
    let meta = load_dataset_meta_from_opened_async(&opened).await?;
    Ok((opened, meta))
}

/// Open and load dataset metadata from a StoreInput (URL string or ObjectStore instance).
pub async fn open_and_load_dataset_meta_from_input_async(
    store_input: StoreInput,
) -> Result<(AsyncOpenedStore, ZarrDatasetMeta), String> {
    let opened = store_input.open_async()?;
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

    let mut arrays: BTreeMap<IStr, ZarrArrayMeta> = BTreeMap::new();
    let mut seen_names: BTreeMap<IStr, usize> = BTreeMap::new();
    let mut dims_seen: BTreeSet<IStr> = BTreeSet::new();
    let mut dims_ordered: Vec<IStr> = Vec::new();
    let mut coord_candidates: BTreeMap<IStr, (Vec<u64>, _)> = BTreeMap::new();
    let mut primary_dims: Option<SmallVec<[IStr; 4]>> = None;
    let mut max_ndim = 0usize;
    // Collect auxiliary coordinates from "coordinates" attribute (CF convention)
    let mut aux_coords: BTreeSet<IStr> = BTreeSet::new();

    for (path, md) in nodes {
        let NodeMetadata::Array(array_md) = md else {
            continue;
        };

        let path_str = path.as_str().istr();
        let leaf = leaf_name(path_str.as_ref());

        let array = Array::new_with_metadata(store.clone(), path_str.as_ref(), array_md.clone())
            .map_err(to_string_err)?;

        let shape: std::sync::Arc<[u64]> = array.shape().into();
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
            coord_candidates.insert(leaf.clone(), (shape.to_vec(), dt));
        }

        // Parse "coordinates" attribute (CF convention) to identify auxiliary coords
        if let Some(attrs) = array.attributes().get("coordinates") {
            if let Some(coord_str) = attrs.as_str() {
                for coord_name in coord_str.split_whitespace() {
                    aux_coords.insert(coord_name.istr());
                }
            }
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
                format!("{leaf}__{n}").istr()
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

    let dims: Vec<IStr> = primary_dims
        .map(|pd| pd.into_iter().collect())
        .unwrap_or(dims_ordered);

    // Collect dimension coordinate arrays (1D arrays matching their dimension name)
    let mut coords: BTreeSet<IStr> = BTreeSet::new();
    for dim in &dims {
        if let Some((shape, _dtype)) = coord_candidates.get(dim)
            && shape.len() == 1
        {
            coords.insert(dim.clone());
        }
    }

    // Add auxiliary coordinates (only those that exist as arrays)
    for aux in &aux_coords {
        if arrays.contains_key(aux) {
            coords.insert(aux.clone());
        }
    }

    let data_vars: Vec<IStr> = arrays
        .keys()
        .filter(|k| !coords.contains(*k))
        .cloned()
        .collect();

    Ok(ZarrDatasetMeta { arrays, dims, data_vars })
}

fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

