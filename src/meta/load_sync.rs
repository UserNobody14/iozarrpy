use std::collections::{BTreeMap, BTreeSet};

use zarrs::array::Array;
use zarrs::hierarchy::NodeMetadata;

use crate::meta::dims::{default_dims, dims_for_array, leaf_name};
use crate::meta::dtype::zarr_dtype_to_polars;
use crate::meta::time_encoding::extract_time_encoding;
use crate::meta::types::{ZarrArrayMeta, ZarrDatasetMeta};
use crate::store::{open_store, OpenedStore, StoreInput};

pub fn open_and_load_dataset_meta(
    zarr_url: &str,
) -> Result<(OpenedStore, ZarrDatasetMeta), String> {
    let opened = open_store(zarr_url)?;
    let meta = load_dataset_meta_from_opened(&opened)?;
    Ok((opened, meta))
}

/// Open and load dataset metadata from a StoreInput (URL string or ObjectStore instance).
pub fn open_and_load_dataset_meta_from_input(
    store_input: StoreInput,
) -> Result<(OpenedStore, ZarrDatasetMeta), String> {
    let opened = store_input.open_sync()?;
    let meta = load_dataset_meta_from_opened(&opened)?;
    Ok((opened, meta))
}

pub fn load_dataset_meta_from_opened(opened: &OpenedStore) -> Result<ZarrDatasetMeta, String> {
    let store = opened.store.clone();
    let root = opened.root.clone();

    let group = zarrs::group::Group::open(store.clone(), &root).map_err(to_string_err)?;
    let nodes = group.traverse().map_err(to_string_err)?;

    let mut arrays: BTreeMap<String, ZarrArrayMeta> = BTreeMap::new();
    let mut seen_names: BTreeMap<String, usize> = BTreeMap::new();
    let mut dims_seen: BTreeSet<String> = BTreeSet::new();
    let mut dims_ordered: Vec<String> = Vec::new();
    let mut coord_candidates = BTreeMap::new();
    let mut primary_dims: Option<Vec<String>> = None;
    let mut max_ndim = 0usize;
    // Collect auxiliary coordinates from "coordinates" attribute (CF convention)
    let mut aux_coords: BTreeSet<String> = BTreeSet::new();

    for (path, md) in nodes {
        if !matches!(md, NodeMetadata::Array(_)) {
            continue;
        }

        let path_str = path.as_str().to_string();
        let leaf = leaf_name(&path_str);

        let array = Array::open(store.clone(), &path_str).map_err(to_string_err)?;
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

        // Parse "coordinates" attribute (CF convention) to identify auxiliary coords
        if let Some(attrs) = array.attributes().get("coordinates") {
            if let Some(coord_str) = attrs.as_str() {
                for coord_name in coord_str.split_whitespace() {
                    aux_coords.insert(coord_name.to_string());
                }
            }
        }

        let polars_dtype = zarr_dtype_to_polars(array.data_type().identifier(), time_encoding.as_ref());

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

    // Collect dimension coordinate arrays (1D arrays matching their dimension name)
    let mut coords: BTreeSet<String> = BTreeSet::new();
    for dim in &dims {
        if let Some((shape, _dtype)) = coord_candidates.get(dim) {
            if shape.len() == 1 {
                coords.insert(dim.clone());
            }
        }
    }

    // Add auxiliary coordinates (only those that exist as arrays)
    for aux in &aux_coords {
        if arrays.contains_key(aux) {
            coords.insert(aux.clone());
        }
    }

    let data_vars: Vec<String> = arrays
        .keys()
        .filter(|k| !coords.contains(*k))
        .cloned()
        .collect();

    Ok(ZarrDatasetMeta { arrays, dims, data_vars })
}

fn to_string_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

