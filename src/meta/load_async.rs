use std::collections::{BTreeMap, BTreeSet};

use smallvec::SmallVec;
use zarrs::array::Array;
use zarrs::hierarchy::NodeMetadata;

use crate::meta::dims::{
    default_dims, dims_for_array, leaf_name,
};
use crate::meta::dtype::zarr_dtype_to_polars;
use crate::meta::time_encoding::extract_time_encoding;
use crate::meta::types::{
    DimensionAnalysis, ZarrArrayMeta,
    ZarrDatasetMeta, ZarrMeta, ZarrNode,
};
use crate::store::{
    AsyncOpenedStore, StoreInput,
    open_store_async,
};
use crate::{IStr, IntoIStr};

pub async fn open_and_load_dataset_meta_async(
    zarr_url: &str,
) -> Result<
    (AsyncOpenedStore, ZarrDatasetMeta),
    String,
> {
    let opened = open_store_async(zarr_url)?;
    let meta =
        load_dataset_meta_from_opened_async(
            &opened,
        )
        .await?;
    Ok((opened, meta))
}

/// Open and load dataset metadata from a StoreInput (URL string or ObjectStore instance).
pub async fn open_and_load_dataset_meta_from_input_async(
    store_input: StoreInput,
) -> Result<
    (AsyncOpenedStore, ZarrDatasetMeta),
    String,
> {
    let opened = store_input.open_async()?;
    let meta =
        load_dataset_meta_from_opened_async(
            &opened,
        )
        .await?;
    Ok((opened, meta))
}

pub async fn load_dataset_meta_from_opened_async(
    opened: &AsyncOpenedStore,
) -> Result<ZarrDatasetMeta, String> {
    let store = opened.store.clone();
    let root = opened.root.clone();

    let group = zarrs::group::Group::async_open(
        store.clone(),
        &root,
    )
    .await
    .map_err(to_string_err)?;
    let nodes = group
        .async_traverse()
        .await
        .map_err(to_string_err)?;

    let mut arrays: BTreeMap<
        IStr,
        ZarrArrayMeta,
    > = BTreeMap::new();
    let mut seen_names: BTreeMap<IStr, usize> =
        BTreeMap::new();
    let mut dims_seen: BTreeSet<IStr> =
        BTreeSet::new();
    let mut dims_ordered: Vec<IStr> = Vec::new();
    let mut coord_candidates: BTreeMap<
        IStr,
        (Vec<u64>, _),
    > = BTreeMap::new();
    let mut primary_dims: Option<
        SmallVec<[IStr; 4]>,
    > = None;
    let mut max_ndim = 0usize;
    // Collect auxiliary coordinates from "coordinates" attribute (CF convention)
    let mut aux_coords: BTreeSet<IStr> =
        BTreeSet::new();

    for (path, md) in nodes {
        let NodeMetadata::Array(array_md) = md
        else {
            continue;
        };

        let path_str = path.as_str().istr();
        let leaf = leaf_name(path_str.as_ref());

        let array = Array::new_with_metadata(
            store.clone(),
            path_str.as_ref(),
            array_md.clone(),
        )
        .map_err(to_string_err)?;

        let shape: std::sync::Arc<[u64]> =
            array.shape().into();
        let dims = dims_for_array(&array)
            .unwrap_or_else(|| {
                default_dims(shape.len())
            });

        for d in &dims {
            if dims_seen.insert(d.clone()) {
                dims_ordered.push(d.clone());
            }
        }
        if dims.len() > max_ndim {
            max_ndim = dims.len();
            primary_dims = Some(dims.clone());
        }

        let time_encoding =
            extract_time_encoding(&array);

        if shape.len() == 1
            && dims.len() == 1
            && leaf == dims[0]
        {
            let dt = zarr_dtype_to_polars(
                array.data_type().identifier(),
                time_encoding.as_ref(),
            );
            coord_candidates.insert(
                leaf.clone(),
                (shape.to_vec(), dt),
            );
        }

        // Parse "coordinates" attribute (CF convention) to identify auxiliary coords
        if let Some(attrs) =
            array.attributes().get("coordinates")
            && let Some(coord_str) =
                attrs.as_str()
        {
            for coord_name in
                coord_str.split_whitespace()
            {
                aux_coords
                    .insert(coord_name.istr());
            }
        }

        let polars_dtype = zarr_dtype_to_polars(
            array.data_type().identifier(),
            time_encoding.as_ref(),
        );

        let name = match seen_names.get_mut(&leaf)
        {
            None => {
                seen_names
                    .insert(leaf.clone(), 1);
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
    let mut coords: BTreeSet<IStr> =
        BTreeSet::new();
    for dim in &dims {
        if let Some((shape, _dtype)) =
            coord_candidates.get(dim)
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

    Ok(ZarrDatasetMeta {
        arrays,
        dims,
        data_vars,
    })
}

// =============================================================================
// Unified Hierarchical Metadata Loading
// =============================================================================

/// Load unified metadata that supports both flat and hierarchical zarr stores.
pub async fn load_zarr_meta_from_opened_async(
    opened: &AsyncOpenedStore,
) -> Result<ZarrMeta, String> {
    let store = opened.store.clone();
    let root_path = opened.root.clone();
    let root_path_str: &str = root_path.as_ref();

    let group = zarrs::group::Group::async_open(
        store.clone(),
        &root_path,
    )
    .await
    .map_err(to_string_err)?;
    let nodes = group
        .async_traverse()
        .await
        .map_err(to_string_err)?;

    // First pass: collect all arrays and identify group structure
    let mut all_arrays: BTreeMap<
        IStr,
        ZarrArrayMeta,
    > = BTreeMap::new();
    let mut group_arrays: BTreeMap<
        IStr,
        Vec<(IStr, ZarrArrayMeta)>,
    > = BTreeMap::new();
    let mut aux_coords: BTreeSet<IStr> =
        BTreeSet::new();

    for (path, md) in nodes {
        let NodeMetadata::Array(array_md) = md
        else {
            continue;
        };

        let path_str = path.as_str();
        let rel_path = if root_path_str != "/"
            && path_str.starts_with(root_path_str)
        {
            let stripped =
                &path_str[root_path_str.len()..];
            if stripped.is_empty() {
                "/"
            } else {
                stripped
            }
        } else {
            path_str
        };
        let leaf = leaf_name(rel_path);

        // Determine the parent group path
        let parent_path =
            parent_group_path(rel_path);

        let array = Array::new_with_metadata(
            store.clone(),
            path_str,
            array_md.clone(),
        )
        .map_err(to_string_err)?;

        let shape: std::sync::Arc<[u64]> =
            array.shape().into();
        let dims = dims_for_array(&array)
            .unwrap_or_else(|| {
                default_dims(shape.len())
            });
        let time_encoding =
            extract_time_encoding(&array);
        let polars_dtype = zarr_dtype_to_polars(
            array.data_type().identifier(),
            time_encoding.as_ref(),
        );

        // Parse "coordinates" attribute (CF convention) to identify auxiliary coords
        if let Some(attrs) =
            array.attributes().get("coordinates")
        {
            if let Some(coord_str) =
                attrs.as_str()
            {
                for coord_name in
                    coord_str.split_whitespace()
                {
                    aux_coords.insert(
                        coord_name.istr(),
                    );
                }
            }
        }

        let arr_meta = ZarrArrayMeta {
            path: path_str.istr(),
            shape,
            dims,
            polars_dtype,
            time_encoding,
        };

        // Store in both flat and grouped maps
        all_arrays.insert(
            rel_path.istr(),
            arr_meta.clone(),
        );
        group_arrays
            .entry(parent_path)
            .or_default()
            .push((leaf, arr_meta));
    }

    // Build the hierarchical node structure
    let root_node = build_node_tree(
        "/".istr(),
        &group_arrays,
        &aux_coords,
    );

    // Build path_to_array map with both full paths and leaf names for root arrays
    let mut path_to_array: BTreeMap<
        IStr,
        ZarrArrayMeta,
    > = BTreeMap::new();
    let root_path: &str = "/";
    for (path, arr) in &all_arrays {
        // Store by full path
        path_to_array
            .insert(path.clone(), arr.clone());

        // For root-level arrays, also store by leaf name
        let path_str: &str = path.as_ref();
        let parent_istr =
            parent_group_path(path_str);
        let parent_path: &str =
            parent_istr.as_ref();
        if parent_path == root_path {
            let leaf = leaf_name(path_str);
            path_to_array
                .insert(leaf, arr.clone());
        }
    }

    // Compute dimension analysis
    let dim_analysis =
        DimensionAnalysis::compute(&root_node);

    // For flat datasets (no children), allow leaf-name lookup even if paths are nested.
    if root_node.children.is_empty() {
        for (path, arr) in &all_arrays {
            let leaf = leaf_name(path.as_ref());
            path_to_array
                .entry(leaf)
                .or_insert_with(|| arr.clone());
        }
    }

    Ok(ZarrMeta {
        root: root_node,
        dim_analysis,
        path_to_array,
    })
}

/// Extract the parent group path from a full array path.
/// E.g., "/model_a/temperature" -> "/model_a", "/temperature" -> "/"
fn parent_group_path(path: &str) -> IStr {
    let path = path.trim_start_matches('/');
    if let Some(pos) = path.rfind('/') {
        format!("/{}", &path[..pos]).istr()
    } else {
        "/".istr()
    }
}

/// Recursively build ZarrNode tree from grouped arrays.
fn build_node_tree(
    path: IStr,
    group_arrays: &BTreeMap<
        IStr,
        Vec<(IStr, ZarrArrayMeta)>,
    >,
    aux_coords: &BTreeSet<IStr>,
) -> ZarrNode {
    let mut node = ZarrNode::new(path.clone());

    // Add arrays directly in this group
    if let Some(arrays) = group_arrays.get(&path)
    {
        let mut dims_set: BTreeSet<IStr> =
            BTreeSet::new();
        let mut coord_arrays: BTreeSet<IStr> =
            BTreeSet::new();

        for (leaf, arr) in arrays {
            node.arrays.insert(
                leaf.clone(),
                arr.clone(),
            );

            // Collect dimensions
            for dim in &arr.dims {
                dims_set.insert(dim.clone());
            }

            // Identify coordinate arrays (1D arrays matching their dimension name)
            if arr.shape.len() == 1
                && arr.dims.len() == 1
                && *leaf == arr.dims[0]
            {
                coord_arrays.insert(leaf.clone());
            }
        }

        // Add auxiliary coords that exist as arrays
        for aux in aux_coords {
            if node.arrays.contains_key(aux) {
                coord_arrays.insert(aux.clone());
            }
        }

        // Local dims are all unique dimensions used by arrays in this node
        node.local_dims =
            dims_set.into_iter().collect();

        // Data vars are non-coordinate arrays
        node.data_vars = node
            .arrays
            .keys()
            .filter(|k| {
                !coord_arrays.contains(*k)
            })
            .cloned()
            .collect();
    }

    // Find and build child groups
    let path_str: &str = path.as_ref();
    let path_prefix = if path_str == "/" {
        "/".to_string()
    } else {
        format!("{}/", path_str)
    };

    for child_path in group_arrays.keys() {
        let child_path_str: &str =
            child_path.as_ref();

        // Check if this is a direct child of the current path
        if child_path_str
            .starts_with(&path_prefix)
            && child_path_str != path_str
        {
            let remainder = &child_path_str
                [path_prefix.len()..];
            // Direct child has no more slashes in the remainder
            if !remainder.contains('/') {
                let child_name = remainder.istr();
                let child_node = build_node_tree(
                    child_path.clone(),
                    group_arrays,
                    aux_coords,
                );
                node.children.insert(
                    child_name, child_node,
                );
            }
        }
    }

    node
}

/// Open and load unified metadata (async).
pub async fn open_and_load_zarr_meta_async(
    zarr_url: &str,
) -> Result<(AsyncOpenedStore, ZarrMeta), String>
{
    let opened = open_store_async(zarr_url)?;
    let meta =
        load_zarr_meta_from_opened_async(&opened)
            .await?;
    Ok((opened, meta))
}

/// Open and load unified metadata from a StoreInput (async).
pub async fn open_and_load_zarr_meta_from_input_async(
    store_input: StoreInput,
) -> Result<(AsyncOpenedStore, ZarrMeta), String>
{
    let opened = store_input.open_async()?;
    let meta =
        load_zarr_meta_from_opened_async(&opened)
            .await?;
    Ok((opened, meta))
}

fn to_string_err<E: std::fmt::Display>(
    e: E,
) -> String {
    e.to_string()
}
