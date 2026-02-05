use std::collections::{BTreeMap, BTreeSet};

use zarrs::array::Array;
use zarrs::array::ArrayShardedExt;
use zarrs::hierarchy::NodeMetadata;
use zarrs::storage::AsyncReadableWritableListableStorage;

use crate::meta::dims::{
    default_dims, dims_for_array, leaf_name,
};
use crate::meta::dtype::zarr_dtype_to_polars;
use crate::meta::time_encoding::extract_time_encoding;
use crate::meta::types::{
    DimensionAnalysis, ZarrArrayMeta, ZarrMeta,
    ZarrNode,
};
use crate::store::AsyncOpenedStore;
use crate::{IStr, IntoIStr};

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

        // Extract inner chunk shape (for sharded arrays) or regular chunk shape
        let zero_idx: Vec<u64> =
            vec![0u64; array.dimensionality()];
        let inner_grid = array.inner_chunk_grid();
        let chunk_shape: std::sync::Arc<[u64]> =
            inner_grid
                .chunk_shape_u64(&zero_idx)
                .ok()
                .flatten()
                .map(|cs| cs.into())
                .unwrap_or_else(|| shape.clone());

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
            chunk_shape,
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

fn to_string_err<E: std::fmt::Display>(
    e: E,
) -> String {
    e.to_string()
}

/// Load unified metadata from a raw async store and root path.
///
/// This is useful for backends like Icechunk that don't use `AsyncOpenedStore`.
pub async fn load_zarr_meta_from_store_async(
    store: &AsyncReadableWritableListableStorage,
    root_path: &str,
) -> Result<ZarrMeta, String> {
    let store = store.clone();

    let group = zarrs::group::Group::async_open(
        store.clone(),
        root_path,
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
        let rel_path = if root_path != "/"
            && path_str.starts_with(root_path)
        {
            let stripped =
                &path_str[root_path.len()..];
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

        // Extract inner chunk shape (for sharded arrays) or regular chunk shape
        let zero_idx: Vec<u64> =
            vec![0u64; array.dimensionality()];
        let inner_grid = array.inner_chunk_grid();
        let chunk_shape: std::sync::Arc<[u64]> =
            inner_grid
                .chunk_shape_u64(&zero_idx)
                .ok()
                .flatten()
                .map(|cs| cs.into())
                .unwrap_or_else(|| shape.clone());

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
            chunk_shape,
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
    let root_path_check: &str = "/";
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
        if parent_path == root_path_check {
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
