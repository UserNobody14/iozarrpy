use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use zarrs::array::{Array, ArrayShardedExt};
use zarrs::hierarchy::NodeMetadata;

use crate::errors::BackendResult;
use crate::meta::dims::{
    default_dims, dims_for_array, leaf_name,
};
use crate::meta::dtype::zarr_dtype_to_polars;
use crate::meta::path::ZarrPath;

use crate::meta::time_encoding::extract_var_encoding;
use crate::meta::types::{
    DimensionAnalysis, ZarrArrayMeta, ZarrMeta,
};
use crate::{IStr, IntoIStr};

use crate::meta::ZarrNode;

/// Recursively build ZarrNode tree from grouped arrays.
/// Keys in `group_arrays` are `ZarrPath` representing group positions.
pub(crate) fn build_node_tree(
    path: &ZarrPath,
    group_arrays: &BTreeMap<
        ZarrPath,
        Vec<(IStr, Arc<ZarrArrayMeta>)>,
    >,
    aux_coords: &BTreeSet<IStr>,
) -> ZarrNode {
    let path_istr = if path.is_root() {
        "/".istr()
    } else {
        format!("/{}", path.to_flat_string())
            .istr()
    };
    let mut node = ZarrNode::new(path_istr);

    if let Some(arrays) =
        group_arrays.get(path)
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

            for dim in &arr.dims {
                dims_set.insert(dim.clone());
            }

            if arr.shape.len() == 1
                && arr.dims.len() == 1
                && *leaf == arr.dims[0]
            {
                coord_arrays
                    .insert(leaf.clone());
            }
        }

        for aux in aux_coords {
            if node.arrays.contains_key(aux) {
                coord_arrays
                    .insert(aux.clone());
            }
        }

        node.local_dims =
            dims_set.into_iter().collect();

        node.data_vars = node
            .arrays
            .keys()
            .filter(|k| {
                !coord_arrays.contains(*k)
            })
            .cloned()
            .collect();
    }

    // Find direct children: paths whose parent == current path
    for child_path in group_arrays.keys() {
        if child_path.parent() == *path
            && child_path != path
        {
            if let Some(child_leaf) =
                child_path.leaf()
            {
                let child_node = build_node_tree(
                    child_path,
                    group_arrays,
                    aux_coords,
                );
                node.children.insert(
                    child_leaf.clone(),
                    child_node,
                );
            }
        }
    }

    node
}

pub(crate) fn load_zarr_meta_inner<
    TStorage: ?Sized,
>(
    store: &Arc<TStorage>,
    nodes: &Vec<(
        zarrs::node::NodePath,
        NodeMetadata,
    )>,
    root_path_str: &str,
) -> BackendResult<ZarrMeta> {
    let mut group_arrays: BTreeMap<
        ZarrPath,
        Vec<(IStr, Arc<ZarrArrayMeta>)>,
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

        let rel_zp = ZarrPath::parse(rel_path);
        let leaf = leaf_name(rel_path);
        let parent_zp = rel_zp.parent();

        let array = Array::new_with_metadata(
            store.clone(),
            path_str,
            array_md.clone(),
        )?;

        let shape: std::sync::Arc<[u64]> =
            array.shape().into();
        let dims = dims_for_array(&array)
            .unwrap_or_else(|| {
                default_dims(shape.len())
            });
        let encoding =
            extract_var_encoding(&array);
        let polars_dtype = zarr_dtype_to_polars(
            array.data_type(),
            encoding.as_ref(),
        );

        let zero_idx: Vec<u64> =
            vec![0u64; array.dimensionality()];
        let inner_grid = array.subchunk_grid();
        let chunk_shape: std::sync::Arc<[u64]> =
            inner_grid
                .chunk_shape_u64(&zero_idx)
                .ok()
                .flatten()
                .map(|cs| cs.into())
                .unwrap_or_else(|| shape.clone());

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
            chunk_grid: array
                .subchunk_grid()
                .into(),
            dims,
            polars_dtype,
            encoding,
            array_metadata: Some(Arc::new(
                array_md.clone(),
            )),
        };

        let arr_meta_arc = Arc::new(arr_meta);

        group_arrays
            .entry(parent_zp)
            .or_default()
            .push((leaf, arr_meta_arc.clone()));
    }

    let root_node = build_node_tree(
        &ZarrPath::root(),
        &group_arrays,
        &aux_coords,
    );

    let dim_analysis =
        DimensionAnalysis::compute(&root_node);

    Ok(ZarrMeta {
        root: root_node,
        dim_analysis,
    })
}
