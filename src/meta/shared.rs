use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use zarrs::array::{
    Array, ArrayMetadata, ArrayShardedExt,
    ChunkGrid,
};
use zarrs::hierarchy::NodeMetadata;

use crate::errors::{
    BackendError, BackendResult,
};
use crate::meta::dims::leaf_name;
use crate::meta::path::ZarrPath;

use crate::meta::types::{
    ZarrArrayMeta, ZarrMeta,
};
use crate::shared::{
    IStr, IntoIStr, MaybeParIter,
};

use crate::meta::ZarrNode;

/// Build this many array nodes on the Rayon pool; below that, stay sequential to
/// avoid thread-pool overhead on tiny hierarchies.
const PARALLEL_ZARR_META_ARRAYS: usize = 2;

struct ArrayMetaLoadJob {
    traverse_idx: usize,
    path_str: String,
    array_md: ArrayMetadata,
}

struct ProcessedArrayMetaJob {
    traverse_idx: usize,
    parent_zp: ZarrPath,
    leaf: IStr,
    arr_meta: Arc<ZarrArrayMeta>,
    aux_coord_names: Vec<IStr>,
}

fn aux_coord_names_for_array<TStorage: ?Sized>(
    array: &Array<TStorage>,
) -> Vec<IStr> {
    let mut out = Vec::new();
    if let Some(attrs) =
        array.attributes().get("coordinates")
        && let Some(coord_str) = attrs.as_str()
    {
        for coord_name in
            coord_str.split_whitespace()
        {
            out.push(coord_name.istr());
        }
    }
    out
}

fn chunk_grids_for_array<TStorage: ?Sized>(
    array: &Array<TStorage>,
) -> (Arc<ChunkGrid>, Option<Arc<ChunkGrid>>) {
    let is_sharded = array.is_sharded();
    let outer_chunk_grid: Arc<ChunkGrid> =
        array.chunk_grid().clone().into();
    let inner_chunk_grid: Option<Arc<ChunkGrid>> =
        if is_sharded {
            Some(array.subchunk_grid().into())
        } else {
            None
        };

    (outer_chunk_grid, inner_chunk_grid)
}

fn process_array_meta_job<
    TStorage: ?Sized + Send + Sync,
>(
    store: &Arc<TStorage>,
    root_path_str: &str,
    job: &ArrayMetaLoadJob,
) -> Result<ProcessedArrayMetaJob, BackendError> {
    let path_str = job.path_str.as_str();
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
        Arc::clone(store),
        path_str,
        job.array_md.clone(),
    )?;

    let (outer_chunk_grid, inner_chunk_grid) =
        chunk_grids_for_array(&array);
    let arr_meta = Arc::new(ZarrArrayMeta::new(
        path_str.istr(),
        outer_chunk_grid,
        inner_chunk_grid,
        Arc::new(job.array_md.clone()),
    ));
    let aux_coord_names =
        aux_coord_names_for_array(&array);

    Ok(ProcessedArrayMetaJob {
        traverse_idx: job.traverse_idx,
        parent_zp,
        leaf,
        arr_meta,
        aux_coord_names,
    })
}

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

    if let Some(arrays) = group_arrays.get(path) {
        let mut dims_set: BTreeSet<IStr> =
            BTreeSet::new();
        let mut coord_arrays: BTreeSet<IStr> =
            BTreeSet::new();

        for (leaf, arr) in arrays {
            node.arrays
                .insert(*leaf, Arc::clone(arr));

            let arr_dims = arr.dims();
            for dim in &arr_dims {
                dims_set.insert(*dim);
            }

            if arr.is_1d()
                && arr_dims.len() == 1
                && *leaf == arr_dims[0]
            {
                coord_arrays.insert(*leaf);
            }
        }

        for aux in aux_coords {
            if node.arrays.contains_key(aux) {
                coord_arrays.insert(*aux);
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
            && let Some(child_leaf) =
                child_path.leaf()
        {
            let child_node = build_node_tree(
                child_path,
                group_arrays,
                aux_coords,
            );
            node.children
                .insert(*child_leaf, child_node);
        }
    }

    node
}

pub(crate) fn load_zarr_meta_inner<
    TStorage: ?Sized + Send + Sync,
>(
    store: &Arc<TStorage>,
    nodes: &[(
        zarrs::node::NodePath,
        NodeMetadata,
    )],
    root_path_str: &str,
) -> BackendResult<ZarrMeta> {
    let jobs: Vec<ArrayMetaLoadJob> = nodes
        .iter()
        .enumerate()
        .filter_map(
            |(traverse_idx, (path, md))| {
                if let NodeMetadata::Array(
                    array_md,
                ) = md
                {
                    Some(ArrayMetaLoadJob {
                        traverse_idx,
                        path_str: path
                            .as_str()
                            .to_string(),
                        array_md: array_md
                            .clone(),
                    })
                } else {
                    None
                }
            },
        )
        .collect();

    let mut processed: Vec<
        ProcessedArrayMetaJob,
    > = jobs
        .maybe_par_iter(PARALLEL_ZARR_META_ARRAYS)
        .map_collect(|job| {
            process_array_meta_job(
                store,
                root_path_str,
                job,
            )
        })?;

    processed.sort_by(|a, b| {
        a.parent_zp.cmp(&b.parent_zp).then(
            a.traverse_idx.cmp(&b.traverse_idx),
        )
    });

    let mut group_arrays: BTreeMap<
        ZarrPath,
        Vec<(IStr, Arc<ZarrArrayMeta>)>,
    > = BTreeMap::new();
    let mut aux_coords: BTreeSet<IStr> =
        BTreeSet::new();

    for job in processed {
        for c in &job.aux_coord_names {
            aux_coords.insert(*c);
        }
        group_arrays
            .entry(job.parent_zp)
            .or_default()
            .push((job.leaf, job.arr_meta));
    }

    let root_node = build_node_tree(
        &ZarrPath::root(),
        &group_arrays,
        &aux_coords,
    );

    Ok(root_node)
}
