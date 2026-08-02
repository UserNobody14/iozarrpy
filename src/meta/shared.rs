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
    /// Absolute path in the store, e.g. `/root/group/var`.
    path: IStr,
    parent_zp: ZarrPath,
    leaf: IStr,
    array_md: Arc<ArrayMetadata>,
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

/// Position of a node relative to the opened root, e.g. `/root/a/var` under
/// root `/root` becomes `a/var`.
fn path_relative_to_root(
    root_path_str: &str,
    path_str: &str,
) -> ZarrPath {
    let rel = if root_path_str == "/" {
        path_str
    } else {
        path_str
            .strip_prefix(root_path_str)
            .unwrap_or(path_str)
    };
    ZarrPath::parse(rel)
}

/// `ZarrNode::path` form of a hierarchy position: absolute, slash-prefixed.
fn node_path_istr(path: &ZarrPath) -> IStr {
    if path.is_root() {
        "/".istr()
    } else {
        format!("/{}", path.to_flat_string())
            .istr()
    }
}

fn process_array_meta_job<
    TStorage: ?Sized + Send + Sync,
>(
    store: &Arc<TStorage>,
    job: &ArrayMetaLoadJob,
) -> Result<ProcessedArrayMetaJob, BackendError> {
    let array = Array::new_with_metadata(
        Arc::clone(store),
        job.path.as_ref(),
        (*job.array_md).clone(),
    )?;

    let (outer_chunk_grid, inner_chunk_grid) =
        chunk_grids_for_array(&array);

    Ok(ProcessedArrayMetaJob {
        traverse_idx: job.traverse_idx,
        parent_zp: job.parent_zp.clone(),
        leaf: job.leaf,
        arr_meta: Arc::new(ZarrArrayMeta::new(
            job.path,
            outer_chunk_grid,
            inner_chunk_grid,
            Arc::clone(&job.array_md),
        )),
        aux_coord_names:
            aux_coord_names_for_array(&array),
    })
}

/// Walk to the node at `path`, creating any missing node along the way so that
/// groups holding no arrays of their own still link the root to descendants.
fn node_at_path<'a>(
    root: &'a mut ZarrNode,
    path: &ZarrPath,
) -> &'a mut ZarrNode {
    let mut node = root;
    let mut prefix = ZarrPath::root();
    for component in path.components() {
        prefix = prefix.push(*component);
        node = node
            .children
            .entry(*component)
            .or_insert_with(|| {
                ZarrNode::new(node_path_istr(
                    &prefix,
                ))
            });
    }
    node
}

/// Derive `local_dims` and `data_vars` from the arrays already on `node`.
///
/// An array is a coordinate when it is 1-D and named after its own dimension,
/// or when another array lists it in its CF `coordinates` attribute.
fn classify_node_arrays(
    node: &mut ZarrNode,
    aux_coords: &BTreeSet<IStr>,
) {
    let mut dims: BTreeSet<IStr> =
        BTreeSet::new();
    let mut coord_arrays: BTreeSet<IStr> =
        BTreeSet::new();

    for (leaf, arr) in &node.arrays {
        let arr_dims = arr.dims();
        dims.extend(arr_dims.iter().copied());

        if arr.is_1d()
            && arr_dims.len() == 1
            && *leaf == arr_dims[0]
        {
            coord_arrays.insert(*leaf);
        }
    }

    coord_arrays.extend(
        aux_coords.iter().copied().filter(
            |aux| node.arrays.contains_key(aux),
        ),
    );

    node.local_dims = dims.into_iter().collect();
    node.data_vars = node
        .arrays
        .keys()
        .filter(|k| !coord_arrays.contains(*k))
        .copied()
        .collect();
}

/// Build the `ZarrNode` tree from arrays grouped by their parent group path.
fn build_node_tree(
    group_arrays: BTreeMap<
        ZarrPath,
        Vec<(IStr, Arc<ZarrArrayMeta>)>,
    >,
    aux_coords: &BTreeSet<IStr>,
) -> ZarrNode {
    let mut root = ZarrNode::new(node_path_istr(
        &ZarrPath::root(),
    ));

    for (group_path, arrays) in group_arrays {
        let node =
            node_at_path(&mut root, &group_path);
        node.arrays.extend(arrays);
        classify_node_arrays(node, aux_coords);
    }

    root
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
    let jobs: Vec<ArrayMetaLoadJob> =
        nodes
            .iter()
            .enumerate()
            .filter_map(
                |(traverse_idx, (path, md))| {
                    let NodeMetadata::Array(
                        array_md,
                    ) = md
                    else {
                        return None;
                    };
                    let path_str = path.as_str();
                    let rel =
                        path_relative_to_root(
                            root_path_str,
                            path_str,
                        );
                    Some(ArrayMetaLoadJob {
                        traverse_idx,
                        path: path_str.istr(),
                        parent_zp: rel.parent(),
                        leaf: *rel.leaf()?,
                        array_md: Arc::new(
                            array_md.clone(),
                        ),
                    })
                },
            )
            .collect();

    let mut processed: Vec<
        ProcessedArrayMetaJob,
    > = jobs
        .maybe_par_iter(PARALLEL_ZARR_META_ARRAYS)
        .map_collect(|job| {
            process_array_meta_job(store, job)
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
        aux_coords
            .extend(job.aux_coord_names.iter());
        group_arrays
            .entry(job.parent_zp)
            .or_default()
            .push((job.leaf, job.arr_meta));
    }

    Ok(build_node_tree(group_arrays, &aux_coords))
}
