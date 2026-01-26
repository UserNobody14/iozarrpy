use std::collections::BTreeMap;

use zarrs::array::Array;

use crate::chunk_plan::CompileError;
use super::{DatasetSelection, DSelection};

use crate::meta::ZarrDatasetMeta;
use crate::{IStr, IntoIStr};


pub(crate) fn plan_dataset_chunk_indices(
    selection: &DatasetSelection,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
    include_dim_coords: bool,
) -> Result<BTreeMap<IStr, Vec<Vec<u64>>>, CompileError> {
    // let selection = if include_dim_coords {
    //     add_dim_coords(selection, meta)
    // } else {
    //     selection.clone()
    // };

    let mut out: BTreeMap<IStr, Vec<Vec<u64>>> = BTreeMap::new();
    for (var, sel) in selection.vars() {
        let var_key = var.istr();
        let Some(var_meta) = meta.arrays.get(&var_key) else {
            continue;
        };
        let arr = Array::open(store.clone(), var_meta.path.as_ref()).map_err(|e| {
            CompileError::Unsupported(format!("failed to open array '{var}': {e}"))
        })?;
        // let grid_shape = arr.chunk_grid().grid_shape().to_vec();
        // let zero = vec![0u64; arr.dimensionality()];
        // let chunk_shape_nz = arr
        //     .chunk_shape(&zero)
        //     .map_err(|e| CompileError::Unsupported(e.to_string()))?;
        // let chunk_shape = chunk_shape_nz.iter().map(|nz| nz.get()).collect::<Vec<_>>();

        // let chunk_set = plan_data_array_chunk_indices(
        //     sel,
        //     &var_meta.dims,
        //     &var_meta.shape,

        // );
        let mut chunk_set = Vec::new();
        for subset in sel.subsets_iter() {
            let chunks = arr.chunks_in_array_subset(subset).map_err(|e| CompileError::Unsupported(e.to_string()))?;
            if let Some(chunks) = chunks {
                for chunk_index in chunks.indices().iter() {
                    // chunk_index is a TinyVec<[u64; 4]>
                    chunk_set.push(chunk_index.iter().copied().collect());
                }
            }
        }
        out.insert(var_key, chunk_set);
    }

    Ok(out)
}

