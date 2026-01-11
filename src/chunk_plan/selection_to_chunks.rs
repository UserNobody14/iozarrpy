use std::collections::{BTreeMap, BTreeSet};

use zarrs::array::Array;

use super::errors::CompileError;
use super::selection::{DataArraySelection, DatasetSelection, HyperRectangleSelection, RangeList};
use super::types::DimChunkRange;
use crate::meta::ZarrDatasetMeta;


fn chunk_ranges_for_range_list(
    ranges: &RangeList,
    dim_len: u64,
    chunk_size: u64,
    grid_dim: u64,
) -> Vec<DimChunkRange> {
    let chunk_size = chunk_size.max(1);
    let clamped = ranges.clamp_to_len(dim_len);
    let mut out: Vec<DimChunkRange> = Vec::new();
    for r in clamped.ranges() {
        if r.is_empty() {
            continue;
        }
        let chunk_start = r.start / chunk_size;
        let last = r.end_exclusive.saturating_sub(1);
        let chunk_end = last / chunk_size;
        if chunk_start >= grid_dim {
            continue;
        }
        let end = chunk_end.min(grid_dim.saturating_sub(1));
        out.push(DimChunkRange {
            start_chunk: chunk_start,
            end_chunk_inclusive: end,
        });
    }
    out.sort_by_key(|r| r.start_chunk);

    // Merge overlapping/adjacent.
    let mut merged: Vec<DimChunkRange> = Vec::with_capacity(out.len());
    for r in out {
        if let Some(last) = merged.last_mut() {
            if r.start_chunk <= last.end_chunk_inclusive.saturating_add(1) {
                last.end_chunk_inclusive = last.end_chunk_inclusive.max(r.end_chunk_inclusive);
                continue;
            }
        }
        merged.push(r);
    }
    merged
}

fn insert_cartesian_ranges(
    dim_ranges: &[Vec<DimChunkRange>],
    dim_idx: usize,
    cur: &mut Vec<u64>,
    out: &mut BTreeSet<Vec<u64>>,
) {
    if dim_idx == dim_ranges.len() {
        out.insert(cur.clone());
        return;
    }
    for r in &dim_ranges[dim_idx] {
        for c in r.start_chunk..=r.end_chunk_inclusive {
            cur.push(c);
            insert_cartesian_ranges(dim_ranges, dim_idx + 1, cur, out);
            cur.pop();
        }
    }
}

pub(super) fn plan_data_array_chunk_indices(
    sel: &DataArraySelection,
    array_dims: &[String],
    array_shape: &[u64],
    grid_shape: &[u64],
    chunk_shape: &[u64],
) -> BTreeSet<Vec<u64>> {
    let mut out: BTreeSet<Vec<u64>> = BTreeSet::new();
    if sel.is_empty() {
        return out;
    }
    if array_dims.len() != array_shape.len()
        || array_dims.len() != grid_shape.len()
        || array_dims.len() != chunk_shape.len()
    {
        return out;
    }

    for rect in &sel.0 {
        if rect.is_empty() {
            continue;
        }

        let mut per_dim: Vec<Vec<DimChunkRange>> = Vec::with_capacity(array_dims.len());
        let mut any_empty = false;
        for (i, dim) in array_dims.iter().enumerate() {
            let rl = rect.get_dim(dim).cloned().unwrap_or_else(RangeList::all);
            let ranges = chunk_ranges_for_range_list(&rl, array_shape[i], chunk_shape[i], grid_shape[i]);
            if ranges.is_empty() {
                any_empty = true;
                break;
            }
            per_dim.push(ranges);
        }
        if any_empty {
            continue;
        }

        insert_cartesian_ranges(&per_dim, 0, &mut Vec::with_capacity(array_dims.len()), &mut out);
    }

    out
}

fn project_dim_range(selection: &DatasetSelection, dim: &str) -> RangeList {
    let mut acc = RangeList::empty();
    for (_var, da) in selection.vars() {
        for rect in &da.0 {
            if rect.is_empty() {
                continue;
            }
            let rl = rect.get_dim(dim).cloned().unwrap_or_else(RangeList::all);
            acc = acc.union(&rl);
        }
    }
    acc
}

fn add_dim_coords(selection: &DatasetSelection, meta: &ZarrDatasetMeta) -> DatasetSelection {
    if selection.0.is_empty() {
        return selection.clone();
    }
    let mut out = selection.clone();
    for dim in &meta.dims {
        if out.0.contains_key(dim) {
            continue;
        }
        let Some(arr) = meta.arrays.get(dim) else {
            continue;
        };
        // Only auto-include 1D dimension coordinate arrays.
        if arr.dims.len() != 1 || arr.shape.len() != 1 || arr.dims[0] != *dim {
            continue;
        }
        let rl = project_dim_range(selection, dim);
        let da = if rl == RangeList::all() {
            DataArraySelection::all()
        } else if rl.is_empty() {
            DataArraySelection::empty()
        } else {
            DataArraySelection(vec![HyperRectangleSelection::all().with_dim(dim.clone(), rl)])
        };
        if !da.is_empty() {
            out.0.insert(dim.clone(), da);
        }
    }
    out
}

pub(crate) fn plan_dataset_chunk_indices(
    selection: &DatasetSelection,
    meta: &ZarrDatasetMeta,
    store: zarrs::storage::ReadableWritableListableStorage,
    include_dim_coords: bool,
) -> Result<BTreeMap<String, Vec<Vec<u64>>>, CompileError> {
    let selection = if include_dim_coords {
        add_dim_coords(selection, meta)
    } else {
        selection.clone()
    };

    let mut out: BTreeMap<String, Vec<Vec<u64>>> = BTreeMap::new();
    for (var, sel) in selection.vars() {
        let Some(var_meta) = meta.arrays.get(var) else {
            continue;
        };
        let arr = Array::open(store.clone(), &var_meta.path).map_err(|e| {
            CompileError::Unsupported(format!("failed to open array '{var}': {e}"))
        })?;
        let grid_shape = arr.chunk_grid().grid_shape().to_vec();
        let zero = vec![0u64; arr.dimensionality()];
        let chunk_shape_nz = arr
            .chunk_shape(&zero)
            .map_err(|e| CompileError::Unsupported(e.to_string()))?;
        let chunk_shape = chunk_shape_nz.iter().map(|nz| nz.get()).collect::<Vec<_>>();

        let chunk_set = plan_data_array_chunk_indices(
            sel,
            &var_meta.dims,
            &var_meta.shape,
            &grid_shape,
            &chunk_shape,
        );
        if !chunk_set.is_empty() {
            out.insert(var.to_string(), chunk_set.into_iter().collect());
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_plan::selection::{DataArraySelection, HyperRectangleSelection, RangeList};
    use crate::chunk_plan::types::IndexRange;

    #[test]
    fn plan_data_array_chunks_1d_range() {
        // Dim length 100, chunk size 10 => 10 chunks.
        // Select [15, 25) => chunks 1 and 2.
        let sel = DataArraySelection(vec![HyperRectangleSelection::all().with_dim(
            "x".to_string(),
            RangeList::from_index_range(IndexRange {
                start: 15,
                end_exclusive: 25,
            }),
        )]);

        let out = plan_data_array_chunk_indices(
            &sel,
            &["x".to_string()],
            &[100],
            &[10],
            &[10],
        );
        let got: Vec<Vec<u64>> = out.into_iter().collect();
        assert_eq!(got, vec![vec![1], vec![2]]);
    }
}

