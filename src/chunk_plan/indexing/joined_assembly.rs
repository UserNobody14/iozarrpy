//! Single-pass joined-batch assembler.
//!
//! Replaces the legacy per-leaf-DataFrame + polars FOJ/Left-join chain with
//! direct index-math gather. For every batch:
//!
//! - **`Single`**: vstack one leaf's chunks into a DataFrame whose dim columns
//!   are user-facing `<dim>` integer global positions. No joins, no synthetic
//!   keys.
//! - **`Joined`**: compute the per-output-dim coverage interval (union over
//!   participants on each dim), then for every (leaf, var) write the chunk's
//!   raw [`crate::reader::ColumnData`] directly into the output column at the
//!   row indices given by the chunk's hyper-rectangle in output coordinates.
//!   Output positions outside a leaf's coverage emit null. Coord leaves
//!   broadcast onto the matching `<dim>` column.
//! - **`Concat`**: diagonal-concat children (no shared keys to join on).
//!
//! Concurrency: per-(leaf, var) column construction is independent — the
//! caller can parallelize at that granularity.

use std::collections::BTreeMap;
use std::sync::Arc;

use polars::prelude::*;
use smallvec::SmallVec;
use snafu::ResultExt;

use super::grid_join_reader::{
    BatchPlan, JoinedCombine,
};
use super::plan::OwnedGridGroup;
use crate::errors::{
    BackendResult, PolarsSnafu,
};
use crate::meta::{VarEncoding, ZarrMeta};
use crate::reader::{
    ColumnData, checked_chunk_len,
    compute_strides,
};
use crate::scan::shared::{
    RawChunkRead, build_coord_column,
    build_var_column,
    chunk_read_plan::VarRead,
    columns::KeepMask, compute_in_bounds_mask,
};
use crate::shared::IStr;

/// One chunk's raw read paired with the leaf+slot that produced it.
pub struct BatchChunkRead {
    pub leaf_idx: usize,
    pub slot: usize,
    pub raw: RawChunkRead,
}

/// Assemble one batch's joined DataFrame from raw per-chunk reads.
pub fn assemble_joined_batch(
    plan: &BatchPlan,
    leaves: &[&OwnedGridGroup],
    chunk_reads: Vec<BatchChunkRead>,
    meta: &ZarrMeta,
) -> BackendResult<Option<DataFrame>> {
    let mut by_leaf: BTreeMap<
        usize,
        Vec<BatchChunkRead>,
    > = BTreeMap::new();
    for r in chunk_reads {
        by_leaf.entry(r.leaf_idx).or_default().push(r);
    }
    assemble_node(
        &plan.combine,
        leaves,
        &by_leaf,
        meta,
    )
}

fn assemble_node(
    node: &JoinedCombine,
    leaves: &[&OwnedGridGroup],
    by_leaf: &BTreeMap<usize, Vec<BatchChunkRead>>,
    meta: &ZarrMeta,
) -> BackendResult<Option<DataFrame>> {
    match node {
        JoinedCombine::Single { leaf_idx } => {
            assemble_single(
                *leaf_idx, leaves, by_leaf, meta,
            )
        }
        JoinedCombine::Joined {
            structural_leaves,
            coord_leaves,
        } => assemble_joined(
            structural_leaves,
            coord_leaves,
            leaves,
            by_leaf,
            meta,
        ),
        JoinedCombine::Concat { children } => {
            let mut child_dfs: Vec<DataFrame> =
                Vec::new();
            for c in children {
                if let Some(df) = assemble_node(
                    c, leaves, by_leaf, meta,
                )? {
                    child_dfs.push(df);
                }
            }
            match child_dfs.len() {
                0 => Ok(None),
                1 => Ok(Some(
                    child_dfs
                        .into_iter()
                        .next()
                        .unwrap(),
                )),
                _ => polars::functions::concat_df_diagonal(&child_dfs)
                    .context(PolarsSnafu {
                        message: "Error diagonal-concatenating independent subtrees".to_string(),
                    })
                    .map(Some),
            }
        }
    }
}

// =============================================================================
// Single-leaf assembly: vstack of per-chunk DataFrames with user-facing dim names
// =============================================================================

fn assemble_single(
    leaf_idx: usize,
    leaves: &[&OwnedGridGroup],
    by_leaf: &BTreeMap<usize, Vec<BatchChunkRead>>,
    meta: &ZarrMeta,
) -> BackendResult<Option<DataFrame>> {
    let leaf = leaves[leaf_idx];
    let Some(chunks) = by_leaf.get(&leaf_idx) else {
        return Ok(None);
    };
    if chunks.is_empty() {
        return Ok(None);
    }
    let dims = leaf.sig.dims();
    let chunk_shape = leaf.sig.retrieval_shape();
    let strides = compute_strides(chunk_shape);
    let chunk_len =
        checked_chunk_len(chunk_shape)?;

    let mut per_chunk_dfs: Vec<DataFrame> =
        Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let cidx = &leaf.chunk_indices[chunk.slot];
        let chunk_subset = leaf.chunk_subsets
            [chunk.slot]
            .as_ref();
        let origin: Vec<u64> = cidx
            .iter()
            .zip(chunk_shape.iter())
            .map(|(i, s)| i * s)
            .collect();
        let keep = compute_in_bounds_mask(
            chunk_len,
            chunk_shape,
            &origin,
            &leaf.array_shape,
            &strides,
            chunk_subset,
        );
        let height = keep.len();
        if height == 0 {
            continue;
        }

        let mut cols: Vec<Column> =
            Vec::with_capacity(
                dims.len()
                    + chunk.raw.var_reads.len(),
            );
        for (dim_idx, dim_name) in
            dims.iter().enumerate()
        {
            cols.push(build_coord_column(
                dim_name.as_ref(),
                dim_idx,
                &keep,
                &strides,
                chunk_shape,
                &origin,
                None,
                None,
            ));
        }
        for (vi, vr) in
            chunk.raw.var_reads.iter().enumerate()
        {
            let data = &chunk.raw.var_data[vi];
            let encoding = meta
                .array_by_path(vr.name)
                .and_then(|m| m.encoding.as_ref());
            cols.push(build_var_column(
                &vr.name,
                data,
                &vr.var_dims,
                &vr.var_chunk_shape,
                &vr.offsets,
                dims,
                chunk_shape,
                &strides,
                &keep,
                encoding,
            ));
        }
        per_chunk_dfs.push(
            DataFrame::new(height, cols).context(
                PolarsSnafu {
                    message:
                        "Error creating per-chunk DataFrame"
                            .to_string(),
                },
            )?,
        );
    }

    if per_chunk_dfs.is_empty() {
        return Ok(None);
    }
    if per_chunk_dfs.len() == 1 {
        return Ok(Some(
            per_chunk_dfs
                .into_iter()
                .next()
                .unwrap(),
        ));
    }
    let mut iter = per_chunk_dfs.into_iter();
    let first = iter.next().unwrap();
    let col_order: Vec<PlSmallStr> =
        first.get_column_names_owned();
    let mut acc = first;
    for df in iter {
        let reordered = df
            .select(col_order.as_slice())
            .context(PolarsSnafu {
                message: "Error reordering chunk columns within leaf slab".to_string(),
            })?;
        acc.vstack_mut(&reordered).context(PolarsSnafu {
            message: "Error vstacking chunk DataFrames within leaf slab".to_string(),
        })?;
    }
    Ok(Some(acc))
}

// =============================================================================
// Joined assembly: cartesian over the union of dim coverage, per-leaf gather
// =============================================================================

/// Output geometry computed once per joined batch.
struct OutputGeometry {
    /// Output dims in canonical order (driver leaf's dims first, then any
    /// extra dims contributed by sibling leaves, in encounter order).
    dims: SmallVec<[IStr; 6]>,
    /// Per output dim: sorted unique global positions covered by some
    /// participant in this batch.
    positions: SmallVec<[Vec<u64>; 6]>,
    /// Output strides in C-order (rightmost dim varies fastest).
    strides: SmallVec<[u64; 6]>,
    /// Output rows = product of `positions[d].len()`.
    rows: usize,
    /// Per output dim: a `(start, end)` range when `positions[d]` is the
    /// dense range `[start..end)`. Lets the dim-column builder skip a
    /// `Vec<u64>` materialization in the common contiguous case.
    /// `None` ⇒ sparse positions; consult `positions[d]` directly.
    contiguous: SmallVec<[Option<(u64, u64)>; 6]>,
}

fn compute_output_geometry(
    structural_leaves: &[usize],
    leaves: &[&OwnedGridGroup],
    by_leaf: &BTreeMap<usize, Vec<BatchChunkRead>>,
) -> OutputGeometry {
    let mut dims: SmallVec<[IStr; 6]> =
        SmallVec::new();
    for &leaf_idx in structural_leaves {
        for &d in leaves[leaf_idx].sig.dims() {
            if !dims.contains(&d) {
                dims.push(d);
            }
        }
    }

    let n_dims = dims.len();
    let mut interval_lo: Vec<Option<u64>> =
        vec![None; n_dims];
    let mut interval_hi: Vec<Option<u64>> =
        vec![None; n_dims];
    let mut sparse_sets: Vec<
        Option<std::collections::BTreeSet<u64>>,
    > = vec![None; n_dims];

    for &leaf_idx in structural_leaves {
        let leaf = leaves[leaf_idx];
        let Some(chunks) = by_leaf.get(&leaf_idx)
        else {
            continue;
        };
        let cs = leaf.sig.retrieval_shape();
        for chunk in chunks {
            let cidx =
                &leaf.chunk_indices[chunk.slot];
            let chunk_subset = leaf.chunk_subsets
                [chunk.slot]
                .as_ref();
            for (d_leaf, dim_name) in
                leaf.sig.dims().iter().enumerate()
            {
                let chunk_origin =
                    cidx[d_leaf] * cs[d_leaf];
                let edge_end = cs[d_leaf].min(
                    leaf.array_shape[d_leaf]
                        .saturating_sub(chunk_origin),
                );
                let (lo, hi) = chunk_subset
                    .map_or(
                        (0u64, edge_end),
                        |sub| {
                            let s =
                                sub.ranges[d_leaf].start;
                            let e = sub.ranges
                                [d_leaf]
                                .end
                                .min(edge_end);
                            (s, e)
                        },
                    );
                if lo >= hi {
                    continue;
                }
                let g_lo = chunk_origin + lo;
                let g_hi = chunk_origin + hi;
                let d_out = dims
                    .iter()
                    .position(|d| d == dim_name)
                    .expect("dim in canonical order");

                if let Some(set) =
                    &mut sparse_sets[d_out]
                {
                    for p in g_lo..g_hi {
                        set.insert(p);
                    }
                    continue;
                }

                match (
                    interval_lo[d_out],
                    interval_hi[d_out],
                ) {
                    (None, _) => {
                        interval_lo[d_out] =
                            Some(g_lo);
                        interval_hi[d_out] =
                            Some(g_hi);
                    }
                    (Some(plo), Some(phi)) => {
                        if g_lo <= phi
                            && plo <= g_hi
                        {
                            interval_lo[d_out] =
                                Some(plo.min(g_lo));
                            interval_hi[d_out] =
                                Some(phi.max(g_hi));
                        } else {
                            let mut set = std::collections::BTreeSet::new();
                            for p in plo..phi {
                                set.insert(p);
                            }
                            for p in g_lo..g_hi {
                                set.insert(p);
                            }
                            sparse_sets[d_out] =
                                Some(set);
                            interval_lo[d_out] =
                                None;
                            interval_hi[d_out] =
                                None;
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    let mut positions: SmallVec<[Vec<u64>; 6]> =
        SmallVec::with_capacity(n_dims);
    let mut contiguous: SmallVec<
        [Option<(u64, u64)>; 6],
    > = SmallVec::with_capacity(n_dims);
    for d in 0..n_dims {
        if let Some(set) = sparse_sets[d].take() {
            let v: Vec<u64> =
                set.into_iter().collect();
            positions.push(v);
            contiguous.push(None);
        } else if let (Some(lo), Some(hi)) =
            (interval_lo[d], interval_hi[d])
        {
            positions.push((lo..hi).collect());
            contiguous.push(Some((lo, hi)));
        } else {
            positions.push(Vec::new());
            contiguous.push(None);
        }
    }

    let dim_sizes: Vec<usize> =
        positions.iter().map(|p| p.len()).collect();
    let rows: usize =
        dim_sizes.iter().product();
    let mut strides: SmallVec<[u64; 6]> =
        smallvec::smallvec![0u64; n_dims];
    if n_dims > 0 {
        strides[n_dims - 1] = 1;
        for i in (0..n_dims - 1).rev() {
            strides[i] = strides[i + 1]
                * dim_sizes[i + 1] as u64;
        }
    }

    OutputGeometry {
        dims,
        positions,
        strides,
        rows,
        contiguous,
    }
}

fn assemble_joined(
    structural_leaves: &[usize],
    coord_leaves: &[usize],
    leaves: &[&OwnedGridGroup],
    by_leaf: &BTreeMap<usize, Vec<BatchChunkRead>>,
    meta: &ZarrMeta,
) -> BackendResult<Option<DataFrame>> {
    let geometry = compute_output_geometry(
        structural_leaves,
        leaves,
        by_leaf,
    );
    if geometry.rows == 0 {
        return Ok(None);
    }

    let mut coord_for_dim: BTreeMap<IStr, usize> =
        BTreeMap::new();
    for &c in coord_leaves {
        coord_for_dim
            .insert(leaves[c].sig.dims()[0], c);
    }

    let mut columns: Vec<Column> = Vec::new();

    for (d_out, &dim_name) in
        geometry.dims.iter().enumerate()
    {
        if let Some(&c_idx) =
            coord_for_dim.get(&dim_name)
        {
            columns.push(
                build_dim_column_from_coord(
                    dim_name,
                    c_idx,
                    leaves,
                    by_leaf,
                    &geometry,
                    d_out,
                    meta,
                ),
            );
        } else {
            columns.push(
                build_dim_column_from_positions(
                    dim_name.as_ref(),
                    &geometry,
                    d_out,
                ),
            );
        }
    }

    for &leaf_idx in structural_leaves {
        let leaf = leaves[leaf_idx];
        let Some(chunks) = by_leaf.get(&leaf_idx)
        else {
            continue;
        };
        if chunks.is_empty() {
            continue;
        }
        for col in build_leaf_var_columns(
            leaf, chunks, &geometry, meta,
        ) {
            columns.push(col);
        }
    }

    DataFrame::new(geometry.rows, columns)
        .context(PolarsSnafu {
            message:
                "Error creating joined DataFrame"
                    .to_string(),
        })
        .map(Some)
}

// =============================================================================
// Dim columns
// =============================================================================

/// Integer dim column: each output row carries the global integer position
/// of that row's dim value. Built via `repeat_tile` on the position list.
fn build_dim_column_from_positions(
    name: &str,
    geometry: &OutputGeometry,
    d_out: usize,
) -> Column {
    let positions = &geometry.positions[d_out];
    if positions.is_empty() {
        return Series::new(
            name.into(),
            Vec::<i64>::new(),
        )
        .into();
    }
    let stride = geometry.strides[d_out] as usize;
    let outer =
        geometry.rows / (positions.len() * stride);
    let positions_i64: Vec<i64> = positions
        .iter()
        .map(|&p| p as i64)
        .collect();
    ColumnData::I64(positions_i64)
        .repeat_tile(stride, outer)
        .into_series(name)
        .into()
}

/// Coord-leaf-driven dim column: each output row's `<dim>` value is gathered
/// from the coord leaf's chunk data at `output_position[d_out] - chunk_origin`.
/// Positions not covered by any coord chunk emit null.
fn build_dim_column_from_coord(
    dim_name: IStr,
    coord_leaf_idx: usize,
    leaves: &[&OwnedGridGroup],
    by_leaf: &BTreeMap<usize, Vec<BatchChunkRead>>,
    geometry: &OutputGeometry,
    d_out: usize,
    meta: &ZarrMeta,
) -> Column {
    let coord_leaf = leaves[coord_leaf_idx];
    let chunks = by_leaf
        .get(&coord_leaf_idx)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let cs = coord_leaf.sig.retrieval_shape();
    let chunk_size_d = cs[0];
    let positions = &geometry.positions[d_out];

    let mut per_pos: Vec<Option<(usize, u64)>> =
        vec![None; positions.len()];
    for (slab_idx, chunk) in chunks.iter().enumerate()
    {
        let c =
            coord_leaf.chunk_indices[chunk.slot][0];
        let chunk_origin = c * chunk_size_d;
        let edge_end = chunk_size_d.min(
            coord_leaf.array_shape[0]
                .saturating_sub(chunk_origin),
        );
        let (lo, hi) = coord_leaf.chunk_subsets
            [chunk.slot]
            .as_ref()
            .map_or(
                (0u64, edge_end),
                |sub| {
                    let s = sub.ranges[0].start;
                    let e =
                        sub.ranges[0].end.min(edge_end);
                    (s, e)
                },
            );
        let g_lo = chunk_origin + lo;
        let g_hi = chunk_origin + hi;
        if let Some((p_lo, _)) =
            geometry.contiguous[d_out]
        {
            let lo_idx =
                g_lo.saturating_sub(p_lo) as usize;
            let hi_idx_raw =
                g_hi.saturating_sub(p_lo) as usize;
            let hi_idx =
                hi_idx_raw.min(per_pos.len());
            for i in lo_idx..hi_idx {
                per_pos[i] = Some((
                    slab_idx,
                    positions[i] - chunk_origin,
                ));
            }
        } else {
            for (pi, &p) in
                positions.iter().enumerate()
            {
                if p >= g_lo && p < g_hi {
                    per_pos[pi] = Some((
                        slab_idx,
                        p - chunk_origin,
                    ));
                }
            }
        }
    }

    let encoding = meta
        .array_by_path(dim_name)
        .and_then(|m| m.encoding.as_ref());

    let stride = geometry.strides[d_out] as usize;
    let size = positions.len();
    let rows = geometry.rows;

    let var_data_arrays: Vec<Arc<ColumnData>> =
        chunks
            .iter()
            .map(|c| {
                Arc::clone(&c.raw.var_data[0])
            })
            .collect();

    let resolve = |row: usize| -> Option<(usize, usize)> {
        let pos_idx = (row / stride) % size;
        let (slab_idx, local) = per_pos[pos_idx]?;
        Some((slab_idx, local as usize))
    };

    series_from_optional_gather(
        dim_name.as_ref(),
        rows,
        &var_data_arrays,
        resolve,
        encoding,
    )
}

// =============================================================================
// Var columns: per-leaf, per-var, gather from chunk raw data into output
// =============================================================================

/// Build all var columns for one structural leaf in a joined batch.
///
/// Computes per-chunk keep masks and per-leaf-axis position lookups once,
/// then loops over the leaf's vars writing one output column per var.
fn build_leaf_var_columns(
    leaf: &OwnedGridGroup,
    chunks: &[BatchChunkRead],
    geometry: &OutputGeometry,
    meta: &ZarrMeta,
) -> Vec<Column> {
    let leaf_dims = leaf.sig.dims();
    let leaf_chunk_shape =
        leaf.sig.retrieval_shape();
    let leaf_chunk_strides =
        compute_strides(leaf_chunk_shape);

    let mut keep_masks: Vec<Arc<KeepMask>> =
        Vec::with_capacity(chunks.len());
    let chunk_len = leaf_chunk_shape
        .iter()
        .product::<u64>()
        as usize;
    for chunk in chunks {
        let cidx = &leaf.chunk_indices[chunk.slot];
        let origin: Vec<u64> = cidx
            .iter()
            .zip(leaf_chunk_shape.iter())
            .map(|(i, s)| i * s)
            .collect();
        let keep = compute_in_bounds_mask(
            chunk_len,
            leaf_chunk_shape,
            &origin,
            &leaf.array_shape,
            &leaf_chunk_strides,
            leaf.chunk_subsets[chunk.slot].as_ref(),
        );
        keep_masks.push(Arc::new(keep));
    }

    let mut per_axis: Vec<
        Vec<Option<(u64, u64)>>,
    > = Vec::with_capacity(leaf_dims.len());
    let mut leaf_axis_to_out: Vec<usize> =
        Vec::with_capacity(leaf_dims.len());
    for (d_leaf, &dim_name) in
        leaf_dims.iter().enumerate()
    {
        let d_out = geometry
            .dims
            .iter()
            .position(|d| d == &dim_name)
            .expect("leaf dim must be in output dims");
        leaf_axis_to_out.push(d_out);
        per_axis.push(build_axis_chunk_lookup(
            leaf,
            d_leaf,
            chunks,
            &geometry.positions[d_out],
        ));
    }

    // Map full chunk-grid-index tuple (in leaf dim order) to slab slot index.
    // Required because a single chunk covers a hyper-rectangle in leaf
    // coordinates: per-axis lookups give per-axis chunk indices, but the slot
    // is uniquely determined only by the full N-D tuple.
    let mut chunk_to_slab: BTreeMap<
        SmallVec<[u64; 4]>,
        usize,
    > = BTreeMap::new();
    for (slab_idx, chunk) in chunks.iter().enumerate()
    {
        let cidx: SmallVec<[u64; 4]> = leaf
            .chunk_indices[chunk.slot]
            .iter()
            .copied()
            .collect();
        chunk_to_slab.insert(cidx, slab_idx);
    }

    let var_reads = if let Some(c) = chunks.first() {
        &c.raw.var_reads
    } else {
        return Vec::new();
    };
    let mut cols = Vec::with_capacity(var_reads.len());
    for (var_pos, vr) in
        var_reads.iter().enumerate()
    {
        if geometry
            .dims
            .iter()
            .any(|d| d == &vr.name)
        {
            continue;
        }
        cols.push(build_var_column_joined(
            vr,
            leaf,
            chunks,
            var_pos,
            geometry,
            meta,
            &keep_masks,
            &per_axis,
            &chunk_to_slab,
            &leaf_axis_to_out,
            leaf_chunk_shape,
            &leaf_chunk_strides,
        ));
    }
    cols
}

#[allow(clippy::too_many_arguments)]
fn build_var_column_joined(
    vr: &VarRead,
    leaf: &OwnedGridGroup,
    chunks: &[BatchChunkRead],
    var_pos: usize,
    geometry: &OutputGeometry,
    meta: &ZarrMeta,
    keep_masks: &[Arc<KeepMask>],
    per_axis: &[Vec<Option<(u64, u64)>>],
    chunk_to_slab: &BTreeMap<
        SmallVec<[u64; 4]>,
        usize,
    >,
    leaf_axis_to_out: &[usize],
    leaf_chunk_shape: &[u64],
    leaf_chunk_strides: &[u64],
) -> Column {
    let n_leaf_dims = leaf_chunk_shape.len();

    let var_axis_of_leaf_dim: Vec<Option<usize>> =
        leaf.sig
            .dims()
            .iter()
            .map(|ld| {
                vr.var_dims
                    .iter()
                    .position(|vd| vd == ld)
            })
            .collect();
    let var_strides =
        compute_strides(&vr.var_chunk_shape);

    let mut out_strides_for_leaf_axis: Vec<u64> =
        Vec::with_capacity(n_leaf_dims);
    let mut out_sizes_for_leaf_axis: Vec<u64> =
        Vec::with_capacity(n_leaf_dims);
    for &d_out in leaf_axis_to_out {
        out_strides_for_leaf_axis
            .push(geometry.strides[d_out]);
        out_sizes_for_leaf_axis.push(
            geometry.positions[d_out].len() as u64,
        );
    }

    let resolve = |row: usize| -> Option<(usize, usize)> {
        let mut chunk_idx: SmallVec<[u64; 4]> =
            SmallVec::with_capacity(n_leaf_dims);
        let mut chunk_local: SmallVec<[u64; 4]> =
            SmallVec::with_capacity(n_leaf_dims);
        for d_leaf in 0..n_leaf_dims {
            let stride =
                out_strides_for_leaf_axis[d_leaf]
                    as usize;
            let size =
                out_sizes_for_leaf_axis[d_leaf]
                    as usize;
            let pos_idx =
                (row / stride) % size;
            let (cidx, local) =
                per_axis[d_leaf][pos_idx]?;
            chunk_idx.push(cidx);
            chunk_local.push(local);
        }
        let slab_idx =
            *chunk_to_slab.get(&chunk_idx)?;

        let mut leaf_local_flat: u64 = 0;
        for d_leaf in 0..n_leaf_dims {
            leaf_local_flat += chunk_local[d_leaf]
                * leaf_chunk_strides[d_leaf];
        }

        if !keep_mask_contains(
            &keep_masks[slab_idx],
            leaf_local_flat as usize,
        ) {
            return None;
        }

        let mut var_local_flat: u64 = 0;
        for (d_leaf, var_axis) in
            var_axis_of_leaf_dim.iter().enumerate()
        {
            let Some(va) = *var_axis else {
                continue;
            };
            let local_with_offset =
                chunk_local[d_leaf]
                    + vr.offsets[va];
            let local_clamped =
                local_with_offset.min(
                    vr.var_chunk_shape[va]
                        .saturating_sub(1),
                );
            var_local_flat +=
                local_clamped * var_strides[va];
        }

        Some((slab_idx, var_local_flat as usize))
    };

    let var_data_arrays: Vec<Arc<ColumnData>> =
        chunks
            .iter()
            .map(|c| {
                Arc::clone(&c.raw.var_data[var_pos])
            })
            .collect();

    let encoding = meta
        .array_by_path(vr.name)
        .and_then(|m| m.encoding.as_ref());

    series_from_optional_gather(
        vr.name.as_ref(),
        geometry.rows,
        &var_data_arrays,
        resolve,
        encoding,
    )
}

/// For one leaf-axis, per output position on that axis: which chunk-grid index
/// along that axis covers it, and the chunk-local position. `None` ⇒
/// uncovered. The full chunk's slab slot is resolved separately by combining
/// chunk indices across all leaf dims.
///
/// Note: chunk_subsets are deliberately ignored when the same `c` (chunk grid
/// index) appears in multiple slots with different per-axis subset ranges, the
/// FIRST encountered range wins. In practice for our planner all slots
/// sharing a chunk index along this axis share the same per-axis subset
/// because the subset is intersected with the chunk-grid alignment.
fn build_axis_chunk_lookup(
    leaf: &OwnedGridGroup,
    d_leaf: usize,
    chunks: &[BatchChunkRead],
    positions: &[u64],
) -> Vec<Option<(u64, u64)>> {
    let cs_d = leaf.sig.retrieval_shape()[d_leaf];
    let alen = leaf.array_shape[d_leaf];
    let mut out: Vec<Option<(u64, u64)>> =
        vec![None; positions.len()];
    for chunk in chunks.iter() {
        let c =
            leaf.chunk_indices[chunk.slot][d_leaf];
        let chunk_origin = c * cs_d;
        let edge_end = cs_d.min(
            alen.saturating_sub(chunk_origin),
        );
        let (lo, hi) = leaf.chunk_subsets
            [chunk.slot]
            .as_ref()
            .map_or(
                (0u64, edge_end),
                |sub| {
                    let s = sub.ranges[d_leaf].start;
                    let e = sub.ranges[d_leaf]
                        .end
                        .min(edge_end);
                    (s, e)
                },
            );
        let g_lo = chunk_origin + lo;
        let g_hi = chunk_origin + hi;
        for (pi, &p) in positions.iter().enumerate()
        {
            if p >= g_lo
                && p < g_hi
                && out[pi].is_none()
            {
                out[pi] = Some((
                    c,
                    p - chunk_origin,
                ));
            }
        }
    }
    out
}

// =============================================================================
// Type-dispatched gather: build Series<Option<T>> from per-output-row source
// =============================================================================

fn keep_mask_contains(
    keep: &KeepMask,
    flat: usize,
) -> bool {
    match keep {
        KeepMask::All(n) => flat < *n,
        KeepMask::Sparse(idx) => {
            idx.binary_search(&flat).is_ok()
        }
    }
}

/// Build a Polars Series from `rows` lookups of `(chunk_in_slab, source_idx)` against
/// `data[chunk_in_slab]`. Dispatches on the source `ColumnData` variant.
///
/// Encoding (Time / ScaleOffset) is applied at the value level — for nulls we
/// emit `None` directly without going through encoding.
fn series_from_optional_gather<
    F: Fn(usize) -> Option<(usize, usize)>,
>(
    name: &str,
    rows: usize,
    data: &[Arc<ColumnData>],
    resolve: F,
    encoding: Option<&VarEncoding>,
) -> Column {
    debug_assert!(!data.is_empty());

    if let Some(VarEncoding::ScaleOffset {
        scale_factor,
        add_offset,
        fill_value,
    }) = encoding
    {
        let scale = *scale_factor;
        let offset = *add_offset;
        let fill = *fill_value;
        let mut out: Vec<Option<f64>> =
            Vec::with_capacity(rows);
        for row in 0..rows {
            let v =
                resolve(row).map(|(c, i)| {
                    let raw =
                        column_data_get_f64(
                            &data[c], i,
                        );
                    if fill.is_some_and(|fv| {
                        raw == fv
                    }) {
                        f64::NAN
                    } else {
                        raw * scale + offset
                    }
                });
            out.push(v);
        }
        return Series::new(name.into(), out)
            .into();
    }

    if let Some(VarEncoding::Time(te)) = encoding {
        let mut out: Vec<Option<i64>> =
            Vec::with_capacity(rows);
        for row in 0..rows {
            let v =
                resolve(row).map(|(c, i)| {
                    let raw =
                        column_data_get_i64(
                            &data[c], i,
                        );
                    te.decode(raw)
                });
            out.push(v);
        }
        let s = Series::new(name.into(), out);
        let casted = s
            .cast(&te.to_polars_dtype())
            .unwrap_or(s);
        return casted.into();
    }

    macro_rules! gather_primitive {
        ($variant:ident, $ty:ty) => {{
            let mut out: Vec<Option<$ty>> =
                Vec::with_capacity(rows);
            for row in 0..rows {
                let v =
                    resolve(row).map(|(c, i)| {
                        match &*data[c] {
                            ColumnData::$variant(v) => v[i],
                            _ => unreachable!(
                                "ColumnData variant mismatch across chunks of one var"
                            ),
                        }
                    });
                out.push(v);
            }
            Series::new(name.into(), out)
        }};
    }

    let series = match &*data[0] {
        ColumnData::Bool(_) => {
            gather_primitive!(Bool, bool)
        }
        ColumnData::I8(_) => {
            gather_primitive!(I8, i8)
        }
        ColumnData::I16(_) => {
            gather_primitive!(I16, i16)
        }
        ColumnData::I32(_) => {
            gather_primitive!(I32, i32)
        }
        ColumnData::I64(_) => {
            gather_primitive!(I64, i64)
        }
        ColumnData::U8(_) => {
            gather_primitive!(U8, u8)
        }
        ColumnData::U16(_) => {
            gather_primitive!(U16, u16)
        }
        ColumnData::U32(_) => {
            gather_primitive!(U32, u32)
        }
        ColumnData::U64(_) => {
            gather_primitive!(U64, u64)
        }
        ColumnData::F32(_) => {
            gather_primitive!(F32, f32)
        }
        ColumnData::F64(_) => {
            gather_primitive!(F64, f64)
        }
        ColumnData::Str(_) => {
            let mut out: Vec<Option<String>> =
                Vec::with_capacity(rows);
            for row in 0..rows {
                out.push(resolve(row).map(
                    |(c, i)| match &*data[c] {
                        ColumnData::Str(v) => {
                            v[i].clone()
                        }
                        _ => unreachable!(),
                    },
                ));
            }
            Series::new(name.into(), out)
        }
        ColumnData::Bin(_) => {
            let mut owned: Vec<Option<Vec<u8>>> =
                Vec::with_capacity(rows);
            for row in 0..rows {
                owned.push(resolve(row).map(
                    |(c, i)| match &*data[c] {
                        ColumnData::Bin(v) => {
                            v[i].clone()
                        }
                        _ => unreachable!(),
                    },
                ));
            }
            let refs: Vec<Option<&[u8]>> = owned
                .iter()
                .map(|o| o.as_deref())
                .collect();
            Series::new(name.into(), refs)
        }
    };

    series.into()
}

fn column_data_get_f64(
    data: &ColumnData,
    i: usize,
) -> f64 {
    match data {
        ColumnData::Bool(v) => v[i] as u8 as f64,
        ColumnData::I8(v) => v[i] as f64,
        ColumnData::I16(v) => v[i] as f64,
        ColumnData::I32(v) => v[i] as f64,
        ColumnData::I64(v) => v[i] as f64,
        ColumnData::U8(v) => v[i] as f64,
        ColumnData::U16(v) => v[i] as f64,
        ColumnData::U32(v) => v[i] as f64,
        ColumnData::U64(v) => v[i] as f64,
        ColumnData::F32(v) => v[i] as f64,
        ColumnData::F64(v) => v[i],
        ColumnData::Str(_)
        | ColumnData::Bin(_) => f64::NAN,
    }
}

fn column_data_get_i64(
    data: &ColumnData,
    i: usize,
) -> i64 {
    match data {
        ColumnData::Bool(v) => i64::from(v[i]),
        ColumnData::I8(v) => i64::from(v[i]),
        ColumnData::I16(v) => i64::from(v[i]),
        ColumnData::I32(v) => i64::from(v[i]),
        ColumnData::I64(v) => v[i],
        ColumnData::U8(v) => i64::from(v[i]),
        ColumnData::U16(v) => i64::from(v[i]),
        ColumnData::U32(v) => i64::from(v[i]),
        ColumnData::U64(v) => v[i] as i64,
        ColumnData::F32(v) => v[i] as i64,
        ColumnData::F64(v) => v[i] as i64,
        ColumnData::Str(_)
        | ColumnData::Bin(_) => 0,
    }
}
