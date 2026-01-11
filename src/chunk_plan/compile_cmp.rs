use super::errors::{CompileError, CoordIndexResolver};
use super::index_ranges::index_range_for_index_dim;
use super::literals::{chunk_ranges_for_index_range, literal_to_scalar, rect_all_dims};
use super::plan::ChunkPlanNode;
use super::prelude::*;
use super::types::{BoundKind, ValueRange};

pub(super) fn compile_cmp(
    col: &str,
    op: Operator,
    lit: &LiteralValue,
    meta: &ZarrDatasetMeta,
    dims: &[String],
    grid_shape: &[u64],
    regular_chunk_shape: &[u64],
    resolver: &mut dyn CoordIndexResolver,
) -> Result<ChunkPlanNode, CompileError> {
    let dim_idx = dims
        .iter()
        .position(|d| d == col)
        .ok_or(CompileError::Unsupported(format!(
            "column '{}' not found in dimensions",
            col
        )))?;

    let time_encoding = meta.arrays.get(col).and_then(|a| a.time_encoding.as_ref());
    let Some(scalar) = literal_to_scalar(lit, time_encoding) else {
        return Err(CompileError::Unsupported(format!(
            "unsupported literal: {:?}",
            lit
        )));
    };

    let mut vr = ValueRange::default();
    match op {
        Operator::Eq => vr.eq = Some(scalar),
        Operator::Gt => vr.min = Some((scalar, BoundKind::Exclusive)),
        Operator::GtEq => vr.min = Some((scalar, BoundKind::Inclusive)),
        Operator::Lt => vr.max = Some((scalar, BoundKind::Exclusive)),
        Operator::LtEq => vr.max = Some((scalar, BoundKind::Inclusive)),
        _ => {
            return Err(CompileError::Unsupported(format!(
                "unsupported operator: {:?}",
                op
            )));
        }
    }

    let idx_range = match resolver.index_range_for_value_range(col, &vr) {
        Ok(Some(r)) => r,
        Ok(None) => {
            // If there's no 1D coordinate array for this dimension, treat it as a pure index dim.
            // This is common for grid dims like (y, x) where the user predicates on integer indices.
            if meta.arrays.get(col).is_none() {
                let dim_len_est = regular_chunk_shape[dim_idx]
                    .max(1)
                    .saturating_mul(grid_shape[dim_idx]);
                index_range_for_index_dim(&vr, dim_len_est).ok_or_else(|| {
                    CompileError::Unsupported("failed to plan index-only dimension".to_owned())
                })?
            } else {
                return Err(CompileError::Unsupported(
                    "failed to get index range for value range".to_owned(),
                ));
            }
        }
        Err(e) => {
            return Err(CompileError::Unsupported(format!(
                "failed to get index range for value range: {:?}",
                e
            )));
        }
    };

    if idx_range.is_empty() {
        return Ok(ChunkPlanNode::Empty);
    }
    let Some(dim_range) = chunk_ranges_for_index_range(
        idx_range,
        regular_chunk_shape[dim_idx].max(1),
        grid_shape[dim_idx],
    ) else {
        return Ok(ChunkPlanNode::Empty);
    };

    let mut rect = rect_all_dims(grid_shape);
    rect[dim_idx] = dim_range;
    Ok(ChunkPlanNode::Rect(rect))
}

