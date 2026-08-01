//! Entry points for chunk planning compilation.
//!
//! Provides `compute_dims_and_lengths_unified` used during expression compilation.

use crate::meta::ZarrMeta;
use crate::shared::IStr;

pub fn compute_dims_and_lengths_unified(
    meta: &ZarrMeta,
) -> (Vec<IStr>, Vec<u64>) {
    let dim_lengths: Vec<u64> = meta
        .dim_analysis
        .all_dims
        .iter()
        .map(|d| {
            meta.dim_analysis
                .dim_lengths
                .get(d)
                .copied()
                .unwrap_or(1)
        })
        .collect();
    (
        meta.dim_analysis.all_dims.to_vec(),
        dim_lengths,
    )
}
