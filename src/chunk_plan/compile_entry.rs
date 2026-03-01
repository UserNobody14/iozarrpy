//! Entry points for chunk planning compilation.
//!
//! Provides `compute_dims_and_lengths_unified` used during expression compilation.

use crate::IStr;
use crate::meta::ZarrMeta;

pub fn compute_dims_and_lengths_unified(
    meta: &ZarrMeta,
) -> (Vec<IStr>, Vec<u64>) {
    let dims = meta.dim_analysis.all_dims.clone();
    let dim_lengths: Vec<u64> = dims
        .iter()
        .map(|d| {
            meta.dim_analysis
                .dim_lengths
                .get(d)
                .copied()
                .unwrap_or(1)
        })
        .collect();
    (dims, dim_lengths)
}
