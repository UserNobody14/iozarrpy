//! Entry points for chunk planning compilation.
//!
//! Provides `compute_dims_and_lengths_unified` used by Criterion benches to set
//! up the same dim/length pair the planner produces internally.

#[cfg(feature = "bench")]
use crate::meta::ZarrMeta;
#[cfg(feature = "bench")]
use crate::shared::IStr;

#[cfg(feature = "bench")]
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
