//! Entry points for chunk planning compilation.
//!
//! Provides `compute_dims_and_lengths_unified` used by Criterion benches to set
//! up the same dim/length pair the planner produces internally.

#[cfg(feature = "bench")]
use crate::chunk_plan::exprs::compile_ctx::Universe;
#[cfg(feature = "bench")]
use crate::meta::ZarrMeta;
#[cfg(feature = "bench")]
use crate::shared::IStr;

#[cfg(feature = "bench")]
pub fn compute_dims_and_lengths_unified(
    meta: &ZarrMeta,
) -> (Vec<IStr>, Vec<u64>) {
    let universe = Universe::from_meta(meta);
    (
        universe.dims.to_vec(),
        universe.shape.to_vec(),
    )
}
