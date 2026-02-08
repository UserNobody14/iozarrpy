//! Entry points for chunk planning compilation.
//!
//! This module provides the main entry points for compiling Polars expressions
//! into chunk plans. The compilation uses a lazy approach:
//! 1. Compile expression to `LazyDatasetSelection` (no I/O)
//! 2. Collect resolution requests and batch-resolve value ranges to index ranges
//! 3. Materialize to concrete `DatasetSelection`
//!
//! This enables efficient I/O batching and concurrent resolution for async stores.

use crate::IStr;
use crate::chunk_plan::indexing::lazy_materialize::{
    MergedCache, collect_requests_with_meta, materialize,
};
use crate::meta::ZarrMeta;

pub(crate) fn compute_dims_and_lengths_unified(
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
