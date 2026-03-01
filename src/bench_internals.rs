//! Re-exports for Criterion benchmarks.
//!
//! Gated behind `#[cfg(feature = "bench")]` in `lib.rs`.
//! Not part of the public API — intended only for `benches/`.

pub use crate::reader::{
    ColumnData, checked_chunk_len, compute_strides,
};

pub use crate::scan::shared::{
    KeepMask, build_coord_column, build_var_column,
    compute_in_bounds_mask,
    compute_var_chunk_indices,
    should_include_column,
};

pub use crate::scan::sync_scan::chunk_to_df_from_grid_with_backend;

pub use crate::shared::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync,
};

pub use crate::chunk_plan::{
    ChunkGridSignature, ChunkSubset,
    DatasetSelection, ExprPlan, GroupedChunkPlan,
    LazyCompileCtx, compile_expr,
    compute_dims_and_lengths_unified,
    selection_to_grouped_chunk_plan_unified_from_meta,
};

pub use crate::errors::BackendError;

pub use crate::meta::{
    DimensionAnalysis, VarEncoding, ZarrArrayMeta,
    ZarrMeta, ZarrNode,
};
