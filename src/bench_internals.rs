//! Re-exports for Criterion benchmarks.
//!
//! Gated behind `#[cfg(feature = "bench")]` in `lib.rs`.
//! Not part of the public API — intended only for `benches/`.

use std::ops::Range;

pub use polars_arrow::array::PrimitiveArray;

pub use crate::reader::{
    ColumnData, checked_chunk_len,
    compute_strides,
};

pub use crate::scan::shared::columns::KeepMask;
pub use crate::scan::shared::{
    build_coord_column, build_var_column,
    compute_in_bounds_mask,
    compute_var_chunk_indices,
    should_include_column,
};

pub use crate::scan::sync_scan::chunk_to_df_from_grid_with_backend;

pub use crate::shared::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync, FromManyIstrs, IStr,
    IntoIStr, IntoManyIstrs,
};

use crate::chunk_plan::coord_resolve::Expansion;
use crate::chunk_plan::exprs::compile_ctx::CoordResolver;
pub use crate::chunk_plan::exprs::compile_ctx::{
    LazyCompileCtx, Universe,
};
use crate::chunk_plan::indexing::types::ValueRangePresent;
pub use crate::chunk_plan::{
    ChunkGridSignature, ChunkSubset,
    compile_expr,
    compute_dims_and_lengths_unified,
};

pub use crate::errors::BackendError;

pub use crate::meta::{
    VarEncoding, ZarrArrayMeta, ZarrMeta,
    ZarrNode,
};

pub struct BenchCoordResolver;

impl CoordResolver for BenchCoordResolver {
    #[allow(private_interfaces)]
    fn resolve(
        &self,
        _dim: IStr,
        _meta: &ZarrMeta,
        dim_len: u64,
        _vr: &ValueRangePresent,
        _expansion: Expansion,
    ) -> Result<Vec<Range<u64>>, BackendError>
    {
        Ok(std::iter::once(0..dim_len).collect())
    }
}
