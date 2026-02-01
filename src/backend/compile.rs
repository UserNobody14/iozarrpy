use std::sync::Arc;

use polars::prelude::Expr;

use crate::backend::traits::{
    BackendError, ChunkedDataBackendAsync,
    ChunkedDataBackendSync, HasAsyncStore,
    HasMetadataBackendAsync,
    HasMetadataBackendSync, HasStore,
};

use crate::chunk_plan::GroupedChunkPlan;
use crate::chunk_plan::{
    PlannerStats,
    compile_expr_to_grouped_chunk_plan_unified,
    compile_expr_to_grouped_chunk_plan_unified_async,
    compile_expr_with_backend_async,
};
use crate::meta::ZarrMeta;

/// Compile a Polars expression to a chunk plan synchronously.
pub trait ChunkedExpressionCompilerSync:
    HasMetadataBackendSync<ZarrMeta>
    + ChunkedDataBackendSync
{
    fn compile_expression_sync(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    >;
}

#[async_trait::async_trait]
pub trait ChunkedExpressionCompilerAsync:
    HasMetadataBackendAsync<ZarrMeta>
    + ChunkedDataBackendAsync
{
    async fn compile_expression_async(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    >;
}

impl<
    B: HasMetadataBackendSync<ZarrMeta>
        + ChunkedDataBackendSync
        + HasStore,
> ChunkedExpressionCompilerSync for B
{
    fn compile_expression_sync(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    > {
        let meta = self.metadata()?;
        Ok(compile_expr_to_grouped_chunk_plan_unified(
            expr,
            &meta,
            self.store().clone(),
        )?)
    }
}

#[async_trait::async_trait]
impl<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync
        + HasAsyncStore,
> ChunkedExpressionCompilerAsync for B
{
    async fn compile_expression_async(
        &self,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    > {
        let meta = self.metadata().await?;
        Ok(compile_expr_to_grouped_chunk_plan_unified_async(
            expr,
            &meta,
            self.async_store().clone(),
        ).await?)
    }
}

/// Trait for backends that can compile expressions using their own
/// chunk reading capabilities, without needing a raw store.
///
/// This is useful for backends that wrap or abstract storage in ways
/// that don't expose a raw zarrs store.
#[async_trait::async_trait]
pub trait ChunkedExpressionCompilerWithBackendAsync:
    HasMetadataBackendAsync<ZarrMeta>
    + ChunkedDataBackendAsync
    + Send
    + Sync
    + 'static
{
    async fn compile_expression_with_backend_async(
        self: Arc<Self>,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    >;
}

#[async_trait::async_trait]
impl<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync
        + Send
        + Sync
        + 'static,
> ChunkedExpressionCompilerWithBackendAsync
    for B
{
    async fn compile_expression_with_backend_async(
        self: Arc<Self>,
        expr: &Expr,
    ) -> Result<
        (GroupedChunkPlan, PlannerStats),
        BackendError,
    > {
        let meta = self.metadata().await?;
        Ok(compile_expr_with_backend_async(
            expr, meta, self,
        )
        .await?)
    }
}
