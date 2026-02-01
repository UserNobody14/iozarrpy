use polars::prelude::Expr;

use crate::backend::traits::{
    BackendError, ChunkedDataBackendAsync,
    ChunkedDataBackendSync, HasAsyncStore,
    HasMetadataBackendAsync,
    HasMetadataBackendSync, HasStore,
};

use crate::chunk_plan::{
    CompileError, PlannerStats,
    compile_expr_to_grouped_chunk_plan,
    compile_expr_to_grouped_chunk_plan_async,
    compile_expr_to_grouped_chunk_plan_unified_async,
};
use crate::chunk_plan::{
    GroupedChunkPlan,
    compile_expr_to_grouped_chunk_plan_unified,
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
