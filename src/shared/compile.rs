use super::traits::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync,
    HasMetadataBackendAsync,
    HasMetadataBackendSync,
};
use crate::errors::BackendError;
use polars::prelude::Expr;

use crate::PlannerStats;
use crate::chunk_plan::GroupedChunkPlan;
use crate::chunk_plan::LazyCompileCtx;
use crate::chunk_plan::compile_expr;
use crate::chunk_plan::selection_to_grouped_chunk_plan_unified_from_meta;
use crate::chunk_plan::{
    compute_dims_and_lengths_unified,
    resolve_expr_plan_async,
    resolve_expr_plan_sync,
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
        + ChunkedDataBackendSync,
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
        let (dims, _dim_lengths) =
            compute_dims_and_lengths_unified(
                &meta,
            );
        let mut ctx =
            LazyCompileCtx::new(&meta, &dims);
        let expr_plan =
            compile_expr(expr, &mut ctx)?;
        let selection = resolve_expr_plan_sync(
            &expr_plan, &meta, self,
        )?;
        let stats =
            PlannerStats { coord_reads: 0 };
        let grouped_plan =
            selection_to_grouped_chunk_plan_unified_from_meta(
                &selection, &meta,
            )?;
        Ok((grouped_plan, stats))
    }
}

#[async_trait::async_trait]
impl<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync,
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
        let (dims, _dim_lengths) =
            compute_dims_and_lengths_unified(
                &meta,
            );
        let mut ctx =
            LazyCompileCtx::new(&meta, &dims);
        let expr_plan =
            compile_expr(expr, &mut ctx)?;
        let selection = resolve_expr_plan_async(
            &expr_plan, &meta, self,
        )
        .await?;
        let stats =
            PlannerStats { coord_reads: 0 };
        let grouped_plan =
            selection_to_grouped_chunk_plan_unified_from_meta(
                &selection, &meta,
            )?;
        Ok((grouped_plan, stats))
    }
}
