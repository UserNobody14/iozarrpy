use super::traits::{
    ChunkedDataBackendAsync,
    ChunkedDataBackendSync,
    HasMetadataBackendAsync,
    HasMetadataBackendSync,
};
use crate::chunk_plan::{
    GridJoinTree, PlannerStats,
    compile_to_tree_async, compile_to_tree_sync,
};
use crate::errors::BackendError;
use crate::meta::ZarrMeta;
use polars::prelude::Expr;

/// Compile a Polars expression directly into a [`GridJoinTree`].
pub trait ChunkedExpressionCompilerSync:
    HasMetadataBackendSync<ZarrMeta>
    + ChunkedDataBackendSync
{
    fn compile_expression_to_tree_sync(
        &self,
        expr: &Expr,
    ) -> Result<
        (Option<GridJoinTree>, PlannerStats),
        BackendError,
    >
    where
        Self: Sized,
    {
        let meta = self.metadata()?;
        compile_to_tree_sync(expr, &meta, self)
    }
}

#[async_trait::async_trait]
pub trait ChunkedExpressionCompilerAsync:
    HasMetadataBackendAsync<ZarrMeta>
    + ChunkedDataBackendAsync
{
    async fn compile_expression_to_tree_async(
        &self,
        expr: &Expr,
    ) -> Result<
        (Option<GridJoinTree>, PlannerStats),
        BackendError,
    >
    where
        Self: Sized,
    {
        let meta = self.metadata().await?;
        compile_to_tree_async(expr, &meta, self).await
    }
}

impl<
    B: HasMetadataBackendSync<ZarrMeta>
        + ChunkedDataBackendSync,
> ChunkedExpressionCompilerSync for B
{
}

#[async_trait::async_trait]
impl<
    B: HasMetadataBackendAsync<ZarrMeta>
        + ChunkedDataBackendAsync,
> ChunkedExpressionCompilerAsync for B
{
}
