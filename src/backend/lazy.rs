use polars_lazy::prelude::AnonymousScan;

use crate::backend::zarr::FullyCachedZarrBackendSync;

impl AnonymousScan
    for FullyCachedZarrBackendSync
{
    fn schema(
        &self,
        _infer_schema_length: Option<usize>,
    ) -> polars::prelude::PolarsResult<
        polars::prelude::SchemaRef,
    > {
        polars::prelude::polars_bail!(ComputeError: "must supply either a schema or a schema function");
    }

    fn allows_predicate_pushdown(&self) -> bool {
        true
    }

    fn allows_projection_pushdown(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        todo!()
    }

    fn scan(
        &self,
        scan_opts: polars::prelude::AnonymousScanArgs,
    ) -> polars::prelude::PolarsResult<
        polars::prelude::DataFrame,
    > {
        todo!()
    }
}
