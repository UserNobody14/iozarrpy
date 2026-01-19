use super::errors::CoordIndexResolver;
use crate::chunk_plan::prelude::ZarrDatasetMeta;
pub(crate) struct CompileCtx<'a> {
    pub(crate) meta: &'a ZarrDatasetMeta,
    pub(crate) dims: &'a [String],
    pub(crate) dim_lengths: &'a [u64],
    pub(crate) vars: &'a [String],
    pub(crate) resolver: &'a mut dyn CoordIndexResolver,
}
