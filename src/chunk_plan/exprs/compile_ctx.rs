use super::errors::CoordIndexResolver;
use crate::chunk_plan::prelude::ZarrDatasetMeta;
use crate::chunk_plan::indexing::selection::DatasetSelection;
use crate::chunk_plan::indexing::selection::dataset_all_for_vars;

pub(crate) struct CompileCtx<'a> {
    pub(crate) meta: &'a ZarrDatasetMeta,
    pub(crate) dims: &'a [String],
    pub(crate) dim_lengths: &'a [u64],
    pub(crate) vars: &'a [String],
    pub(crate) resolver: &'a mut dyn CoordIndexResolver,
}

impl CompileCtx<'_> {
    /// Conservative “select all” for the currently selected variables.
    pub(crate) fn all(&self) -> DatasetSelection {
        dataset_all_for_vars(self.vars.to_vec())
    }
}

