use super::errors::CoordIndexResolver;
use super::prelude::ZarrDatasetMeta;
use super::selection::DatasetSelection;

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
        DatasetSelection::all_for_vars(self.vars.to_vec())
    }
}

