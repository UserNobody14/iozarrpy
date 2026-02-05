//! Compilation context for lazy chunk planning.

use crate::IStr;
use crate::chunk_plan::prelude::ZarrDatasetMeta;
use crate::meta::ZarrMeta;

/// Compilation context for the lazy compilation path.
///
/// Does not contain a resolver - instead, value ranges are stored as-is
/// and resolved in a separate batch phase.
pub(crate) struct LazyCompileCtx<'a> {
    pub(crate) meta: &'a ZarrDatasetMeta,
    pub(crate) unified_meta: Option<&'a ZarrMeta>,
    pub(crate) dims: &'a [IStr],
    pub(crate) vars: &'a [IStr],
}

impl<'a> LazyCompileCtx<'a> {
    /// Create a new lazy compilation context.
    pub(crate) fn new(
        meta: &'a ZarrDatasetMeta,
        unified_meta: Option<&'a ZarrMeta>,
        dims: &'a [IStr],
        vars: &'a [IStr],
    ) -> Self {
        Self {
            meta,
            unified_meta,
            dims,
            vars,
        }
    }

    /// Get the index of a dimension by name.
    pub(crate) fn dim_index(
        &self,
        dim: &str,
    ) -> Option<usize> {
        self.dims.iter().position(|d| {
            <IStr as AsRef<str>>::as_ref(d) == dim
        })
    }
}
