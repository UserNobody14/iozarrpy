//! Compilation context for lazy chunk planning.

use crate::IStr;
use crate::meta::ZarrMeta;

/// Compilation context for the lazy compilation path.
///
/// Does not contain a resolver - instead, value ranges are stored as-is
/// and resolved in a separate batch phase.
pub struct LazyCompileCtx<'a> {
    pub meta: &'a ZarrMeta,
    pub dims: &'a [IStr],
}

impl<'a> LazyCompileCtx<'a> {
    pub fn new(
        meta: &'a ZarrMeta,
        dims: &'a [IStr],
    ) -> Self {
        Self { meta, dims }
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
