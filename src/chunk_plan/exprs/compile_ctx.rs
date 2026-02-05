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
    pub(crate) dim_lengths: &'a [u64],
    pub(crate) vars: &'a [IStr],
}

impl<'a> LazyCompileCtx<'a> {
    /// Create a new lazy compilation context.
    pub(crate) fn new(
        meta: &'a ZarrDatasetMeta,
        unified_meta: Option<&'a ZarrMeta>,
        dims: &'a [IStr],
        dim_lengths: &'a [u64],
        vars: &'a [IStr],
    ) -> Self {
        Self {
            meta,
            unified_meta,
            dims,
            dim_lengths,
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

    /// Get the length of a dimension by index.
    pub(crate) fn dim_length(
        &self,
        idx: usize,
    ) -> Option<u64> {
        self.dim_lengths.get(idx).copied()
    }

    /// Check if a column is a dimension.
    pub(crate) fn is_dimension(
        &self,
        col: &str,
    ) -> bool {
        self.dims.iter().any(|d| {
            <IStr as AsRef<str>>::as_ref(d) == col
        })
    }
}
