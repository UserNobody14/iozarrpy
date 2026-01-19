use super::errors::CoordIndexResolver;
use crate::chunk_plan::prelude::ZarrDatasetMeta;

/// Compilation context for the eager (blocking) compilation path.
///
/// Contains a resolver that performs immediate value-to-index resolution.
pub(crate) struct CompileCtx<'a> {
    pub(crate) meta: &'a ZarrDatasetMeta,
    pub(crate) dims: &'a [String],
    pub(crate) dim_lengths: &'a [u64],
    pub(crate) vars: &'a [String],
    pub(crate) resolver: &'a mut dyn CoordIndexResolver,
}

/// Compilation context for the lazy compilation path.
///
/// Does not contain a resolver - instead, value ranges are stored as-is
/// and resolved in a separate batch phase.
pub(crate) struct LazyCompileCtx<'a> {
    pub(crate) meta: &'a ZarrDatasetMeta,
    pub(crate) dims: &'a [String],
    pub(crate) dim_lengths: &'a [u64],
    pub(crate) vars: &'a [String],
}

impl<'a> LazyCompileCtx<'a> {
    /// Create a new lazy compilation context.
    pub(crate) fn new(
        meta: &'a ZarrDatasetMeta,
        dims: &'a [String],
        dim_lengths: &'a [u64],
        vars: &'a [String],
    ) -> Self {
        Self {
            meta,
            dims,
            dim_lengths,
            vars,
        }
    }

    /// Get the index of a dimension by name.
    pub(crate) fn dim_index(&self, dim: &str) -> Option<usize> {
        self.dims.iter().position(|d| d == dim)
    }

    /// Get the length of a dimension by index.
    pub(crate) fn dim_length(&self, idx: usize) -> Option<u64> {
        self.dim_lengths.get(idx).copied()
    }

    /// Check if a column is a dimension.
    pub(crate) fn is_dimension(&self, col: &str) -> bool {
        self.dims.contains(&col.to_string())
    }
}
