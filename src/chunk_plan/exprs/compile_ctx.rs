//! Compilation context for chunk planning.
//!
//! Carries the metadata, the dim universe, and a [`CoordResolver`] that
//! maps `(dim, value-range, expansion)` → resolved index ranges. The
//! resolver is sync; async backends pre-load coord arrays in parallel and
//! feed a lookup-only resolver into compilation.

use std::ops::Range;

use smallvec::SmallVec;

use crate::chunk_plan::coord_resolve::Expansion;
use crate::chunk_plan::indexing::types::ValueRangePresent;
use crate::errors::BackendError;
use crate::meta::ZarrMeta;
use crate::shared::IStr;

/// Resolves a value-range constraint on a dim to a list of concrete
/// index ranges. Implementations are pre-populated tables — the actual
/// backend I/O happens before compile, in parallel across independent
/// dims (sync uses rayon, async uses `try_join_all`).
pub trait CoordResolver {
    fn resolve(
        &self,
        dim: IStr,
        meta: &ZarrMeta,
        dim_len: u64,
        vr: &ValueRangePresent,
        expansion: Expansion,
    ) -> Result<Vec<Range<u64>>, BackendError>;
}

/// Pre-computed dim universe + shape for the dataset, shared across
/// compile passes and the builder. Built once per `compile_to_tree_*`
/// call.
#[derive(Debug, Clone)]
pub struct Universe {
    pub dims: SmallVec<[IStr; 4]>,
    pub shape: SmallVec<[u64; 4]>,
}

impl Universe {
    pub fn from_meta(meta: &ZarrMeta) -> Self {
        let dims: SmallVec<[IStr; 4]> = meta
            .dim_analysis
            .all_dims
            .iter()
            .copied()
            .collect();
        let shape: SmallVec<[u64; 4]> = dims
            .iter()
            .map(|d| {
                lookup_dim_len(meta, d)
                    .unwrap_or(0)
            })
            .collect();
        Self { dims, shape }
    }
}

/// Compilation context.
pub struct LazyCompileCtx<'a> {
    pub meta: &'a ZarrMeta,
    pub universe: &'a Universe,
    pub resolver: &'a dyn CoordResolver,
}

impl<'a> LazyCompileCtx<'a> {
    pub fn new(
        meta: &'a ZarrMeta,
        universe: &'a Universe,
        resolver: &'a dyn CoordResolver,
    ) -> Self {
        Self {
            meta,
            universe,
            resolver,
        }
    }

    /// Aliased for compile sites that do membership checks.
    pub(crate) fn dims(&self) -> &[IStr] {
        &self.universe.dims
    }

    /// Get the index of a dimension by name.
    pub(crate) fn dim_index(
        &self,
        dim: &str,
    ) -> Option<usize> {
        self.universe.dims.iter().position(|d| {
            <IStr as AsRef<str>>::as_ref(d) == dim
        })
    }

    /// Length of `dim` according to dim_analysis or the dim's coord array.
    pub(crate) fn dim_len(
        &self,
        dim: &IStr,
    ) -> Option<u64> {
        lookup_dim_len(self.meta, dim)
    }

    /// Resolve a value-range to index ranges for `dim`.
    pub(crate) fn resolve(
        &self,
        dim: IStr,
        vr: &ValueRangePresent,
        expansion: Expansion,
    ) -> Result<Vec<Range<u64>>, BackendError>
    {
        let dim_len = self
            .dim_len(&dim)
            .ok_or_else(|| unknown_dim(dim))?;
        self.resolver.resolve(
            dim, self.meta, dim_len, vr, expansion,
        )
    }
}

pub(crate) fn lookup_dim_len(
    meta: &ZarrMeta,
    dim: &IStr,
) -> Option<u64> {
    meta.dim_analysis
        .dim_lengths
        .get(dim)
        .copied()
        .or_else(|| {
            meta.array_by_path(*dim).and_then(
                |a| a.shape.first().copied(),
            )
        })
}

pub(crate) fn unknown_dim(
    dim: IStr,
) -> BackendError {
    BackendError::other(format!(
        "unknown dim '{}' (no dim length in meta and no \
         coordinate array)",
        AsRef::<str>::as_ref(&dim),
    ))
}
