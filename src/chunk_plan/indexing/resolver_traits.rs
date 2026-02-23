//! Traits for coordinate-to-index resolution.
//!
//! Provides error types and the dim-length lookup needed for index-only dimensions.

use crate::IStr;

/// Error type for resolution operations.
#[derive(Debug, Clone)]
pub(crate) enum ResolutionError {
    /// Resolution was attempted but failed (e.g., non-monotonic array).
    Unresolvable(String),
}

impl std::fmt::Display for ResolutionError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            ResolutionError::Unresolvable(
                msg,
            ) => {
                write!(f, "unresolvable: {}", msg)
            }
        }
    }
}

impl std::error::Error for ResolutionError {}

/// Pre-computed metadata for resolving a single dimension's coordinate array.
pub(crate) struct DimResolutionCtx {
    pub(crate) n: u64,
    pub(crate) chunk_size: u64,
    pub(crate) time_enc:
        Option<crate::meta::TimeEncoding>,
    pub(crate) array_path: IStr,
}

impl DimResolutionCtx {
    pub(crate) fn from_meta(
        dim: &IStr,
        meta: &crate::meta::ZarrMeta,
    ) -> Option<Self> {
        let coord_meta =
            meta.array_by_path(dim.clone())?;
        if coord_meta.shape.len() != 1 {
            return None;
        }
        let n = coord_meta.shape[0];
        Some(Self {
            n,
            chunk_size: coord_meta
                .chunk_shape
                .first()
                .copied()
                .unwrap_or(n),
            time_enc: coord_meta
                .time_encoding
                .clone(),
            array_path: coord_meta.path.clone(),
        })
    }
}
