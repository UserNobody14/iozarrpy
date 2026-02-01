use std::fmt::Display;

use crate::chunk_plan::indexing::types::{
    IndexRange, ValueRange,
};

#[derive(Debug, Clone)]
pub(crate) enum CompileError {
    Unsupported(String),
    MissingPrimaryDims(String),
}

impl Display for CompileError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // Format the error as a string
        match self {
            CompileError::Unsupported(msg) => {
                write!(f, "unsupported: {msg}")
            }
            CompileError::MissingPrimaryDims(
                msg,
            ) => {
                write!(
                    f,
                    "missing primary dims: {msg}"
                )
            }
        }
    }
}

impl std::error::Error for CompileError {}

#[derive(Debug)]
pub(crate) enum ResolveError {
    UnsupportedCoordDtype(String),
    MissingCoord(String),
    OutOfBounds,
    Zarr(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            ResolveError::UnsupportedCoordDtype(dt) => {
                write!(f, "unsupported coord dtype: {dt}")
            }
            ResolveError::MissingCoord(dim) => {
                write!(f, "missing coord array: {dim}")
            }
            ResolveError::OutOfBounds => {
                write!(f, "coord index out of bounds")
            }
            ResolveError::Zarr(msg) => {
                write!(f, "zarr error: {msg}")
            }
        }
    }
}

impl std::error::Error for ResolveError {}

pub(crate) trait CoordIndexResolver {
    fn index_range_for_value_range(
        &mut self,
        dim: &str,
        range: &ValueRange,
    ) -> Result<Option<IndexRange>, ResolveError>;
}
