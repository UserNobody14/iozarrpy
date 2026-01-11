use super::types::{IndexRange, ValueRange};

#[derive(Debug)]
pub(crate) enum CompileError {
    Unsupported(String),
    MissingPrimaryDims(String),
}

#[derive(Debug)]
pub(crate) enum ResolveError {
    UnsupportedCoordDtype(String),
    MissingCoord(String),
    OutOfBounds,
    Zarr(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::UnsupportedCoordDtype(dt) => {
                write!(f, "unsupported coord dtype: {dt}")
            }
            ResolveError::MissingCoord(dim) => write!(f, "missing coord array: {dim}"),
            ResolveError::OutOfBounds => write!(f, "coord index out of bounds"),
            ResolveError::Zarr(msg) => write!(f, "zarr error: {msg}"),
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

    fn coord_read_count(&self) -> u64 {
        0
    }
}

