use super::types::{CoordScalar, IndexRange, ValueRange};

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

pub(crate) trait CoordIndexResolver {
    fn index_range_for_value_range(
        &mut self,
        dim: &str,
        range: &ValueRange,
    ) -> Result<Option<IndexRange>, ResolveError>;

    fn index_for_value(
        &mut self,
        dim: &str,
        value: &CoordScalar,
    ) -> Result<Option<u64>, ResolveError>;

    fn coord_read_count(&self) -> u64 {
        0
    }
}

