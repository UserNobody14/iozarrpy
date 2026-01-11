use super::errors::{CoordIndexResolver, ResolveError};
use super::prelude::*;
use super::types::{CoordScalar, IndexRange, ValueRange};

pub(crate) struct IdentityIndexResolver<'a> {
    meta: &'a ZarrDatasetMeta,
}

impl<'a> IdentityIndexResolver<'a> {
    pub(crate) fn new(meta: &'a ZarrDatasetMeta) -> Self {
        Self { meta }
    }
}

impl CoordIndexResolver for IdentityIndexResolver<'_> {
    fn index_range_for_value_range(
        &mut self,
        dim: &str,
        _range: &ValueRange,
    ) -> Result<Option<IndexRange>, ResolveError> {
        // Only supports index-like dims where coord array doesn't exist.
        if self.meta.arrays.contains_key(dim) {
            return Ok(None);
        }
        // Without an explicit shape for a pure index dim, we can't resolve in index space yet.
        Ok(None)
    }

    fn index_for_value(
        &mut self,
        _dim: &str,
        _value: &CoordScalar,
    ) -> Result<Option<u64>, ResolveError> {
        Ok(None)
    }
}

