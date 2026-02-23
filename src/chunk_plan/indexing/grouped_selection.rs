//! Generic grouped selection types.
//!
//! `GroupedSelection` groups variables by their dimension signature to avoid
//! duplicating selection objects for variables that share the same dimensions.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::IStr;

use super::selection::Emptyable;
use super::types::DimSignature;

/// Trait for array-level selection types (lazy or concrete).
pub trait ArraySelectionType:
    Emptyable + Clone
{
}

/// Grouped selection: maps dimension signatures to shared selections.
///
/// Instead of storing one selection per variable, we store one selection per
/// unique dimension signature. Variables are mapped to their signature for lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedSelection<Sel> {
    by_dims: BTreeMap<Arc<DimSignature>, Sel>,
    var_to_sig: BTreeMap<IStr, Arc<DimSignature>>,
}

impl<Sel: ArraySelectionType>
    GroupedSelection<Sel>
{
    pub fn new() -> Self {
        Self {
            by_dims: BTreeMap::new(),
            var_to_sig: BTreeMap::new(),
        }
    }

    /// Iterate over variables and their selections.
    pub fn vars(
        &self,
    ) -> impl Iterator<Item = (&str, &Sel)> {
        self.var_to_sig.iter().filter_map(
            |(var, sig)| {
                self.by_dims.get(sig).map(|sel| {
                    (var.as_ref(), sel)
                })
            },
        )
    }

    /// Create a GroupedSelection from raw parts.
    pub(crate) fn from_parts(
        by_dims: BTreeMap<Arc<DimSignature>, Sel>,
        var_to_sig: BTreeMap<
            IStr,
            Arc<DimSignature>,
        >,
    ) -> Self {
        Self {
            by_dims,
            var_to_sig,
        }
    }
}

impl<Sel: ArraySelectionType> Default
    for GroupedSelection<Sel>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Sel: ArraySelectionType> Emptyable
    for GroupedSelection<Sel>
{
    fn empty() -> Self {
        Self::new()
    }

    fn is_empty(&self) -> bool {
        self.by_dims.is_empty()
            || self
                .by_dims
                .values()
                .all(|s| s.is_empty())
    }
}
