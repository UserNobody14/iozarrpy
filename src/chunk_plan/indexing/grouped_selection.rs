//! Generic grouped selection types.
//!
//! This module provides `GroupedSelection` and `DatasetSelectionBase`, which group
//! variables by their dimension signature to avoid duplicating selection objects
//! for variables that share the same dimensions.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::IStr;
use crate::meta::ZarrMeta;

use super::selection::Emptyable;
use super::types::DimSignature;

/// Trait for array-level selection types (lazy or concrete).
///
/// This trait unifies `LazyArraySelection` and `DataArraySelection` so they
/// can be used generically in `GroupedSelection` and `DatasetSelectionBase`.
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
    /// Selection by dimension signature (shared via Arc).
    by_dims: BTreeMap<Arc<DimSignature>, Sel>,
    /// Variable name to signature lookup.
    var_to_sig: BTreeMap<IStr, Arc<DimSignature>>,
}

impl<Sel: ArraySelectionType>
    GroupedSelection<Sel>
{
    /// Create a new empty grouped selection.
    pub fn new() -> Self {
        Self {
            by_dims: BTreeMap::new(),
            var_to_sig: BTreeMap::new(),
        }
    }

    /// Create a grouped selection for variables with the given selection.
    ///
    /// Groups variables by their dimension signature from metadata.
    pub fn for_vars_with_selection(
        vars: impl IntoIterator<Item = IStr>,
        meta: &ZarrMeta,
        sel: Sel,
    ) -> Self {
        let mut by_dims: BTreeMap<
            Arc<DimSignature>,
            Sel,
        > = BTreeMap::new();
        let mut var_to_sig: BTreeMap<
            IStr,
            Arc<DimSignature>,
        > = BTreeMap::new();
        // Cache of signature -> Arc for deduplication
        let mut sig_cache: BTreeMap<
            DimSignature,
            Arc<DimSignature>,
        > = BTreeMap::new();

        for var in vars {
            let sig = if let Some(array_meta) =
                meta.array_by_path(var.clone())
            {
                DimSignature::from_dims_only(
                    array_meta.dims.clone(),
                )
            } else {
                // Variable not found in meta - use empty signature
                DimSignature::from_dims_only(
                    smallvec::SmallVec::new(),
                )
            };

            let sig_arc = sig_cache
                .entry(sig.clone())
                .or_insert_with(|| Arc::new(sig))
                .clone();

            var_to_sig
                .insert(var, sig_arc.clone());

            // Only insert the selection once per signature
            by_dims
                .entry(sig_arc)
                .or_insert_with(|| sel.clone());
        }

        Self {
            by_dims,
            var_to_sig,
        }
    }

    pub fn to_optional(self) -> Option<Self> {
        if self.is_empty() {
            None
        } else {
            Some(self)
        }
    }

    /// Check if there are no variables in this selection.
    pub fn is_empty(&self) -> bool {
        self.var_to_sig.is_empty()
    }

    /// Iterate over variables and their selections.
    ///
    /// Note: Multiple variables may point to the same selection object.
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

    /// Iterate over unique (signature, selection) pairs.
    pub fn by_signature(
        &self,
    ) -> impl Iterator<Item = (&DimSignature, &Sel)>
    {
        self.by_dims
            .iter()
            .map(|(sig, sel)| (sig.as_ref(), sel))
    }

    /// Get the internal by_dims map (for materialization).
    pub(crate) fn by_dims(
        &self,
    ) -> &BTreeMap<Arc<DimSignature>, Sel> {
        &self.by_dims
    }

    /// Get the internal var_to_sig map (for materialization).
    pub(crate) fn var_to_sig(
        &self,
    ) -> &BTreeMap<IStr, Arc<DimSignature>> {
        &self.var_to_sig
    }

    /// Create a GroupedSelection from raw parts (for materialization).
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

