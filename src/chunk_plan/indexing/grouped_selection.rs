//! Generic grouped selection types.
//!
//! This module provides `GroupedSelection` and `DatasetSelectionBase`, which group
//! variables by their dimension signature to avoid duplicating selection objects
//! for variables that share the same dimensions.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::IStr;
use crate::meta::ZarrMeta;

use super::selection::{
    Emptyable, SetOperations,
};
use super::types::DimSignature;

/// Trait for array-level selection types (lazy or concrete).
///
/// This trait unifies `LazyArraySelection` and `DataArraySelection` so they
/// can be used generically in `GroupedSelection` and `DatasetSelectionBase`.
pub trait ArraySelectionType:
    SetOperations + Clone
{
    /// Create a selection that selects all indices.
    fn all() -> Self;
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

    /// Create a grouped selection for variables with all indices selected.
    pub fn all_for_vars(
        vars: impl IntoIterator<Item = IStr>,
        meta: &ZarrMeta,
    ) -> Self {
        Self::for_vars_with_selection(
            vars,
            meta,
            Sel::all(),
        )
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

impl<Sel: ArraySelectionType> SetOperations
    for GroupedSelection<Sel>
{
    fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        let mut by_dims = self.by_dims.clone();
        let mut var_to_sig =
            self.var_to_sig.clone();

        // Add variables from other
        for (var, sig) in &other.var_to_sig {
            var_to_sig
                .entry(var.clone())
                .or_insert_with(|| sig.clone());
        }

        // Union selections by signature
        for (sig, sel) in &other.by_dims {
            by_dims
                .entry(sig.clone())
                .and_modify(|existing| {
                    *existing =
                        existing.union(sel)
                })
                .or_insert_with(|| sel.clone());
        }

        Self {
            by_dims,
            var_to_sig,
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::new();
        }

        let mut by_dims = BTreeMap::new();
        let mut var_to_sig = BTreeMap::new();

        // Only keep variables that are in both
        for (var, sig) in &self.var_to_sig {
            if other.var_to_sig.contains_key(var)
            {
                var_to_sig.insert(
                    var.clone(),
                    sig.clone(),
                );
            }
        }

        // Intersect selections by signature
        for (sig, sel_a) in &self.by_dims {
            if let Some(sel_b) =
                other.by_dims.get(sig)
            {
                let intersected =
                    sel_a.intersect(sel_b);
                if !intersected.is_empty() {
                    by_dims.insert(
                        sig.clone(),
                        intersected,
                    );
                }
            }
        }

        Self {
            by_dims,
            var_to_sig,
        }
    }

    fn difference(&self, other: &Self) -> Self {
        if self.is_empty() {
            return Self::new();
        }
        if other.is_empty() {
            return self.clone();
        }

        // For difference A \ B:
        // - Variables only in A: keep with original selection
        // - Variables in both A and B: compute selection difference
        //   - If difference is non-empty, keep with differenced selection
        //   - If difference is empty (e.g., all - all), exclude the variable
        //
        // Note: When both selections are "all", all.difference(all) returns empty
        // (handled in LazyArraySelection::difference), so selector XOR works correctly.

        let mut by_dims: BTreeMap<
            Arc<DimSignature>,
            Sel,
        > = BTreeMap::new();
        let mut var_to_sig: BTreeMap<
            IStr,
            Arc<DimSignature>,
        > = BTreeMap::new();

        // Track which signatures come from variables-only-in-self
        let mut original_sigs: std::collections::HashSet<
            Arc<DimSignature>,
        > = std::collections::HashSet::new();

        // First pass: handle variables only in self
        for (var, sig) in &self.var_to_sig {
            if !other.var_to_sig.contains_key(var)
            {
                var_to_sig.insert(
                    var.clone(),
                    sig.clone(),
                );
                if let Some(sel) =
                    self.by_dims.get(sig)
                {
                    by_dims
                        .entry(sig.clone())
                        .or_insert_with(|| {
                            sel.clone()
                        });
                    original_sigs
                        .insert(sig.clone());
                }
            }
        }

        // Second pass: handle variables in both
        for (var, sig) in &self.var_to_sig {
            if other.var_to_sig.contains_key(var)
            {
                // Don't overwrite signatures from variables-only-in-self
                if original_sigs.contains(sig) {
                    continue;
                }

                if let (
                    Some(sel_a),
                    Some(sel_b),
                ) = (
                    self.by_dims.get(sig),
                    other.by_dims.get(sig),
                ) {
                    let diff =
                        sel_a.difference(sel_b);
                    if !diff.is_empty() {
                        var_to_sig.insert(
                            var.clone(),
                            sig.clone(),
                        );
                        by_dims
                            .entry(sig.clone())
                            .or_insert_with(
                                || diff,
                            );
                    }
                    // If diff is empty, the variable is excluded (correct for selector XOR)
                }
            }
        }

        Self {
            by_dims,
            var_to_sig,
        }
    }

    fn exclusive_or(&self, other: &Self) -> Self {
        self.difference(other)
            .union(&other.difference(self))
    }
}

/// Generic dataset selection enum.
///
/// This is parameterized by the array selection type, allowing the same structure
/// to be used for both lazy (`LazyArraySelection`) and concrete (`DataArraySelection`)
/// selections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetSelectionBase<Sel> {
    /// No selection was made (equivalent to "select all").
    NoSelectionMade,
    /// Everything has been excluded.
    Empty,
    /// Standard selection mapping dimension signatures to selections.
    Selection(GroupedSelection<Sel>),
}

impl<Sel: ArraySelectionType> Default
    for DatasetSelectionBase<Sel>
{
    fn default() -> Self {
        Self::NoSelectionMade
    }
}

impl<Sel: ArraySelectionType>
    DatasetSelectionBase<Sel>
{
    /// Create an empty selection (selects nothing).
    pub fn empty() -> Self {
        Self::Empty
    }

    // TODO: consider removing?
    // /// Returns true if this is an empty selection.
    // pub fn is_empty_selection(&self) -> bool {
    //     matches!(self, Self::Empty)
    // }

    // /// Iterate over variables and their selections.
    // pub fn vars(
    //     &self,
    // ) -> Box<dyn Iterator<Item = (&str, &Sel)> + '_>
    // {
    //     match self {
    //         Self::Selection(sel) => {
    //             Box::new(sel.vars())
    //         }
    //         Self::NoSelectionMade
    //         | Self::Empty => {
    //             Box::new(std::iter::empty())
    //         }
    //     }
    // }

    // /// Get the selection for a specific variable.
    // pub fn get(&self, var: &str) -> Option<&Sel> {
    //     match self {
    //         Self::Selection(sel) => sel.get(var),
    //         Self::NoSelectionMade
    //         | Self::Empty => None,
    //     }
    // }

    // /// Get the inner grouped selection, if present.
    // pub fn as_selection(
    //     &self,
    // ) -> Option<&GroupedSelection<Sel>> {
    //     match self {
    //         Self::Selection(sel) => Some(sel),
    //         _ => None,
    //     }
    // }

    /// Create a selection for variables with the given selection.
    pub fn for_vars_with_selection(
        vars: impl IntoIterator<Item = IStr>,
        meta: &ZarrMeta,
        sel: Sel,
    ) -> Self {
        let grouped =
            GroupedSelection::for_vars_with_selection(
                vars, meta, sel,
            );
        if grouped.is_empty() {
            Self::Empty
        } else {
            Self::Selection(grouped)
        }
    }

    /// Create a selection for variables with all indices selected.
    pub fn all_for_vars(
        vars: impl IntoIterator<Item = IStr>,
        meta: &ZarrMeta,
    ) -> Self {
        let grouped =
            GroupedSelection::all_for_vars(
                vars, meta,
            );
        if grouped.is_empty() {
            Self::Empty
        } else {
            Self::Selection(grouped)
        }
    }
}

impl<Sel: ArraySelectionType> Emptyable
    for DatasetSelectionBase<Sel>
{
    fn empty() -> Self {
        Self::Empty
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

impl<Sel: ArraySelectionType> SetOperations
    for DatasetSelectionBase<Sel>
{
    fn union(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::NoSelectionMade, _)
            | (_, Self::NoSelectionMade) => {
                Self::NoSelectionMade
            }
            (Self::Empty, x)
            | (x, Self::Empty) => x.clone(),
            (
                Self::Selection(a),
                Self::Selection(b),
            ) => {
                let unioned = a.union(b);
                if unioned.is_empty() {
                    Self::Empty
                } else {
                    Self::Selection(unioned)
                }
            }
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::NoSelectionMade, x)
            | (x, Self::NoSelectionMade) => {
                x.clone()
            }
            (Self::Empty, _)
            | (_, Self::Empty) => Self::Empty,
            (
                Self::Selection(a),
                Self::Selection(b),
            ) => {
                let intersected = a.intersect(b);
                if intersected.is_empty() {
                    Self::Empty
                } else {
                    Self::Selection(intersected)
                }
            }
        }
    }

    fn difference(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::NoSelectionMade, _) => {
                Self::NoSelectionMade
            }
            (_, Self::NoSelectionMade) => {
                Self::Empty
            }
            (Self::Empty, _) => Self::Empty,
            (x, Self::Empty) => x.clone(),
            (
                Self::Selection(a),
                Self::Selection(b),
            ) => {
                let diff = a.difference(b);
                if diff.is_empty() {
                    Self::Empty
                } else {
                    Self::Selection(diff)
                }
            }
        }
    }

    fn exclusive_or(&self, other: &Self) -> Self {
        self.difference(other)
            .union(&other.difference(self))
    }
}
