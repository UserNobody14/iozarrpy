//! Set-operation traits used by the lazy expression-compilation pipeline.
//!
//! The eager value-space selection types (`DataArraySelection`,
//! `DatasetSelection`, `GroupedSelection`) were removed when the planner
//! switched to driving [`crate::chunk_plan::indexing::builder::GridJoinTreeBuilder`]
//! directly from `ExprPlan`. The traits below remain because
//! [`crate::chunk_plan::exprs::expr_plan::ExprPlan`] and
//! [`crate::chunk_plan::indexing::lazy_selection::LazyArraySelection`] still
//! express boolean combinators in terms of them.

pub trait Emptyable {
    fn empty() -> Self;
    fn is_empty(&self) -> bool;
}

/// Operations for sets of selections.
#[allow(dead_code)] // `exclusive_or` is not called on a `SetOperations` trait object yet
pub trait SetOperations: Emptyable {
    fn union(&self, other: &Self) -> Self;
    fn intersect(&self, other: &Self) -> Self;
    fn difference(&self, other: &Self) -> Self;
    fn exclusive_or(&self, other: &Self) -> Self;
}
