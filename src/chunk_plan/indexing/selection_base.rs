use super::selection::{
    Emptyable, SetOperations,
};

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
    Selection(Sel),
}

impl<Sel> DatasetSelectionBase<Sel> {
    pub fn from_optional(
        sel: Option<Sel>,
    ) -> Self {
        if let Some(sel) = sel {
            Self::Selection(sel)
        } else {
            Self::Empty
        }
    }
}

impl<Sel: Emptyable> Emptyable
    for DatasetSelectionBase<Sel>
{
    /// Create an empty selection (selects nothing).
    fn is_empty(&self) -> bool {
        match self {
            Self::NoSelectionMade => false,
            Self::Empty => true,
            Self::Selection(sel) => {
                sel.is_empty()
            }
        }
    }

    fn empty() -> Self {
        Self::Empty
    }
}

impl<Sel: SetOperations + Clone> SetOperations
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
            (Self::NoSelectionMade, _)
            | (_, Self::NoSelectionMade) => {
                Self::NoSelectionMade
            }
            (Self::Empty, _)
            | (_, Self::Empty) => Self::Empty,
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
        match (self, other) {
            (Self::NoSelectionMade, _)
            | (_, Self::NoSelectionMade) => {
                Self::NoSelectionMade
            }
            (Self::Empty, _)
            | (_, Self::Empty) => Self::Empty,
            (
                Self::Selection(a),
                Self::Selection(b),
            ) => {
                let xor = a.exclusive_or(b);
                if xor.is_empty() {
                    Self::Empty
                } else {
                    Self::Selection(xor)
                }
            }
        }
    }
}
