use super::selection::Emptyable;

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
