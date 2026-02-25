use super::selection::Emptyable;

/// Generic dataset selection enum.
///
/// Parameterized by the array selection type, used for concrete `DataArraySelection`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetSelectionBase<Sel> {
    /// No selection was made (equivalent to "select all").
    NoSelectionMade,
    /// Everything has been excluded.
    Empty,
    /// Standard selection mapping dimension signatures to selections.
    Selection(Sel),
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
