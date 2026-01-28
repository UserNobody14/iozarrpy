use super::literals::strip_wrappers;
use crate::chunk_plan::prelude::*;
use crate::{IStr, IntoIStr};

pub(super) fn expr_to_col_name(
    e: &Expr,
) -> Option<IStr> {
    if let Expr::Column(name) = strip_wrappers(e)
    {
        Some(name.istr())
    } else {
        None
    }
}
