use super::literals::strip_wrappers;
use crate::chunk_plan::prelude::*;

pub(super) fn expr_to_col_name(e: &Expr) -> Option<&str> {
    if let Expr::Column(name) = strip_wrappers(e) {
        Some(name.as_str())
    } else {
        None
    }
}

