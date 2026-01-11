use super::literals::strip_wrappers;
use super::prelude::*;

pub(super) fn expr_to_col_name(e: &Expr) -> Option<&str> {
    match strip_wrappers(e) {
        Expr::Column(name) => Some(name.as_str()),
        _ => None,
    }
}

