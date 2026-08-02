use crate::shared::{IStr, IntoIStr};

pub(crate) fn leaf_name(path: &str) -> IStr {
    path.rsplit('/')
        .next()
        .unwrap_or_default()
        .istr()
}
