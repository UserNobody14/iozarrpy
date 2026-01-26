use smallvec::SmallVec;
use zarrs::array::Array;

use crate::{IStr, IntoIStr};

pub(crate) fn leaf_name(path: &str) -> IStr {
    path.rsplit('/').next().unwrap_or_default().istr()
}

pub(crate) fn default_dims(n: usize) -> SmallVec<[IStr; 4]> {
    (0..n).map(|i| format!("dim_{i}").istr()).collect()
}

pub(crate) fn dims_for_array<TStorage: ?Sized>(array: &Array<TStorage>) -> Option<SmallVec<[IStr; 4]>> {
    if let Some(v) = array.attributes().get("_ARRAY_DIMENSIONS") {
        if let Some(list) = v.as_array() {
            let out: SmallVec<[IStr; 4]> = list
                .iter()
                .filter_map(|x| x.as_str().map(|s| s.istr()))
                .collect();
            if !out.is_empty() {
                return Some(out);
            }
        }
    }

    if let Some(names) = array.dimension_names() {
        let out: SmallVec<[IStr; 4]> = names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                n.as_ref()
                    .map(|s| s.istr())
                    .unwrap_or_else(|| format!("dim_{i}").istr())
            })
            .collect();
        return Some(out);
    }

    None
}

