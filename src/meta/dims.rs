use zarrs::array::Array;

pub(crate) fn leaf_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or_default().to_string()
}

pub(crate) fn default_dims(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("dim_{i}")).collect()
}

pub(crate) fn dims_for_array<TStorage: ?Sized>(array: &Array<TStorage>) -> Option<Vec<String>> {
    if let Some(v) = array.attributes().get("_ARRAY_DIMENSIONS") {
        if let Some(list) = v.as_array() {
            let out: Vec<String> = list
                .iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect();
            if !out.is_empty() {
                return Some(out);
            }
        }
    }

    if let Some(names) = array.dimension_names() {
        let out: Vec<String> = names
            .iter()
            .enumerate()
            .map(|(i, n)| n.clone().unwrap_or_else(|| format!("dim_{i}")))
            .collect();
        return Some(out);
    }

    None
}

