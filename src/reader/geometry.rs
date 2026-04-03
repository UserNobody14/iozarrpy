/// Compute strides for row-major indexing of an N-dimensional array.
///
/// For shape [a, b, c], strides are [b*c, c, 1].
#[inline]
pub fn compute_strides(
    shape: &[u64],
) -> Vec<u64> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1u64; shape.len()];
    for i in
        (0..shape.len().saturating_sub(1)).rev()
    {
        strides[i] =
            strides[i + 1] * shape[i + 1];
    }
    strides
}
