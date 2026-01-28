pub(crate) fn compute_strides(
    chunk_shape: &[u64],
) -> Vec<u64> {
    let mut strides =
        vec![1u64; chunk_shape.len()];
    for i in (0..chunk_shape.len()).rev() {
        if i + 1 < chunk_shape.len() {
            strides[i] = strides[i + 1]
                * chunk_shape[i + 1];
        }
    }
    strides
}
