use zarrs::array::Array;

use crate::IStr;

pub(crate) fn compute_var_chunk_info(
    primary_chunk_indices: &[u64],
    primary_chunk_shape: &[u64],
    primary_dims: &[IStr],
    var_dims: &[IStr],
    var_array: &Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let mut var_chunk_indices =
        Vec::with_capacity(var_dims.len());
    let mut offsets =
        Vec::with_capacity(var_dims.len());

    for var_dim in var_dims {
        if let Some(primary_d) = primary_dims
            .iter()
            .position(|pd| pd == var_dim)
        {
            let primary_chunk_idx =
                primary_chunk_indices[primary_d];
            let primary_chunk_size =
                primary_chunk_shape[primary_d];
            let global_start = primary_chunk_idx
                * primary_chunk_size;

            let var_dim_idx = var_dims
                .iter()
                .position(|vd| vd == var_dim)
                .unwrap();

            let zero_indices: Vec<u64> =
                vec![0; var_dims.len()];
            let var_regular_chunk_shape =
                var_array
                    .chunk_shape(&zero_indices)
                    .map_err(to_string_err)?;
            let var_chunk_size =
                var_regular_chunk_shape
                    [var_dim_idx]
                    .get();

            let var_chunk_idx =
                global_start / var_chunk_size;
            let offset =
                global_start % var_chunk_size;

            var_chunk_indices.push(var_chunk_idx);
            offsets.push(offset);
        } else {
            return Err(format!(
                "variable dimension {var_dim} not found in primary dims"
            ));
        }
    }

    Ok((var_chunk_indices, offsets))
}

fn to_string_err<E: std::fmt::Display>(
    e: E,
) -> String {
    e.to_string()
}
