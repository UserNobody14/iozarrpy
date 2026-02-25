use crate::errors::{
    BackendError, BackendResult,
};
use snafu::ResultExt;

const DEFAULT_MAX_CHUNK_ELEMS: usize = 50_000_000;

pub(crate) fn max_chunk_elems() -> usize {
    std::env::var("RAINBEAR_MAX_CHUNK_ELEMS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CHUNK_ELEMS)
}

pub(crate) fn checked_chunk_len(
    shape: &[u64],
) -> BackendResult<usize> {
    let mut acc: usize = 1;
    for &d in shape {
        let d_usize: usize = d.try_into().context(
            crate::errors::backend::ChunkDimTooLargeSnafu {
                dim: d,
                max: usize::MAX,
            }
        )?;
        acc = acc.checked_mul(d_usize).ok_or(
            BackendError::other(
                "chunk size overflow".to_string(),
            ),
        )?;
        if acc > max_chunk_elems() {
            return Err(BackendError::other(
                "refusing to allocate an extremely large chunk; set RAINBEAR_MAX_CHUNK_ELEMS to override".to_string(),
            ));
        }
    }
    Ok(acc)
}
