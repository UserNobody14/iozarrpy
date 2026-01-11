use pyo3::prelude::*;

const DEFAULT_MAX_CHUNK_ELEMS: usize = 50_000_000;

pub(crate) fn max_chunk_elems() -> usize {
    std::env::var("RAINBEAR_MAX_CHUNK_ELEMS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_CHUNK_ELEMS)
}

pub(crate) fn checked_chunk_len(shape: &[u64]) -> PyResult<usize> {
    let mut acc: usize = 1;
    for &d in shape {
        let d_usize: usize = d.try_into().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyMemoryError, _>(
                "chunk shape dimension does not fit in usize",
            )
        })?;
        acc = acc.checked_mul(d_usize).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyMemoryError, _>("chunk size overflow")
        })?;
        if acc > max_chunk_elems() {
            return Err(PyErr::new::<pyo3::exceptions::PyMemoryError, _>(
                "refusing to allocate an extremely large chunk; set RAINBEAR_MAX_CHUNK_ELEMS to override",
            ));
        }
    }
    Ok(acc)
}

