use std::collections::HashMap;

use pyo3::prelude::*;

/// Configure how non-standard Zarr V3 codec names in metadata are interpreted.
///
/// Args:
///     aliases: Map from codec name as stored on disk to a zarrs canonical name.
///         Supported targets: ``\"numcodecs.bitround\"`` (or ``\"bitround\"``),
///         ``\"numcodecs.fixedscaleoffset\"`` (or ``\"fixedscaleoffset\"``).
///         The fixed-scale-offset target applies NumPy-style ``dtype`` / ``astype``
///         normalization before zarrs parses them.
#[pyfunction]
#[pyo3(signature = (*, aliases=None))]
pub fn configure_zarr_codecs(
    aliases: Option<HashMap<String, String>>,
) -> PyResult<()> {
    crate::codec_compat::set_codec_aliases(aliases.unwrap_or_default())
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(e)
        })
}
