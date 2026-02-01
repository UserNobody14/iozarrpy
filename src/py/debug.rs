//! Debug utilities for chunk planning inspection.

use pyo3::prelude::*;
#[pyfunction]
pub(crate) fn print_extension_info() -> String {
    "Rainbear extension module loaded successfully"
        .to_string()
}
