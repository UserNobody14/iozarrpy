use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod debug;
pub(crate) mod expr_extract;
mod selected_chunks;

pub(crate) fn init_module(
    m: &Bound<PyModule>,
) -> PyResult<()> {
    use crate::py::debug::print_extension_info;
    use crate::py::selected_chunks::{
        _selected_variables_debug,
        selected_chunks,
    };

    m.add_function(wrap_pyfunction!(
        print_extension_info,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        selected_chunks,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        _selected_variables_debug,
        m
    )?)?;

    m.add_class::<crate::backend::PyZarrBackend>(
    )?;
    m.add_class::<crate::backend::PyZarrBackendSync>(
    )?;

    Ok(())
}
