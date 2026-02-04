use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod debug;
pub(crate) mod expr_extract;

pub(crate) fn init_module(
    m: &Bound<PyModule>,
) -> PyResult<()> {
    use crate::py::debug::print_extension_info;

    m.add_function(wrap_pyfunction!(
        print_extension_info,
        m
    )?)?;

    m.add_class::<crate::backend::PyZarrBackend>(
    )?;
    m.add_class::<crate::backend::PyZarrBackendSync>(
    )?;
    m.add_class::<crate::backend::PyIcechunkBackend>(
    )?;

    Ok(())
}
