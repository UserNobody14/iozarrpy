use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod debug;
pub(crate) mod expr_extract;
mod scan_async;
mod selected_chunks;

pub(crate) fn init_module(m: &Bound<PyModule>) -> PyResult<()> {
    use crate::py::debug::{
        _debug_chunk_planning, _debug_chunk_planning_async, _debug_coord_array, _debug_expr_ast,
        _debug_literal_conversion, print_extension_info,
    };
    use crate::py::scan_async::scan_zarr_async;
    use crate::py::selected_chunks::{_selected_chunks_debug, _selected_variables_debug, selected_chunks};

    m.add_function(wrap_pyfunction!(print_extension_info, m)?)?;
    m.add_function(wrap_pyfunction!(selected_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(_selected_chunks_debug, m)?)?;
    m.add_function(wrap_pyfunction!(_selected_variables_debug, m)?)?;
    m.add_function(wrap_pyfunction!(_debug_expr_ast, m)?)?;
    m.add_function(wrap_pyfunction!(_debug_chunk_planning, m)?)?;
    m.add_function(wrap_pyfunction!(_debug_chunk_planning_async, m)?)?;
    m.add_function(wrap_pyfunction!(_debug_coord_array, m)?)?;
    m.add_function(wrap_pyfunction!(_debug_literal_conversion, m)?)?;
    m.add_function(wrap_pyfunction!(scan_zarr_async, m)?)?;

    m.add_class::<crate::source::ZarrSource>()?;

    Ok(())
}
