use pyo3::wrap_pyfunction;



mod zarr_meta;
mod zarr_source;
mod zarr_store;
mod test_utils;

use pyo3::prelude::*;

use crate::zarr_source::ZarrSource;

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from rainbear!".to_string()
}


#[pymodule]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_bin, m)?)?;
    m.add_function(wrap_pyfunction!(test_utils::_create_demo_store, m)?)?;

    m.add_class::<ZarrSource>()?;

    Ok(())
}