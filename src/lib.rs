use pyo3::prelude::*;

mod chunk_plan;
mod error;
mod meta;
mod py;
mod reader;
mod scan;
mod source;
mod store;

#[pymodule]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    py::init_module(m)
}
