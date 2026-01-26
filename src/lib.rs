use pyo3::prelude::*;

mod chunk_plan;
mod meta;
mod py;
mod reader;
mod scan;
mod source;
mod store;

#[pymodule]
fn _core(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Register object store builders under rainbear._core.store
    // This allows users to create stores with full connection pooling
    pyo3_object_store::register_store_module(py, m, "rainbear._core", "store")?;
    pyo3_object_store::register_exceptions_module(py, m, "rainbear._core", "exceptions")?;

    py::init_module(m)
}
