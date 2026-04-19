#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::single_range_in_vec_init)]

use pyo3::prelude::*;

mod backend;
mod chunk_plan;
mod codec_compat;
mod errors;
mod meta;
mod py;
mod reader;
mod scan;
mod shared;
mod store;

#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod bench_internals;

pub(crate) use shared::PlannerStats;

use polars::prelude::*;

#[pymodule]
fn _core(
    py: Python<'_>,
    m: &Bound<PyModule>,
) -> PyResult<()> {
    codec_compat::ensure_zarr_compat_registered();

    // Initialize tokio-console subscriber for async profiling (when feature enabled)
    #[cfg(feature = "tokio-console")]
    {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            console_subscriber::init();
        });
    }

    // Register object store builders under rainbear._core.store
    // This allows users to create stores with full connection pooling
    pyo3_object_store::register_store_module(
        py,
        m,
        "rainbear._core",
        "store",
    )?;
    pyo3_object_store::register_exceptions_module(
        py,
        m,
        "rainbear._core",
        "exceptions",
    )?;

    py::init_module(m)
}
