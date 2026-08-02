#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::single_range_in_vec_init)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::needless_pass_by_value)]
#![warn(clippy::redundant_allocation)]
#![warn(clippy::borrowed_box)]
#![warn(clippy::inefficient_to_string)]
#![warn(clippy::needless_range_loop)]
#![warn(clippy::manual_ok_or)]
#![warn(clippy::option_if_let_else)]
// Additional perf lints (pedantic / nursery groups, opted-in)
#![warn(clippy::large_types_passed_by_value)]
#![warn(clippy::iter_with_drain)]
#![warn(clippy::redundant_clone)]
#![warn(clippy::needless_pass_by_ref_mut)]
#![warn(clippy::manual_let_else)]
#![warn(clippy::trivial_regex)]
// Hot path / allocation / copy avoidance (beyond the lints above)
#![warn(clippy::needless_collect)]
#![warn(clippy::slow_vector_initialization)]
#![warn(clippy::stable_sort_primitive)]
#![warn(clippy::unnecessary_to_owned)]
#![warn(clippy::or_fun_call)]
#![warn(clippy::naive_bytecount)]
#![warn(clippy::single_char_pattern)]
#![warn(clippy::single_char_add_str)]
#![warn(clippy::unused_peekable)]
#![warn(clippy::map_clone)]
#![warn(clippy::unnecessary_sort_by)]
#![warn(clippy::vec_init_then_push)]
#![warn(clippy::unused_rounding)]
#![warn(clippy::reserve_after_initialization)]
#![warn(clippy::large_enum_variant)]
#![warn(clippy::box_collection)]
#![warn(clippy::linkedlist)]
// Async / Tokio / PyO3 boundary lints
#![warn(clippy::await_holding_invalid_type)]
#![warn(clippy::large_futures)]
#![warn(clippy::await_holding_lock)]
#![warn(clippy::await_holding_refcell_ref)]
#![warn(clippy::future_not_send)]
#![warn(clippy::let_underscore_future)]
#![warn(clippy::unused_async)]
#![warn(clippy::async_yields_async)]
#![warn(clippy::manual_async_fn)]
#![warn(clippy::redundant_async_block)]
#![warn(clippy::significant_drop_in_scrutinee)]
#![warn(clippy::mutex_integer)]
#![warn(clippy::disallowed_methods)]
#![warn(clippy::dbg_macro)]

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
