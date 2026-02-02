use pyo3::prelude::*;

mod backend;
mod chunk_plan;
mod meta;
mod py;
mod reader;
mod scan;
mod store;

use polars::prelude::*;

/// Interned string type used throughout the codebase for dimension/variable names.
/// Uses `ArcIntern` for automatic deduplication, O(1) equality, and reference counting.
pub type IStr = internment::ArcIntern<str>;

/// Helper trait to create IStr from various string types
pub trait IntoIStr {
    fn istr(self) -> IStr;
}

pub trait FromIStr {
    fn from_istr(istr: IStr) -> Self;
}

// impl FromIStr for &str {
//     fn from_istr(istr: IStr) -> Self {
//         istr.to_string().as_str()
//     }
// }

impl FromIStr for String {
    fn from_istr(istr: IStr) -> Self {
        istr.clone().to_string()
    }
}

impl FromIStr for PlSmallStr {
    fn from_istr(istr: IStr) -> Self {
        PlSmallStr::from(istr.clone().to_string())
    }
}
impl IntoIStr for &str {
    fn istr(self) -> IStr {
        IStr::from(self)
    }
}

impl IntoIStr for String {
    fn istr(self) -> IStr {
        IStr::from(self.as_str())
    }
}

impl IntoIStr for &String {
    fn istr(self) -> IStr {
        IStr::from(self.as_str())
    }
}

#[pymodule]
fn _core(
    py: Python<'_>,
    m: &Bound<PyModule>,
) -> PyResult<()> {
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
