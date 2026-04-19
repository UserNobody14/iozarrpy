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
use smallvec::SmallVec;
use std::borrow::Borrow;

/// Interned string type used throughout the codebase for dimension/variable names.
pub type IStr = internment::Intern<str>;

/// Helper trait to create IStr from various string types
pub trait IntoIStr {
    fn istr(self) -> IStr;
}

pub trait FromIStr {
    fn from_istr(istr: IStr) -> Self;
}

// Traits and implementations for automatically mapping vecs of IStrs to vecs of strings and vice versa
pub trait FromManyIstrs<
    T: FromIStr,
    I: Borrow<IStr>,
> where
    Self: IntoIterator<Item = I>,
{
    fn from_istrs(self) -> Vec<T>;
}

pub trait IntoManyIstrs<T: IntoIStr> {
    fn into_istrs(self) -> Vec<IStr>;
}

impl<T: FromIStr> FromManyIstrs<T, IStr>
    for Vec<IStr>
{
    fn from_istrs(self) -> Vec<T> {
        self.into_iter()
            .map(|istr| {
                T::from_istr(
                    istr.borrow().clone(),
                )
            })
            .collect()
    }
}

impl<T: IntoIStr> IntoManyIstrs<T> for Vec<T> {
    fn into_istrs(self) -> Vec<IStr> {
        self.into_iter()
            .map(|t| t.istr())
            .collect()
    }
}

impl<'a, T: FromIStr> FromManyIstrs<T, &'a IStr>
    for &'a [IStr]
{
    fn from_istrs(self) -> Vec<T> {
        self.into_iter()
            .map(|istr| {
                T::from_istr(
                    istr.borrow().clone(),
                )
            })
            .collect()
    }
}

impl<T: IntoIStr + Clone> IntoManyIstrs<T>
    for &[T]
{
    fn into_istrs(self) -> Vec<IStr> {
        self.into_iter()
            .map(|t| t.clone().istr())
            .collect()
    }
}

// Smallvec (arbitrary length)
impl<T: FromIStr, I: Borrow<IStr>, const N: usize>
    FromManyIstrs<T, I> for SmallVec<[I; N]>
{
    fn from_istrs(self) -> Vec<T> {
        self.into_iter()
            .map(|istr| {
                T::from_istr(
                    istr.borrow().clone(),
                )
            })
            .collect()
    }
}

// impl FromIStr for &str {
//     fn from_istr(istr: IStr) -> Self {
//         istr.to_string().as_str()
//     }
// }

impl FromIStr for IStr {
    // No-op
    fn from_istr(istr: IStr) -> Self {
        istr
    }
}

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

impl IntoIStr for IStr {
    fn istr(self) -> IStr {
        self
    }
}

impl IntoIStr for &IStr {
    fn istr(self) -> IStr {
        *self
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
