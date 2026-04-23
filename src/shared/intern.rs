use polars::prelude::PlSmallStr;
use smallvec::SmallVec;
use std::{borrow::Borrow, sync::Arc};

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
pub trait FromManyIstrs<I: Borrow<IStr>> {
    fn from_istrs(
        istrs: impl IntoIterator<Item = I>,
    ) -> Self;
}

pub trait IntoManyIstrs<T: IntoIStr> {
    fn into_istrs(self) -> Vec<IStr>;
}

impl<T: FromIStr> FromManyIstrs<IStr> for Vec<T> {
    fn from_istrs(
        istrs: impl IntoIterator<Item = IStr>,
    ) -> Vec<T> {
        istrs
            .into_iter()
            .map(|istr| {
                T::from_istr(*istr.borrow())
            })
            .collect()
    }
}

impl<T: IntoIStr> IntoManyIstrs<T> for Vec<T>
where
    T: IntoIStr,
{
    fn into_istrs(self) -> Vec<IStr> {
        self.into_iter()
            .map(|t| t.istr())
            .collect()
    }
}

impl<'a, T: FromIStr, const COUNT: usize>
    FromManyIstrs<&'a IStr> for [T; COUNT]
where
    T: FromIStr,
    'a: 'a,
{
    fn from_istrs(
        istrs: impl IntoIterator<Item = &'a IStr>,
    ) -> [T; COUNT] {
        return istrs
            .into_iter()
            .map(|istr| T::from_istr(*istr))
            .collect::<Vec<T>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Expected exactly COUNT elements"));
    }
}

impl<T: IntoIStr + Clone> IntoManyIstrs<T>
    for &[T]
{
    fn into_istrs(self) -> Vec<IStr> {
        self.iter()
            .map(|t| t.clone().istr())
            .collect()
    }
}

// Smallvec (arbitrary length)
impl<T: FromIStr, I: Borrow<IStr>, const N: usize>
    FromManyIstrs<I> for SmallVec<[T; N]>
{
    fn from_istrs(
        istrs: impl IntoIterator<Item = I>,
    ) -> SmallVec<[T; N]> {
        return istrs
            .into_iter()
            .map(|istr| {
                T::from_istr(*istr.borrow())
            })
            .collect::<Vec<T>>()
            .into();
    }
}

impl<T: IntoIStr> IntoManyIstrs<T> for &Arc<[T]>
where
    T: Clone + IntoIStr,
{
    fn into_istrs(self) -> Vec<IStr> {
        self.iter()
            .map(|t| t.clone().istr())
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

impl IntoIStr for PlSmallStr {
    fn istr(self) -> IStr {
        IStr::from(self.as_str())
    }
}

impl IntoIStr for &PlSmallStr {
    fn istr(self) -> IStr {
        IStr::from(self.clone().as_str())
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
