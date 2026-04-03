use std::fmt;

use smallvec::SmallVec;

use crate::{IStr, IntoIStr};

/// A structured path through the zarr hierarchy, stored as components
/// rather than a slash-delimited string.
///
/// e.g. `["model_a", "temperature"]` instead of `"model_a/temperature"`.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord,
)]
pub struct ZarrPath(SmallVec<[IStr; 4]>);

impl ZarrPath {
    pub fn root() -> Self {
        Self(SmallVec::new())
    }

    pub fn single(name: IStr) -> Self {
        let mut sv = SmallVec::new();
        sv.push(name);
        Self(sv)
    }

    pub fn from_components(
        components: impl IntoIterator<Item = IStr>,
    ) -> Self {
        Self(components.into_iter().collect())
    }

    /// Parse a slash-separated path string into components.
    /// Strips leading/trailing slashes, filters empty segments.
    pub fn parse(s: &str) -> Self {
        let components: SmallVec<[IStr; 4]> = s
            .split('/')
            .filter(|seg| !seg.is_empty())
            .map(|seg| seg.istr())
            .collect();
        Self(components)
    }

    pub fn components(&self) -> &[IStr] {
        &self.0
    }

    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn first(&self) -> Option<&IStr> {
        self.0.first()
    }

    pub fn leaf(&self) -> Option<&IStr> {
        self.0.last()
    }

    /// Return a new path with the first component removed.
    pub fn tail(&self) -> Self {
        if self.0.len() <= 1 {
            Self::root()
        } else {
            Self(self.0[1..].into())
        }
    }

    /// Return a new path with the last component removed.
    pub fn parent(&self) -> Self {
        if self.0.is_empty() {
            Self::root()
        } else {
            Self(self.0[..self.0.len() - 1].into())
        }
    }

    /// Append a component to produce a new path.
    pub fn push(&self, component: IStr) -> Self {
        let mut new = self.0.clone();
        new.push(component);
        Self(new)
    }

    /// Join with `/` to produce a flat string (for storage/Polars boundary).
    pub fn to_flat_string(&self) -> String {
        let parts: Vec<&str> =
            self.0.iter().map(|c| c.as_ref()).collect();
        parts.join("/")
    }

    /// Convert to `IStr` for interop with existing APIs.
    pub fn to_istr(&self) -> IStr {
        self.to_flat_string().istr()
    }
}

impl fmt::Display for ZarrPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_root() {
            write!(f, "/")
        } else {
            write!(f, "{}", self.to_flat_string())
        }
    }
}

impl From<&str> for ZarrPath {
    fn from(s: &str) -> Self {
        Self::parse(s)
    }
}

impl From<IStr> for ZarrPath {
    fn from(s: IStr) -> Self {
        let s_ref: &str = s.as_ref();
        Self::parse(s_ref)
    }
}

impl From<&IStr> for ZarrPath {
    fn from(s: &IStr) -> Self {
        let s_ref: &str = s.as_ref();
        Self::parse(s_ref)
    }
}

impl From<ZarrPath> for IStr {
    fn from(p: ZarrPath) -> Self {
        p.to_istr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple() {
        let p = ZarrPath::parse("model_a/temperature");
        assert_eq!(p.len(), 2);
        assert_eq!(p.components()[0].as_ref(), "model_a");
        assert_eq!(p.components()[1].as_ref(), "temperature");
    }

    #[test]
    fn parse_leading_slash() {
        let p = ZarrPath::parse("/model_a/temperature");
        assert_eq!(p.len(), 2);
        assert_eq!(p.to_flat_string(), "model_a/temperature");
    }

    #[test]
    fn parse_root() {
        let p = ZarrPath::parse("/");
        assert!(p.is_root());
    }

    #[test]
    fn parent_and_leaf() {
        let p = ZarrPath::parse("a/b/c");
        assert_eq!(p.parent().to_flat_string(), "a/b");
        let leaf: &str = p.leaf().unwrap().as_ref();
        assert_eq!(leaf, "c");
    }

    #[test]
    fn push() {
        let p = ZarrPath::parse("model_a");
        let child = p.push("temperature".istr());
        assert_eq!(child.to_flat_string(), "model_a/temperature");
    }

    #[test]
    fn tail() {
        let p = ZarrPath::parse("a/b/c");
        assert_eq!(p.tail().to_flat_string(), "b/c");
    }

    #[test]
    fn roundtrip() {
        let original = "level_1/level_2/var";
        let p = ZarrPath::parse(original);
        assert_eq!(p.to_flat_string(), original);
    }
}
