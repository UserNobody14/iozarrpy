/// Selection for a concrete (dimensions and chunks resolved) array
/// Akin to ArraySubsetList, a Vec of HyperRectangle's
pub(crate) mod array;
/// Selection (both lazy and concrete versions) for a single hyper-rectangle. Wraps ArraySubset
pub(crate) mod hyper_rectangle;
/// Selection for a (potentially multi-level or flat) tree of arrays
pub(crate) mod tree;
