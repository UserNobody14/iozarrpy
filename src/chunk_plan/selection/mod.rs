

/// Selection (both lazy and concrete versions) for a single hyper-rectangle. Wraps ArraySubset
mod hyper_rectangle;
/// Selection for a concrete (dimensions and chunks resolved) array
/// Akin to ArraySubsetList, a Vec of HyperRectangle's
mod array;
/// Selection for an unresolved array