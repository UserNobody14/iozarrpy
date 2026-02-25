/// Selection for an array, expressed as a disjunction (OR) of hyper-rectangles.
/// Describes actual homogenous arrays and their chunks
/// Contains no information about which variables are selected
/// Akin to ArraySubsetList, a Vec of HyperRectangle's
pub(crate) mod array;

/// Selection for a group of variables with the same array selection and dimensions
pub(crate) mod homogenous_group;

/// Selection for a (flat) set of homogenous groups,
/// each with a different chunk grid (as transparent as possible)
pub(crate) mod flat_set;

/// Selection for a tree of flat sets
pub(crate) mod tree;

/// A "constraint", basically a predicate that can be applied to one or more of the above
/// to produce a new selection
pub(crate) mod constraint;
