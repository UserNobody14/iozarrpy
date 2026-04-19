//! Threshold-gated parallel iteration helpers.
//!
//! Rayon's `par_iter` is unconditional: even when the input has only one or two
//! items, entering the global pool, allocating a `Job`, and synchronising the
//! result still costs a few microseconds. For workloads where the per-item cost
//! is comparable to that overhead (small metadata jobs, single-chunk reads,
//! tiny variable plans, ...), forcing parallelism is a net loss.
//!
//! The [`MaybeParIter`] trait lets each call site declare a threshold below
//! which the iteration stays sequential and above which it dispatches to
//! `par_iter`. The threshold is intentionally explicit because the right value
//! depends entirely on per-item cost and must be tuned per call site.
//!
//! # Example
//!
//! ```ignore
//! use crate::shared::MaybeParIter;
//!
//! const PARALLEL_THRESHOLD: usize = 4;
//!
//! let results: Vec<U> = items
//!     .maybe_par_iter(PARALLEL_THRESHOLD)
//!     .map_collect(|item| process(item))?;
//! ```

use rayon::prelude::*;

/// Slice-like values that can be iterated either sequentially or in parallel
/// depending on a caller-supplied threshold.
pub trait MaybeParIter<'a> {
    type Item: Sync + 'a;

    /// Returns an iterator wrapper that runs sequentially when the underlying
    /// slice has fewer than `threshold` items, and via Rayon otherwise.
    fn maybe_par_iter(
        &'a self,
        threshold: usize,
    ) -> MaybePar<'a, Self::Item>;
}

impl<'a, T: Sync + 'a> MaybeParIter<'a> for [T] {
    type Item = T;

    fn maybe_par_iter(
        &'a self,
        threshold: usize,
    ) -> MaybePar<'a, T> {
        if self.len() < threshold {
            MaybePar::Seq(self)
        } else {
            MaybePar::Par(self)
        }
    }
}

impl<'a, T: Sync + 'a> MaybeParIter<'a> for Vec<T> {
    type Item = T;

    fn maybe_par_iter(
        &'a self,
        threshold: usize,
    ) -> MaybePar<'a, T> {
        self.as_slice().maybe_par_iter(threshold)
    }
}

/// Iterator wrapper produced by [`MaybeParIter::maybe_par_iter`]. The two
/// variants are dispatched to `Iterator` or `ParallelIterator` respectively.
pub enum MaybePar<'a, T: Sync> {
    Seq(&'a [T]),
    Par(&'a [T]),
}

impl<'a, T: Sync + 'a> MaybePar<'a, T> {
    /// Maps `f` over every item and collects the (fallible) results into a
    /// `Vec`. Sequential when the wrapper is `Seq`, parallel otherwise.
    pub fn map_collect<U, E, F>(
        self,
        f: F,
    ) -> Result<Vec<U>, E>
    where
        U: Send,
        E: Send,
        F: Fn(&T) -> Result<U, E> + Sync + Send,
    {
        match self {
            MaybePar::Seq(items) => {
                items.iter().map(f).collect()
            }
            MaybePar::Par(items) => {
                items.par_iter().map(f).collect()
            }
        }
    }
}
