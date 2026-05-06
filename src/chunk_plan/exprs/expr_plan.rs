//! `ExprPlan` — eagerly-resolved expression plan.
//!
//! `Active` plans carry the variables an expression touches plus a concrete
//! [`RectangleSet`] of in-bounds index cells. Set operations
//! (intersect/union/difference/exclusive_or/boolean_not) delegate directly
//! to the `RectangleSet` impls — there is no separate lazy IR layer.
//!
//! `RectangleSet` instances with empty `dims` act as a "vars-only" sentinel
//! produced by [`ExprPlan::unconstrained_vars`]; they impose no rect filter
//! and the combine helpers treat them as identity.

use std::collections::BTreeSet;

use smallvec::SmallVec;

use crate::chunk_plan::indexing::index_set::RectangleSet;
use crate::shared::IStr;

/// Which variables an expression references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VarSet {
    /// All variables (identity for intersect, absorbing for union).
    All,
    /// Specific named variables.
    Specific(SmallVec<[IStr; 8]>),
}

impl VarSet {
    pub(crate) fn single(name: IStr) -> Self {
        let mut sv = SmallVec::new();
        sv.push(name);
        Self::Specific(sv)
    }

    pub(crate) fn from_vec(v: Vec<IStr>) -> Self {
        Self::Specific(v.into())
    }

    pub(crate) fn is_empty(&self) -> bool {
        matches!(self, Self::Specific(v) if v.is_empty())
    }

    pub(crate) fn intersect(
        &self,
        other: &Self,
    ) -> Self {
        match (self, other) {
            (Self::All, x) | (x, Self::All) => {
                x.clone()
            }
            (
                Self::Specific(a),
                Self::Specific(b),
            ) => {
                let b_set: BTreeSet<IStr> =
                    b.iter().copied().collect();
                let v: SmallVec<[IStr; 8]> = a
                    .iter()
                    .filter(|v| b_set.contains(v))
                    .copied()
                    .collect();
                Self::Specific(v)
            }
        }
    }

    pub(crate) fn union(
        &self,
        other: &Self,
    ) -> Self {
        match (self, other) {
            (Self::All, _) | (_, Self::All) => {
                Self::All
            }
            (
                Self::Specific(a),
                Self::Specific(b),
            ) => {
                let mut v: SmallVec<[IStr; 8]> =
                    a.iter().copied().collect();
                let mut seen: BTreeSet<IStr> =
                    v.iter().copied().collect();
                for &item in b {
                    if seen.insert(item) {
                        v.push(item);
                    }
                }
                Self::Specific(v)
            }
        }
    }

    pub(crate) fn difference(
        &self,
        other: &Self,
    ) -> Self {
        match (self, other) {
            (_, Self::All) => {
                Self::Specific(SmallVec::new())
            }
            (x, Self::Specific(b))
                if b.is_empty() =>
            {
                x.clone()
            }
            (Self::All, _) => Self::All,
            (
                Self::Specific(a),
                Self::Specific(b),
            ) => {
                let b_set: BTreeSet<IStr> =
                    b.iter().copied().collect();
                let v: SmallVec<[IStr; 8]> = a
                    .iter()
                    .filter(|v| !b_set.contains(v))
                    .copied()
                    .collect();
                Self::Specific(v)
            }
        }
    }
}

/// Result of compiling a Polars expression for chunk planning.
#[derive(Debug, Clone)]
pub enum ExprPlan {
    /// No selection made (identity for intersect, absorbing for union).
    NoConstraint,
    /// Everything excluded.
    Empty,
    /// Active constraints on dimensions + which variables are needed.
    Active {
        vars: VarSet,
        rects: RectangleSet,
    },
}

impl ExprPlan {
    /// Active plan that carries variables but no rect constraint. The
    /// `RectangleSet` is the empty-dims sentinel; combine helpers treat it
    /// as identity so var-only plans don't filter the index cube.
    pub(crate) fn unconstrained_vars(
        vars: VarSet,
    ) -> Self {
        if vars.is_empty() {
            Self::NoConstraint
        } else {
            Self::Active {
                vars,
                rects: RectangleSet::empty(
                    SmallVec::new(),
                    SmallVec::new(),
                ),
            }
        }
    }

    /// Build an Active plan with concrete rectangles. Empty rect set ⇒
    /// `Empty`.
    pub(crate) fn active(
        vars: VarSet,
        rects: RectangleSet,
    ) -> Self {
        if rects.is_empty() && !rects.dims.is_empty() {
            Self::Empty
        } else {
            Self::Active { vars, rects }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Replace vars with explicit refs while keeping rectangles.
    pub(crate) fn with_vars(
        self,
        vars: VarSet,
    ) -> Self {
        match self {
            Self::NoConstraint => {
                Self::unconstrained_vars(vars)
            }
            Self::Empty => Self::Empty,
            Self::Active { rects, .. } => {
                if vars.is_empty() {
                    Self::NoConstraint
                } else {
                    Self::Active { vars, rects }
                }
            }
        }
    }

    /// Add variables without changing rectangles.
    pub(crate) fn add_vars(
        &self,
        extra: VarSet,
    ) -> Self {
        match self {
            Self::NoConstraint => {
                Self::NoConstraint
            }
            Self::Empty => Self::Empty,
            Self::Active { vars, rects } => {
                Self::Active {
                    vars: vars.union(&extra),
                    rects: rects.clone(),
                }
            }
        }
    }

    pub(crate) fn intersect(
        &self,
        other: &Self,
    ) -> Self {
        match (self, other) {
            (Self::NoConstraint, x)
            | (x, Self::NoConstraint) => {
                x.clone()
            }
            (Self::Empty, _)
            | (_, Self::Empty) => Self::Empty,
            (
                Self::Active {
                    vars: va,
                    rects: ra,
                },
                Self::Active {
                    vars: vb,
                    rects: rb,
                },
            ) => Self::active(
                va.intersect(vb),
                combine_rects(ra, rb, |a, b| {
                    a.intersect(b)
                }),
            ),
        }
    }

    pub(crate) fn union(
        &self,
        other: &Self,
    ) -> Self {
        match (self, other) {
            (Self::NoConstraint, _)
            | (_, Self::NoConstraint) => {
                Self::NoConstraint
            }
            (Self::Empty, x)
            | (x, Self::Empty) => x.clone(),
            (
                Self::Active {
                    vars: va,
                    rects: ra,
                },
                Self::Active {
                    vars: vb,
                    rects: rb,
                },
            ) => Self::Active {
                vars: va.union(vb),
                rects: combine_rects(
                    ra,
                    rb,
                    |a, b| a.union(b),
                ),
            },
        }
    }

    pub(crate) fn difference(
        &self,
        other: &Self,
    ) -> Self {
        match (self, other) {
            (Self::Empty, _) => Self::Empty,
            (x, Self::Empty) => x.clone(),
            (_, Self::NoConstraint) => Self::Empty,
            (Self::NoConstraint, _) => {
                Self::NoConstraint
            }
            (
                Self::Active {
                    vars: va,
                    rects: ra,
                },
                Self::Active {
                    vars: vb,
                    rects: rb,
                },
            ) => {
                // Both vars-only: the difference is purely about variables.
                if ra.dims.is_empty()
                    && rb.dims.is_empty()
                {
                    let v_diff =
                        va.difference(vb);
                    if v_diff.is_empty() {
                        return Self::Empty;
                    }
                    return Self::Active {
                        vars: v_diff,
                        rects: ra.clone(),
                    };
                }
                // Filter A by NOT B: keep A's vars.
                Self::active(
                    va.clone(),
                    combine_rects(
                        ra,
                        rb,
                        |a, b| a.difference(b),
                    ),
                )
            }
        }
    }

    pub(crate) fn exclusive_or(
        &self,
        other: &Self,
    ) -> Self {
        self.difference(other)
            .union(&other.difference(self))
    }

    pub(crate) fn boolean_not(&self) -> Self {
        match self {
            Self::NoConstraint => Self::Empty,
            Self::Empty => Self::NoConstraint,
            Self::Active { vars, rects } => {
                if rects.dims.is_empty() {
                    // vars-only: complement in
                    // var-space is conservative.
                    return Self::NoConstraint;
                }
                let negated = rects.negate();
                if negated.is_empty() {
                    Self::Empty
                } else {
                    Self::Active {
                        vars: vars.clone(),
                        rects: negated,
                    }
                }
            }
        }
    }
}

/// Combine two `RectangleSet`s, treating empty-dim sentinels as identity.
fn combine_rects(
    a: &RectangleSet,
    b: &RectangleSet,
    f: impl FnOnce(
        &RectangleSet,
        &RectangleSet,
    ) -> RectangleSet,
) -> RectangleSet {
    match (a.dims.is_empty(), b.dims.is_empty()) {
        (true, true) => a.clone(),
        (true, false) => b.clone(),
        (false, true) => a.clone(),
        (false, false) => f(a, b),
    }
}

impl PartialEq for ExprPlan {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Self::NoConstraint, Self::NoConstraint)
                | (Self::Empty, Self::Empty)
        )
    }
}
impl Eq for ExprPlan {}
