//! `ExprPlan` — intermediate representation that separates dimension constraints
//! from variable tracking during expression compilation.
//!
//! Resolution to concrete `DatasetSelection` is handled by
//! `resolve_expr_plan_sync`/`resolve_expr_plan_async` in `lazy_materialize`.

use smallvec::SmallVec;

use crate::IStr;
use crate::chunk_plan::indexing::lazy_selection::LazyArraySelection;
use crate::chunk_plan::indexing::selection::SetOperations;

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
        if v.is_empty() {
            Self::Specific(SmallVec::new())
        } else {
            Self::Specific(v.into())
        }
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
                let v: SmallVec<[IStr; 8]> = a
                    .iter()
                    .filter(|v| b.contains(v))
                    .cloned()
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
                let mut v = a.clone();
                for item in b {
                    if !v.contains(item) {
                        v.push(item.clone());
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
                let v: SmallVec<[IStr; 8]> = a
                    .iter()
                    .filter(|v| !b.contains(v))
                    .cloned()
                    .collect();
                Self::Specific(v)
            }
        }
    }
}

/// Result of compiling a Polars expression for chunk planning.
///
/// Separates dimension constraints from variable references. The expensive
/// `GroupedSelection` construction is deferred to `into_lazy_dataset_selection`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprPlan {
    /// No selection made (identity for intersect, absorbing for union).
    NoConstraint,
    /// Everything excluded.
    Empty,
    /// Active constraints on dimensions + which variables are needed.
    Active {
        vars: VarSet,
        constraints: Box<LazyArraySelection>,
    },
}

impl ExprPlan {
    pub(crate) fn unconstrained_vars(
        vars: VarSet,
    ) -> Self {
        if vars.is_empty() {
            Self::NoConstraint
        } else {
            Self::Active {
                vars,
                constraints: Box::new(
                    LazyArraySelection::all(),
                ),
            }
        }
    }

    pub(crate) fn constrained(
        vars: VarSet,
        constraints: LazyArraySelection,
    ) -> Self {
        if constraints.is_empty() {
            Self::Empty
        } else {
            Self::Active {
                vars,
                constraints: Box::new(
                    constraints,
                ),
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Replace vars with explicit refs while keeping constraints.
    /// Used by ternary/filter to combine explicit column refs with
    /// constraints from a predicate (whose own vars may be `All`).
    pub(crate) fn with_vars(
        self,
        vars: VarSet,
    ) -> Self {
        match self {
            Self::NoConstraint => {
                Self::unconstrained_vars(vars)
            }
            Self::Empty => Self::Empty,
            Self::Active {
                constraints, ..
            } => {
                if vars.is_empty() {
                    Self::NoConstraint
                } else {
                    Self::Active {
                        vars,
                        constraints,
                    }
                }
            }
        }
    }

    /// Add variables without changing constraints.
    /// Used for `Expr::Over` where partition vars need fetching
    /// but shouldn't weaken the function's constraints.
    pub(crate) fn add_vars(
        &self,
        extra: VarSet,
    ) -> Self {
        match self {
            Self::NoConstraint => {
                Self::NoConstraint
            }
            Self::Empty => Self::Empty,
            Self::Active {
                vars,
                constraints,
            } => Self::Active {
                vars: vars.union(&extra),
                constraints: constraints.clone(),
            },
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
                    constraints: ca,
                },
                Self::Active {
                    vars: vb,
                    constraints: cb,
                },
            ) => {
                let vars = va.intersect(vb);
                let constraints =
                    ca.intersect(cb);
                if constraints.is_empty() {
                    Self::Empty
                } else {
                    Self::Active {
                        vars,
                        constraints: Box::new(
                            constraints,
                        ),
                    }
                }
            }
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
                    constraints: ca,
                },
                Self::Active {
                    vars: vb,
                    constraints: cb,
                },
            ) => {
                let vars = va.union(vb);
                let constraints = ca.union(cb);
                Self::Active {
                    vars,
                    constraints: Box::new(
                        constraints,
                    ),
                }
            }
        }
    }

    pub(crate) fn difference(
        &self,
        other: &Self,
    ) -> Self {
        match (self, other) {
            (Self::Empty, _) => Self::Empty,
            (x, Self::Empty) => x.clone(),
            (_, Self::NoConstraint) => {
                Self::Empty
            }
            (Self::NoConstraint, _) => {
                Self::NoConstraint
            }
            (
                Self::Active {
                    vars: va,
                    constraints: ca,
                },
                Self::Active {
                    vars: vb,
                    constraints: cb,
                },
            ) => {
                if ca == cb {
                    // Same constraints: difference is purely about variables
                    let v_diff =
                        va.difference(vb);
                    if v_diff.is_empty() {
                        Self::Empty
                    } else {
                        Self::Active {
                            vars: v_diff,
                            constraints: ca
                                .clone(),
                        }
                    }
                } else {
                    let c_diff =
                        ca.difference(cb);
                    if c_diff.is_empty() {
                        Self::Empty
                    } else {
                        Self::Active {
                            vars: va.clone(),
                            constraints: Box::new(
                                c_diff,
                            ),
                        }
                    }
                }
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
            Self::NoConstraint => Self::NoConstraint,
            Self::Empty => Self::Empty,
            Self::Active { vars, constraints } => Self::Active { vars: vars.clone(), constraints: Box::new(LazyArraySelection::BooleanNot(Box::new(constraints.as_ref().to_owned()))),
            },
        }
    }
}
