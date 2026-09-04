//! Factor representations and operations for factor graphs.
//!
//! This module provides the core abstractions used to represent factors,
//! factor graphs, and operations on factors.
//!
//! A [`Factor`] represents a function over a set of variables identified by
//! [`VariableId`]. Concrete factor types provide different representations
//! and capabilities. For example, [`DenseFactor`] represents a discrete factor
//! using a dense multidimensional array, while [`UnaryFactor`] represents a
//! discrete factor over a single discrete variable.
//!
//! [`FactorOps`] defines algebraic operations on factors, parameterized by a
//! [`Semiring`]. This allows for multiple forms of the reduction and combination
//! operations to be defined on a factor to support different inference problems.
//!
//! [`DiscreteFactor`] identifies factors whose variables are discrete and have
//! finite cardinalities.
//!
//! The module also contains [`FactorGraph`], which represents the structure
//! connecting variables and factors.

pub mod dense_factor;
pub mod factor_graph;
pub mod factor_kind;
pub mod scalar_factor;
pub mod unary_factor;
pub mod variable;

mod log_utils;
mod utils;

pub use dense_factor::DenseFactor;
pub use factor_graph::{FactorGraph, FactorId};
pub use factor_kind::FactorKind;
pub use scalar_factor::ScalarFactor;
pub use unary_factor::UnaryFactor;
pub use variable::{Variable, VariableId};

pub use crate::semiring::{LogMaxProduct, LogSumProduct, Semiring};

/// A factor in the sense used in probabilistic graphical models.
///
/// A factor represents a function over the variables in its scope. The
/// concrete representation and operations supported by a factor depend on
/// its type.
///
/// Factors are identified by the variables returned by [`Self::scope`].
pub trait Factor {
    /// Variables in the factor's scope.
    fn scope(&self) -> &[VariableId];

    /// The number of variables in the factor.
    fn ndim(&self) -> usize {
        self.scope().len()
    }
}

/// Operations supported by a factor under a particular semiring.
///
/// The semiring determines how factors are combined and marginalized.
/// For example, [`LogSumProduct`] performs sum-product operations in
/// log-space, while [`LogMaxProduct`] performs max-product operations.
///
/// Implementations consume `self` because factor operations generally
/// construct a new factor rather than modifying the existing one.
pub trait FactorOps<S: Semiring>: Factor {
    /// Reduce the factor by eliminating the given variables.
    fn reduce(self, vars: &[VariableId]) -> FactorKind;

    /// Combine this factor with another factor.
    fn combine(self, other: FactorKind) -> FactorKind;
}

/// A factor over discrete variables with finite cardinalities.
///
/// The cardinality at position `i` corresponds to the variable at position
/// `i` in [`Factor::scope`].
pub trait DiscreteFactor: Factor {
    /// Cardinalities of all variables in scope.
    fn card(&self) -> &[usize];
}
