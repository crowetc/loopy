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
pub mod factor_kind;
pub mod scalar_factor;
pub mod unary_factor;

mod log_utils;
mod utils;

pub use dense_factor::DenseFactor;
pub use factor_kind::FactorKind;
pub use scalar_factor::ScalarFactor;
pub use unary_factor::UnaryFactor;

pub use crate::semiring::{LogMaxProduct, LogSumProduct, Semiring};
use crate::variable::VariableId;

/// An identifier for a factor within a [`FactorGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FactorId(usize);

impl FactorId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the index underlying this identifier.
    pub fn index(self) -> usize {
        self.0
    }
}

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

/// Operations supported by a factor under a particular inference regime.
///
/// The semiring identifies the inference regime under which the operations
/// are performed. Factor implementations provide the concrete behavior for
/// each supported semiring.
///
/// For example, [`LogSumProduct`] represents sum-product inference in
/// log-space, while [`LogMaxProduct`] represents max-product inference in
/// log-space.
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
