//! Factor representations and operations.
//!
//! This module defines the core abstractions used to represent factors in a
//! factor graph.
//!
//! A [`Factor`] represents a function over a set of variables identified by
//! [`VariableId`]. Concrete factor types provide different representations for
//! that function. For example, [`DenseFactor`] stores a discrete factor as a
//! dense multidimensional array, while [`UnaryFactor`] represents a discrete
//! factor over a single variable.
//!
//! [`FactorOps`] defines semiring-dependent factor algebra, including
//! combination and reduction.
//!
//! [`FactorNormalize`] defines semiring-dependent normalization for factor
//! representations that support it.
//!
//! [`FactorDistance`] defines a semiring-independent comparison between factors
//! of the same representation. It is primarily used to measure changes between
//! successive messages during iterative inference.
//!
//! [`DiscreteFactor`] identifies factors over finite discrete variables and
//! exposes the cardinality associated with each variable in the factor's scope.
//!
//! [`FactorKind`] provides a common representation for the factor types
//! supported by the library.

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

use crate::semiring::Semiring;
use crate::variable::VariableId;

/// Identifies a factor within a factor graph.
///
/// A `FactorId` is a stable index assigned when a factor is added to the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FactorId(usize);

impl FactorId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the index underlying this identifier.
    pub fn index(self) -> usize {
        self.0
    }
}

/// A factor over a set of variables.
///
/// A factor represents a function whose arguments are the variables in its
/// scope. Concrete factor types determine how that function is represented and
/// which operations are supported.
///
/// The order of variables returned by [`Self::scope`] defines the dimension
/// ordering used by the factor representation.
pub trait Factor {
    /// Variables in the factor's scope.
    fn scope(&self) -> &[VariableId];

    /// The number of variables in the factor.
    fn ndim(&self) -> usize {
        self.scope().len()
    }
}

/// Measures the difference between two factors of the same representation.
///
/// The distance is independent of the inference semiring and can be used to
/// compare successive values of the same logical factor or message.
///
/// Implementations assume that `self` and `other` have compatible scopes and
/// representations. Violating those invariants is considered a programming
/// error.
pub trait FactorDistance: Factor {
    /// Returns the distance between corresponding factors.
    fn distance(&self, other: &Self) -> f64;
}

/// Semiring-dependent operations supported by a factor.
///
/// The type parameter `S` identifies the inference algebra under which the
/// operations are performed. A concrete factor implements this trait for each
/// semiring it supports.
///
/// For example, under [`LogSumProduct`], reduction marginalizes variables using
/// log-sum-exp, while under [`LogMaxProduct`] it eliminates variables using a
/// maximum.
///
/// Both operations consume the factor and return a [`FactorKind`] because the
/// resulting representation may differ from the input. In particular,
/// reduction may lower the dimensionality of a factor and combination may
/// increase it.
pub trait FactorOps<S: Semiring>: Factor {
    /// Eliminates the specified variables from the factor.
    ///
    /// Variables not present in the factor's scope have no effect.
    fn reduce(self, vars: &[VariableId]) -> FactorKind;

    /// Combines this factor with another factor under semiring `S`.
    ///
    /// The resulting factor has scope equal to the union of the two input
    /// scopes.
    fn combine(self, other: FactorKind) -> FactorKind;
}

/// Semiring-dependent normalization supported by a factor.
///
/// Normalization rescales a factor according to the selected semiring while
/// preserving its scope and representation.
///
/// Unlike [`FactorOps`], normalization is an optional capability and need not
/// be implemented by every factor representation that supports factor algebra.
pub trait FactorNormalize<S: Semiring>: Factor {
    /// Returns the normalized factor.
    fn normalize(self) -> Self;
}

/// A factor over finite discrete variables.
///
/// The cardinality at position `i` corresponds to the variable at position `i`
/// in [`Factor::scope`].
pub trait DiscreteFactor: Factor {
    /// Cardinalities of all variables in scope.
    fn card(&self) -> &[usize];
}
