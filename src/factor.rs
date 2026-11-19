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
/// Both operations consume the left-hand factor and return a [`FactorKind`]
/// because the resulting representation may differ from the input. In
/// particular, reduction may lower the dimensionality of a factor, while
/// combination may increase it.
pub trait FactorOps<S: Semiring>: Factor + Clone {
    /// Eliminates the specified variables from the factor.
    ///
    /// Variables not present in the factor's scope have no effect.
    fn reduce(self, vars: &[VariableId]) -> FactorKind;

    /// Combines this factor with another factor under semiring `S`.
    ///
    /// The resulting factor has scope equal to the union of the two input
    /// scopes.
    fn combine(self, other: &FactorKind) -> FactorKind;

    /// Eliminates variables, writing the result into `output`.
    ///
    /// Existing output storage may be reused when compatible.
    /// The previous value of `output` does not affect the result.
    fn reduce_into(&self, vars: &[VariableId], output: &mut FactorKind) {
        *output = self.clone().reduce(vars);
    }

    /// Combines two factors, writing the result into `output`.
    ///
    /// Existing output storage may be reused when compatible.
    /// The previous value of `output` does not affect the result.
    fn combine_into(&self, other: &FactorKind, output: &mut FactorKind) {
        *output = self.clone().combine(other);
    }
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::{LogMaxProduct, LogSumProduct};

    struct TestFactor {
        scope: Vec<VariableId>,
    }

    impl Factor for TestFactor {
        fn scope(&self) -> &[VariableId] {
            &self.scope
        }
    }

    #[test]
    fn factor_id_returns_index() {
        let id = FactorId::new(3);

        assert_eq!(id.index(), 3);
    }

    #[test]
    fn ndim_returns_scope_length() {
        let factor = TestFactor {
            scope: vec![VariableId::new(0), VariableId::new(1), VariableId::new(2)],
        };

        assert_eq!(factor.ndim(), 3);
    }

    #[test]
    fn ndim_is_zero_for_empty_scope() {
        let factor = TestFactor { scope: vec![] };

        assert_eq!(factor.ndim(), 0);
    }

    fn assert_same_factor(actual: &FactorKind, expected: &FactorKind) {
        assert_eq!(actual.scope(), expected.scope());

        match (actual, expected) {
            (FactorKind::Unary(a), FactorKind::Unary(b)) => {
                assert_eq!(a.data(), b.data());
            }
            (FactorKind::Scalar(a), FactorKind::Scalar(b)) => {
                assert_eq!(a.value(), b.value());
            }
            (FactorKind::Dense(a), FactorKind::Dense(b)) => {
                assert_eq!(a.card(), b.card());
                assert_eq!(a.data(), b.data());
            }
            _ => panic!("factor representations differ"),
        }
    }

    fn check_combine_into<S: Semiring>()
    where
        FactorKind: FactorOps<S>,
    {
        let x = VariableId::new(0);
        let y = VariableId::new(1);

        let lhs = FactorKind::Unary(UnaryFactor::new(x, vec![0.0, -1.0]));
        let rhs = FactorKind::Unary(UnaryFactor::new(x, vec![-2.0, -3.0]));

        let expected = <FactorKind as FactorOps<S>>::combine(lhs.clone(), &rhs);

        // Start with an unrelated destination.
        let mut output = FactorKind::Scalar(ScalarFactor::new(42.0));

        <FactorKind as FactorOps<S>>::combine_into(&lhs, &rhs, &mut output);

        assert_same_factor(&output, &expected);

        // Reuse the destination for a different result scope.
        let other = FactorKind::Unary(UnaryFactor::new(y, vec![-0.5, -1.5]));

        let expected = <FactorKind as FactorOps<S>>::combine(lhs.clone(), &other);

        <FactorKind as FactorOps<S>>::combine_into(&lhs, &other, &mut output);

        assert_same_factor(&output, &expected);

        // Inputs must remain unchanged.
        match &lhs {
            FactorKind::Unary(f) => assert_eq!(f.data(), &[0.0, -1.0]),
            _ => panic!("expected unary factor"),
        }
        match &rhs {
            FactorKind::Unary(f) => assert_eq!(f.data(), &[-2.0, -3.0]),
            _ => panic!("expected unary factor"),
        }
    }

    fn check_reduce_into<S: Semiring>()
    where
        FactorKind: FactorOps<S>,
    {
        let x = VariableId::new(0);
        let lhs = FactorKind::Unary(UnaryFactor::new(x, vec![0.0, -1.0]));

        let mut output = FactorKind::Scalar(ScalarFactor::new(42.0));

        // No elimination: result remains unary.
        let expected = <FactorKind as FactorOps<S>>::reduce(lhs.clone(), &[]);

        <FactorKind as FactorOps<S>>::reduce_into(&lhs, &[], &mut output);

        assert_same_factor(&output, &expected);

        // Eliminate x: result becomes scalar.
        let expected = <FactorKind as FactorOps<S>>::reduce(lhs.clone(), &[x]);

        <FactorKind as FactorOps<S>>::reduce_into(&lhs, &[x], &mut output);

        assert_same_factor(&output, &expected);

        // Repeat to ensure the previous output does not matter.
        <FactorKind as FactorOps<S>>::reduce_into(&lhs, &[x], &mut output);

        assert_same_factor(&output, &expected);
    }

    #[test]
    fn destination_operations_log_sum_product() {
        check_combine_into::<LogSumProduct>();
        check_reduce_into::<LogSumProduct>();
    }

    #[test]
    fn destination_operations_log_max_product() {
        check_combine_into::<LogMaxProduct>();
        check_reduce_into::<LogMaxProduct>();
    }
}
