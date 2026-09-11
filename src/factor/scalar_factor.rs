//! Scalar factor representation.
//!
//! A [`ScalarFactor`] represents a constant factor with an empty scope and a
//! single log-space value.

use crate::semiring::{LogMaxProduct, LogSumProduct};
use crate::variable::VariableId;

use super::{
    DenseFactor, Factor, FactorDistance, FactorKind, FactorNormalize, FactorOps, UnaryFactor,
};

/// A constant factor with an empty scope.
///
/// A scalar factor contains no variables and stores a single value in
/// log-space.
#[derive(Clone, Debug)]
pub struct ScalarFactor {
    value: f64,
}

impl ScalarFactor {
    /// Creates a scalar factor from a log-space value.
    pub fn new(value: f64) -> Self {
        Self { value }
    }

    /// Creates a scalar factor from a linear-space value.
    pub fn from_linear(linear: f64) -> Self {
        ScalarFactor::new(linear.ln())
    }

    /// Returns the factor value in log-space.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Combines this scalar factor with another factor in log-space.
    fn combine_factor(self, other: FactorKind) -> FactorKind {
        match other {
            FactorKind::Scalar(other) => FactorKind::Scalar(Self::new(self.value + other.value)),

            FactorKind::Unary(unary) => {
                let data = unary
                    .data()
                    .iter()
                    .map(|value| value + self.value)
                    .collect();

                FactorKind::Unary(UnaryFactor::new(unary.var(), data))
            }

            FactorKind::Dense(dense) => {
                let (scope, data) = dense.into_parts();
                let data = data.mapv(|value| value + self.value);

                FactorKind::Dense(DenseFactor::new(scope, data))
            }
        }
    }
}

impl Factor for ScalarFactor {
    fn scope(&self) -> &[VariableId] {
        &[]
    }
}

impl FactorDistance for ScalarFactor {
    fn distance(&self, other: &Self) -> f64 {
        (self.value() - other.value()).abs()
    }
}

impl FactorOps<LogSumProduct> for ScalarFactor {
    fn reduce(self, _vars: &[VariableId]) -> FactorKind {
        FactorKind::Scalar(self)
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        self.combine_factor(other)
    }
}

impl FactorOps<LogMaxProduct> for ScalarFactor {
    fn reduce(self, _vars: &[VariableId]) -> FactorKind {
        FactorKind::Scalar(self)
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        self.combine_factor(other)
    }
}

impl FactorNormalize<LogSumProduct> for ScalarFactor {
    fn normalize(self) -> Self {
        ScalarFactor::new(0.0)
    }
}

impl FactorNormalize<LogMaxProduct> for ScalarFactor {
    fn normalize(self) -> Self {
        ScalarFactor::new(0.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn v(id: usize) -> VariableId {
        VariableId::new(id)
    }

    #[test]
    fn test_from_linear() {
        let factor = ScalarFactor::from_linear(4.0);

        assert!((factor.value() - 4.0_f64.ln()).abs() < 1e-12);
        assert!(factor.scope().is_empty());
    }

    #[test]
    fn test_distance() {
        let lhs = ScalarFactor::new(2.0);
        let rhs = ScalarFactor::new(5.0);

        assert!((lhs.distance(&rhs) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_reduce() {
        let factor = ScalarFactor::new(2.0);

        let result = <ScalarFactor as FactorOps<LogSumProduct>>::reduce(factor, &[v(0)]);

        match result {
            FactorKind::Scalar(scalar) => {
                assert_eq!(scalar.value(), 2.0);
            }
            _ => panic!("Expected scalar factor"),
        }
    }

    #[test]
    fn test_combine_scalar_x_scalar() {
        let lhs = ScalarFactor::new(2.0);
        let rhs = ScalarFactor::new(3.0);

        let result =
            <ScalarFactor as FactorOps<LogSumProduct>>::combine(lhs, FactorKind::Scalar(rhs));

        match result {
            FactorKind::Scalar(scalar) => {
                assert_eq!(scalar.value(), 5.0);
            }
            _ => panic!("Expected scalar factor"),
        }
    }

    #[test]
    fn test_combine_scalar_x_unary() {
        let scalar = ScalarFactor::new(10.0);
        let unary = UnaryFactor::new(v(0), vec![1.0, 2.0]);

        let result =
            <ScalarFactor as FactorOps<LogSumProduct>>::combine(scalar, FactorKind::Unary(unary));

        match result {
            FactorKind::Unary(unary) => {
                assert_eq!(unary.scope(), &[v(0)]);
                assert_eq!(unary.data(), &[11.0, 12.0]);
            }
            _ => panic!("Expected unary factor"),
        }
    }

    #[test]
    fn test_combine_scalar_x_dense() {
        let scalar = ScalarFactor::new(10.0);

        let dense = DenseFactor::new(vec![v(0), v(1)], array![[1.0, 2.0], [3.0, 4.0]].into_dyn());

        let result =
            <ScalarFactor as FactorOps<LogSumProduct>>::combine(scalar, FactorKind::Dense(dense));

        match result {
            FactorKind::Dense(dense) => {
                let expected = array![[11.0, 12.0], [13.0, 14.0]].into_dyn();

                assert_eq!(dense.scope(), &[v(0), v(1)]);
                assert_eq!(dense.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }

    #[test]
    fn test_normalize() {
        let factor = ScalarFactor::new(4.0);

        let normalized = <ScalarFactor as FactorNormalize<LogSumProduct>>::normalize(factor);

        assert_eq!(normalized.value(), 0.0);
        assert!(normalized.scope().is_empty());
    }
}
