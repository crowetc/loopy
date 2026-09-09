//! Scalar factor: a factor with empty scope and a single log-value.
//!
//! A `ScalarFactor` stores:
//! - `value`: a single log-potential

use crate::semiring::{LogMaxProduct, LogSumProduct};
use crate::variable::VariableId;

use super::{DenseFactor, Factor, FactorKind, FactorNormalize, FactorOps, FactorDistance, UnaryFactor};

/// A scalar factor (log-space).
///
/// Represents a constant log-value with **no variables** in scope.
/// Internally stores:
/// - `value`: the single log-potential
#[derive(Clone, Debug)]
pub struct ScalarFactor {
    value: f64,
}

impl ScalarFactor {
    /// Create a scalar factor from a **log-space** value.
    pub fn new(value: f64) -> Self {
        Self { value }
    }

    /// Construct a scalar factor from a **linear-space** value.
    ///
    /// # Panics
    /// Panics if `linear` does not contain exactly one element.
    pub fn from_linear(linear: f64) -> Self {
        ScalarFactor::new(linear.ln())
    }

    /// Access the underlying log-value.
    pub fn value(&self) -> f64 {
        self.value
    }
}

/// Implement `Factor` trait
impl Factor for ScalarFactor {
    /// A scalar factor has empty scope
    fn scope(&self) -> &[VariableId] {
        &[] // empty scope
    }
}

impl FactorDistance for ScalarFactor {
    fn distance(&self, other: &Self) -> f64 {
        (self.value() - other.value()).abs()
    }
}

impl FactorOps<LogSumProduct> for ScalarFactor {
    /// Marginalizing a scalar factor is a no-op.
    fn reduce(self, _vars: &[VariableId]) -> FactorKind {
        FactorKind::Scalar(self)
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        match other {
            FactorKind::Scalar(s2) => {
                FactorKind::Scalar(ScalarFactor::new(self.value() + s2.value()))
            }

            FactorKind::Unary(u) => {
                let data = u.data().iter().map(|x| x + self.value()).collect();
                FactorKind::Unary(UnaryFactor::new(u.scope()[0], data))
            }

            FactorKind::Dense(d) => {
                let data = d.data().mapv(|x| x + self.value());
                FactorKind::Dense(DenseFactor::new(d.scope().to_vec(), data))
            }
        }
    }
}

impl FactorOps<LogMaxProduct> for ScalarFactor {
    fn reduce(self, _vars: &[VariableId]) -> FactorKind {
        FactorKind::Scalar(self)
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        match other {
            FactorKind::Scalar(other) => {
                FactorKind::Scalar(ScalarFactor::new(self.value() + other.value()))
            }

            FactorKind::Unary(unary) => {
                <UnaryFactor as FactorOps<LogMaxProduct>>::combine(unary, FactorKind::Scalar(self))
            }

            FactorKind::Dense(dense) => {
                <DenseFactor as FactorOps<LogMaxProduct>>::combine(dense, FactorKind::Scalar(self))
            }
        }
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
