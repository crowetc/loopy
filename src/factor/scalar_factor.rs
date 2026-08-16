//! Scalar factor: a factor with empty scope and a single log-value.
//!
//! A `ScalarFactor` stores:
//! - `value`: a single log-potential

use super::{Factor, FactorKind};

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
    fn scope(&self) -> &[usize] {
        &[]  // empty scope
    }

    /// Marginalizing a scalar factor is a no-op.
    fn marginalize(self, _vars: &[usize]) -> FactorKind {
        FactorKind::Scalar(self)
    }
}
