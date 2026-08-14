//! Unary factor: a factor over a single discrete variable (log-space).
//!
//! A `UnaryFactor` stores:
//! - `var`: the variable ID
//! - `data`: log-potentials for each domain element

use super::{DiscreteFactor, Factor, FactorKind};
use super::log_utils::lse_two_pass;

/// A unary factor over a single variable (log-space).
///
/// Represents a function `f(var)` where `var` ranges over a discrete domain.
/// Internally stores:
/// - `var`: the variable ID
/// - `data`: log-potentials for each domain element
/// - `card`: a stable one-element slice for trait compatibility
#[derive(Clone, Debug)]
pub struct UnaryFactor {
    var: usize,
    data: Vec<f64>,
    card: [usize; 1],
}

impl UnaryFactor {
    /// Create a unary factor for a variable with the given **log-space** data.
    ///
    /// Panics if `data` is empty.
    pub fn new(var: usize, data: Vec<f64>) -> Self {
        assert!(!data.is_empty());
        let card = [data.len()];
        Self { var, data, card }
    }

    /// Construct a unary factor from **linear-space** values.
    ///
    /// Converts each value to log-space via `ln`.
    ///
    /// # Panics
    /// Panics if `linear` is empty.
    pub fn from_linear(var: usize, linear: Vec<f64>) -> Self {
        assert!(!linear.is_empty());
        let data: Vec<f64> = linear.into_iter().map(|x| x.ln()).collect();
        let card = [data.len()];
        Self { var, data, card }
    }

    /// Borrow the underlying log-space data.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Return the variable ID.
    pub fn var(&self) -> usize {
        self.var
    }

    /// Consume and return the owned log-space vector.
    pub fn into_data(self) -> Vec<f64> {
        self.data
    }
}

/// Implement `Factor` trait
impl Factor for UnaryFactor {
    fn scope(&self) -> &[usize] {
        std::slice::from_ref(&self.var)
    }

    /// Marginalize variables `vars`.
    ///
    /// - If this unary variable is eliminated, return a unary factor with a
    ///   single log-sum-exp value.
    /// - Otherwise, return the factor unchanged.
    fn marginalize(self, vars: &[usize]) -> FactorKind {
        if vars.contains(&self.var) {
            let total = lse_two_pass(self.data());
            FactorKind::Unary(UnaryFactor::new(self.var, vec![total]))
        } else {
            FactorKind::Unary(self)
        }
    }
}

/// Implement `DiscreteFactor` trait.
impl DiscreteFactor for UnaryFactor {
    fn card(&self) -> &[usize] {
        &self.card
    }
}
