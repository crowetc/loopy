//! Unary factor: a factor over a single discrete variable (log-space).
//!
//! A `UnaryFactor` stores:
//! - `var`: the variable ID
//! - `data`: log-potentials for each domain element
use ndarray::{ArrayD, IxDyn};

use super::{DiscreteFactor, Factor, FactorKind, DenseFactor, ScalarFactor};
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
            FactorKind::Scalar(ScalarFactor::new(total))
        } else {
            FactorKind::Unary(self)
        }
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        match other {
            FactorKind::Unary(u2) => {
                let v1 = self.scope()[0];
                let v2 = u2.scope()[0];

                // ------------------------------------------------------------
                // Case A: same variable → unary result
                // ------------------------------------------------------------
                if v1 == v2 {
                    let data = self.data.iter()
                        .zip(u2.data().iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>();

                    return FactorKind::Unary(UnaryFactor::new(v1, data));
                }

                // ------------------------------------------------------------
                // Case B: different variables → 2-variable dense result
                // ------------------------------------------------------------
                let x_size = self.data.len();
                let y_size = u2.data().len();

                let new_scope = vec![v1, v2];
                let out_shape = vec![x_size, y_size];

                let mut out = ArrayD::<f64>::zeros(IxDyn(&out_shape));

                let mut out_iter = out.iter_mut();

                for x in 0..x_size {
                    let u1_val = self.data[x];
                    for y in 0..y_size {
                        let u2_val = u2.data()[y];
                        *out_iter.next().unwrap() = u1_val + u2_val;
                    }
                }

                FactorKind::Dense(DenseFactor::new(new_scope, out))
            }

            FactorKind::Scalar(s) => {
                let data = self.data.iter()
                    .map(|x| x + s.value())
                    .collect::<Vec<_>>();

                FactorKind::Unary(UnaryFactor::new(self.scope()[0], data))
            }

            FactorKind::Dense(d) => {
                // let DenseFactor handle it
                d.combine(FactorKind::Unary(self))
            }
        }
    }
}

/// Implement `DiscreteFactor` trait.
impl DiscreteFactor for UnaryFactor {
    fn card(&self) -> &[usize] {
        &self.card
    }
}
