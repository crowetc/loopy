//! Unary factor: a factor over a single discrete variable (log-space).

use super::FactorKind;
use super::Factor;
use super::DiscreteFactor;
use super::log_utils::lse_two_pass;

/// A unary factor over a single variable (log-space).
///
/// Stores:
/// - `var`: the variable ID
/// - `data`: the factor values for each domain element (log-potentials)
#[derive(Clone, Debug)]
pub struct UnaryFactor {
    var: usize,
    data: Vec<f64>,      // log-potentials
    card: [usize; 1],    // stable slice
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

    /// Convenience constructor: accept linear-space values and convert to log-space.
    pub fn from_linear(var: usize, linear: Vec<f64>) -> Self {
        assert!(!linear.is_empty());
        let data: Vec<f64> = linear.into_iter().map(|x| x.ln()).collect();
        let card = [data.len()];
        Self { var, data, card }
    }

    /// Access the underlying log-space data.
    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

impl Factor for UnaryFactor {
    fn scope(&self) -> &[usize] {
        std::slice::from_ref(&self.var)
    }

    /// Marginalize variables `vars`. If this unary variable is eliminated,
    /// return a UnaryFactor with a single log-sum element (log-space).
    fn marginalize(self, vars: &[usize]) -> FactorKind {
        if vars.contains(&self.var) {
            // Reduce to a single log-value
            let total = lse_two_pass(self.data());
            FactorKind::Unary(UnaryFactor::new(self.var, vec![total]))
        } else {
            FactorKind::Unary(self)
        }
    }

    // /// Combine (multiply / add in log-space) this unary factor with another factor.
    // /// Specialized fast paths:
    // /// - Unary × Unary (same var): elementwise add -> Unary
    // /// - Unary × Unary (different vars): outer-sum -> Dense
    // /// - Unary × Dense: broadcast-add unary into dense (in-place when possible)
    // fn combine(self, other: FactorKind) -> FactorKind {
    //     match other {
    //         FactorKind::Unary(u2) => {
    //             if self.var == u2.var {
    //                 // same variable: elementwise add
    //                 assert_eq!(self.data.len(), u2.data.len());
    //                 let data = self
    //                     .data
    //                     .into_iter()
    //                     .zip(u2.data.into_iter())
    //                     .map(|(a, b)| a + b)
    //                     .collect::<Vec<_>>();
    //                 FactorKind::Unary(UnaryFactor::new(self.var, data))
    //             } else {
    //                 // different variables: outer-sum -> Dense
    //                 let scope = vec![self.var, u2.var];
    //                 let card1 = self.data.len();
    //                 let card2 = u2.data.len();
    //                 let mut out_vec = Vec::with_capacity(card1 * card2);
    //                 for &a in self.data.iter() {
    //                     for &b in u2.data.iter() {
    //                         out_vec.push(a + b);
    //                     }
    //                 }
    //                 let out = ArrayD::from_shape_vec(IxDyn(&[card1, card2]), out_vec)
    //                     .expect("shape and data length must match");
    //                 FactorKind::Dense(DenseFactor::new(scope, out))
    //             }
    //         }

    //         FactorKind::Dense(d) => {
    //             // Delegate to utils helper which will add unary into dense in-place
    //             // or produce a new Dense factor if the unary var is not present.
    //             utils::add_unary_into_dense_inplace(d, self)
    //         }
    //     }
    // }
}

impl DiscreteFactor for UnaryFactor {
    fn card(&self) -> &[usize] {
        &self.card
    }
}
