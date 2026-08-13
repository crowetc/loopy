//! Unary factor: a factor over a single discrete variable (log-space).

use super::Factor;
use super::DiscreteFactor;
use super::log_utils::{lse_update, lse_finalize};

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
    fn marginalize(&self, vars: &[usize]) -> Self {
        if vars.contains(&self.var) {
            // Numerically-stable log-sum-exp over all entries
            let mut cur_max = f64::NEG_INFINITY;
            let mut cur_sum = 0.0;
            for &x in &self.data {
                lse_update(x, &mut cur_max, &mut cur_sum);
            }
            let total = lse_finalize(cur_max, cur_sum);
            UnaryFactor::new(self.var, vec![total])
        } else {
            self.clone()
        }
    }
}

impl DiscreteFactor for UnaryFactor {
    fn card(&self) -> &[usize] {
        &self.card
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope() {
        // use from_linear convenience to build factor
        let f = UnaryFactor::from_linear(7, vec![0.1, 0.9]);
        assert_eq!(f.scope(), &[7]);
    }

    #[test]
    fn test_card() {
        let f = UnaryFactor::from_linear(3, vec![1.0, 2.0, 3.0]);
        assert_eq!(f.card(), &[3]);
    }

    #[test]
    fn test_marginalize_kept() {
        let f = UnaryFactor::from_linear(2, vec![1.0, 2.0, 3.0]);
        let g = f.marginalize(&[]);
        // g.data() is log-space; compare exponentiated values with tolerance
        let got: Vec<f64> = g.data().iter().map(|x| x.exp()).collect();
        let expected = [1.0_f64, 2.0, 3.0];
        let tol = 1e-12;
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() <= tol,
                "values differ: got {:?}, expected {:?}",
                got,
                expected
            );
        }
    }

    #[test]
    fn test_marginalize_eliminated() {
        let f = UnaryFactor::from_linear(2, vec![1.0, 2.0, 3.0]);
        let g = f.marginalize(&[2]);
        // sum = 6.0 -> log(6.0)
        let expected = (6.0_f64).ln();
        assert_eq!(g.data(), &[expected]);
        assert_eq!(g.card(), &[1]);
    }
}
