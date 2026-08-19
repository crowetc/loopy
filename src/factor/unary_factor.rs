//! Unary factor: a factor over a single discrete variable (log-space).
//!
//! A `UnaryFactor` stores:
//! - `var`: the variable ID
//! - `data`: log-potentials for each domain element
use ndarray::{ArrayD, IxDyn};

use super::log_utils::lse_two_pass;
use super::{DenseFactor, DiscreteFactor, Factor, FactorKind, ScalarFactor};

/// A unary factor over a single discrete variable (log-space).
///
/// The underlying values are stored in log-space.
#[derive(Clone, Debug)]
pub struct UnaryFactor {
    var: usize,
    data: Vec<f64>,
    card: [usize; 1],
}

impl UnaryFactor {
    /// Create a unary factor for a variable with the given **log-space** data.
    ///
    /// # Panics
    /// Panics if `data` is empty.
    pub fn new(var: usize, data: Vec<f64>) -> Self {
        assert!(!data.is_empty());
        let card = [data.len()];
        Self { var, data, card }
    }

    /// Construct a unary factor from **linear-space** values.
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

    /// Combine two unary factors.
    ///
    /// If the factors share a variable, the result remains unary.
    ///
    /// If they are over different variables, the result is a dense factor
    /// over the sorted union of the two variables.
    fn combine_unary(self, other: UnaryFactor) -> FactorKind {
        let var1 = self.var;
        let var2 = other.var;

        // ------------------------------------------------------------
        // Same variable -> unary result
        // ------------------------------------------------------------

        if var1 == var2 {
            assert_eq!(
                self.card()[0],
                other.card()[0],
                "cardinality mismatch for variable {}",
                var1
            );

            let data = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            return FactorKind::Unary(UnaryFactor::new(var1, data));
        }

        // ------------------------------------------------------------
        // Different variables → dense result
        //
        // The output scope is always sorted.
        // ------------------------------------------------------------

        let (scope, first, second) = if var1 < var2 {
            (
                vec![var1, var2],
                self.data.as_slice(),
                other.data.as_slice(),
            )
        } else {
            (
                vec![var2, var1],
                other.data.as_slice(),
                self.data.as_slice(),
            )
        };

        let mut output = ArrayD::<f64>::zeros(IxDyn(&[first.len(), second.len()]));

        let mut output_iter = output.iter_mut();

        for &a in first {
            for &b in second {
                *output_iter
                    .next()
                    .expect("output iterator length must match shape") = a + b;
            }
        }
        FactorKind::Dense(DenseFactor::new(scope, output))
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
            FactorKind::Dense(d) => d.combine(FactorKind::Unary(self)),
            FactorKind::Unary(other) => self.combine_unary(other),
            FactorKind::Scalar(s) => {
                let data = self.data.iter().map(|x| x + s.value()).collect::<Vec<_>>();

                FactorKind::Unary(UnaryFactor::new(self.var, data))
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn ln_array(v: &[f64]) -> Vec<f64> {
        v.iter().map(|x| x.ln()).collect()
    }

    //
    // Construction Tests
    //

    #[test]
    fn test_new() {
        let f = UnaryFactor::new(3, vec![1.0, 2.0, 3.0]);

        assert_eq!(f.var(), 3);
        assert_eq!(f.scope(), &[3]);
        assert_eq!(f.card(), &[3]);
        assert_eq!(f.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_linear() {
        let f = UnaryFactor::from_linear(2, vec![1.0, 10.0, 100.0]);

        let expected = ln_array(&[1.0, 10.0, 100.0]);

        assert_eq!(f.var(), 2);
        assert_eq!(f.scope(), &[2]);
        assert_eq!(f.card(), &[3]);

        for (actual, expected) in f.data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-12);
        }
    }

    #[test]
    fn test_into_data() {
        let f = UnaryFactor::new(7, vec![1.0, 2.0, 3.0]);

        assert_eq!(f.into_data(), vec![1.0, 2.0, 3.0]);
    }

    //
    // Marginalize Tests
    //

    #[test]
    fn test_marginalize_in_scope() {
        let f = UnaryFactor::new(0, ln_array(&[1.0, 2.0]));

        let result = f.marginalize(&[0]);

        match result {
            FactorKind::Scalar(s) => {
                let expected = (1.0_f64 + 2.0).ln();
                assert!((s.value() - expected).abs() <= 1e-12);
            }
            _ => panic!("Expected scalar factor"),
        }
    }

    #[test]
    fn test_marginalize_out_of_scope() {
        let f = UnaryFactor::new(0, vec![1.0, 2.0]);

        let result = f.marginalize(&[1]);

        match result {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[0]);
                assert_eq!(u.data(), &[1.0, 2.0]);
            }
            _ => panic!("Expected unary factor"),
        }
    }

    //
    // Combine Tests
    //

    #[test]
    fn test_combine_unary_x_unary_intersect() {
        let f = UnaryFactor::new(0, vec![1.0, 2.0]);
        let g = UnaryFactor::new(0, vec![10.0, 20.0]);

        let result = f.combine(FactorKind::Unary(g));

        match result {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[0]);
                assert_eq!(u.data(), &[11.0, 22.0]);
            }
            _ => panic!("Expected unary factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_unary_intersect_three_states() {
        let f = UnaryFactor::new(0, vec![1.0, 2.0, 3.0]);
        let g = UnaryFactor::new(0, vec![10.0, 20.0, 30.0]);

        let result = f.combine(FactorKind::Unary(g));

        match result {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[0]);
                assert_eq!(u.data(), &[11.0, 22.0, 33.0]);
            }
            _ => panic!("Expected unary factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_unary_disjoint() {
        let f = UnaryFactor::new(0, vec![1.0, 2.0]);
        let g = UnaryFactor::new(1, vec![10.0, 20.0]);

        let result = f.combine(FactorKind::Unary(g));

        match result {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[0, 1]);

                let expected = array![[11.0, 21.0], [12.0, 22.0],].into_dyn();

                assert_eq!(d.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_unary_unsorted_scope() {
        let f = UnaryFactor::new(1, vec![1.0, 2.0]);
        let g = UnaryFactor::new(0, vec![10.0, 20.0]);

        let result = f.combine(FactorKind::Unary(g));

        match result {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[0, 1]);

                let expected = array![[11.0, 12.0], [21.0, 22.0],].into_dyn();

                assert_eq!(d.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_scalar() {
        let f = UnaryFactor::new(0, vec![1.0, 2.0]);
        let scalar = ScalarFactor::new(10.0);

        let result = f.combine(FactorKind::Scalar(scalar));

        match result {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[0]);
                assert_eq!(u.data(), &[11.0, 12.0]);
            }
            _ => panic!("Expected unary factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_dense_intersect() {
        let f = UnaryFactor::new(1, vec![10.0, 20.0]);

        let dense = DenseFactor::new(vec![0, 1], array![[1.0, 2.0], [3.0, 4.0],].into_dyn());

        let result = f.combine(FactorKind::Dense(dense));

        match result {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[0, 1]);

                let expected = array![[11.0, 22.0], [13.0, 24.0],].into_dyn();

                assert_eq!(d.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_dense_disjoint() {
        let f = UnaryFactor::new(2, vec![10.0, 20.0]);

        let dense = DenseFactor::new(vec![0, 1], array![[1.0, 2.0], [3.0, 4.0],].into_dyn());

        let result = f.combine(FactorKind::Dense(dense));
        match result {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[0, 1, 2]);

                let expected =
                    array![[[11.0, 12.0], [13.0, 14.0]], [[21.0, 22.0], [23.0, 24.0]],].into_dyn();

                assert_eq!(d.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_dense_unsorted_scope() {
        let f = UnaryFactor::new(1, vec![10.0, 20.0]);

        let dense = DenseFactor::new(vec![1, 0], array![[1.0, 2.0], [3.0, 4.0],].into_dyn());

        let result = f.combine(FactorKind::Dense(dense));

        match result {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[0, 1]);

                let expected = array![[11.0, 23.0], [12.0, 24.0],].into_dyn();

                assert_eq!(d.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }
}
