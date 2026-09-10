//! Unary discrete factor representation.
//!
//! A [`UnaryFactor`] represents a factor over a single finite discrete variable
//! using a vector of log-space values.
//!
//! Each element corresponds to one state of the variable, and the vector length
//! defines the variable's cardinality.
use ndarray::{ArrayD, IxDyn};

use crate::semiring::{LogMaxProduct, LogSumProduct};
use crate::variable::VariableId;

use super::log_utils::lse_two_pass;
use super::{
    DenseFactor, DiscreteFactor, Factor, FactorDistance, FactorKind, FactorNormalize, FactorOps,
    ScalarFactor,
};

/// A unary factor over a single finite discrete variable.
///
/// Factor values are stored in log-space, with one value for each state of the
/// variable.
#[derive(Clone, Debug)]
pub struct UnaryFactor {
    var: VariableId,
    data: Vec<f64>,
    card: [usize; 1],
}

impl UnaryFactor {
    /// Creates a unary factor for a variable with the given log-space data.
    ///
    /// # Panics
    /// Panics if `data` is empty.
    pub fn new(var: VariableId, data: Vec<f64>) -> Self {
        assert!(!data.is_empty());
        let card = [data.len()];
        Self { var, data, card }
    }

    /// Constructs a unary factor from linear-space values.
    ///
    /// # Panics
    /// Panics if `linear` is empty.
    pub fn from_linear(var: VariableId, linear: Vec<f64>) -> Self {
        assert!(!linear.is_empty());
        let data: Vec<f64> = linear.into_iter().map(|x| x.ln()).collect();
        let card = [data.len()];
        Self { var, data, card }
    }

    /// Returns the factor values in log-space.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns the variable identifier.
    pub fn var(&self) -> VariableId {
        self.var
    }

    /// Consumes the factor and returns its log-space values.
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
        let lhs_var = self.var;
        let rhs_var = other.var;

        // Shared variable: result remains unary.

        if lhs_var == rhs_var {
            assert_eq!(
                self.card()[0],
                other.card()[0],
                "cardinality mismatch for variable {}",
                lhs_var
            );

            let data = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&lhs, &rhs)| lhs + rhs)
                .collect();

            return FactorKind::Unary(UnaryFactor::new(lhs_var, data));
        }

        // Disjoint variables: result becomes a dense Cartesian product.

        let (scope, lhs_data, rhs_data) = if lhs_var < rhs_var {
            (
                vec![lhs_var, rhs_var],
                self.data.as_slice(),
                other.data.as_slice(),
            )
        } else {
            (
                vec![rhs_var, lhs_var],
                other.data.as_slice(),
                self.data.as_slice(),
            )
        };

        let mut out_data = ArrayD::<f64>::zeros(IxDyn(&[lhs_data.len(), rhs_data.len()]));

        let mut out_iter = out_data.iter_mut();

        for &lhs in lhs_data {
            for &rhs in rhs_data {
                *out_iter
                    .next()
                    .expect("output iterator length must match shape") = lhs + rhs;
            }
        }
        FactorKind::Dense(DenseFactor::new(scope, out_data))
    }
}

impl Factor for UnaryFactor {
    fn scope(&self) -> &[VariableId] {
        std::slice::from_ref(&self.var)
    }
}

impl DiscreteFactor for UnaryFactor {
    fn card(&self) -> &[usize] {
        &self.card
    }
}

impl FactorDistance for UnaryFactor {
    fn distance(&self, other: &Self) -> f64 {
        assert_eq!(
            self.var(),
            other.var(),
            "distance requires matching variables"
        );
        assert_eq!(
            self.data().len(),
            other.data().len(),
            "distance requires matching cardinalities"
        );

        self.data()
            .iter()
            .zip(other.data())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max)
    }
}

impl FactorOps<LogSumProduct> for UnaryFactor
where
    DenseFactor: FactorOps<LogSumProduct>,
    ScalarFactor: FactorOps<LogSumProduct>,
{
    /// Reduces the factor over the specified variables.
    ///
    /// If this factor's variable is reduced, the result is a scalar factor.
    /// Otherwise, the factor is returned unchanged.
    fn reduce(self, vars: &[VariableId]) -> FactorKind {
        if vars.contains(&self.var) {
            let reduced = lse_two_pass(self.data());
            FactorKind::Scalar(ScalarFactor::new(reduced))
        } else {
            FactorKind::Unary(self)
        }
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        match other {
            FactorKind::Dense(d) => {
                <DenseFactor as FactorOps<LogSumProduct>>::combine(d, FactorKind::Unary(self))
            }
            FactorKind::Unary(other) => self.combine_unary(other),
            FactorKind::Scalar(s) => {
                let data = self.data.iter().map(|x| x + s.value()).collect::<Vec<_>>();

                FactorKind::Unary(UnaryFactor::new(self.var, data))
            }
        }
    }
}

impl FactorOps<LogMaxProduct> for UnaryFactor
where
    DenseFactor: FactorOps<LogMaxProduct>,
    ScalarFactor: FactorOps<LogMaxProduct>,
{
    fn reduce(self, vars: &[VariableId]) -> FactorKind {
        if vars.contains(&self.var) {
            let reduced = self
                .data()
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            FactorKind::Scalar(ScalarFactor::new(reduced))
        } else {
            FactorKind::Unary(self)
        }
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        match other {
            FactorKind::Dense(d) => {
                <DenseFactor as FactorOps<LogMaxProduct>>::combine(d, FactorKind::Unary(self))
            }
            FactorKind::Unary(other) => self.combine_unary(other),
            FactorKind::Scalar(s) => {
                let data = self.data().iter().map(|x| x + s.value()).collect();

                FactorKind::Unary(UnaryFactor::new(self.var, data))
            }
        }
    }
}

impl FactorNormalize<LogSumProduct> for UnaryFactor {
    fn normalize(self) -> Self {
        let normalizer = lse_two_pass(self.data());

        let data = self.data.iter().map(|value| value - normalizer).collect();

        UnaryFactor::new(self.var(), data)
    }
}

impl FactorNormalize<LogMaxProduct> for UnaryFactor {
    fn normalize(self) -> Self {
        let normalizer = self
            .data
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let data = self.data.iter().map(|value| value - normalizer).collect();

        UnaryFactor::new(self.var(), data)
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

    fn ln_array(v: &[f64]) -> Vec<f64> {
        v.iter().map(|x| x.ln()).collect()
    }

    //
    // Construction Tests
    //

    #[test]
    fn test_new() {
        let f = UnaryFactor::new(v(3), vec![1.0, 2.0, 3.0]);

        assert_eq!(f.var(), v(3));
        assert_eq!(f.scope(), &[v(3)]);
        assert_eq!(f.card(), &[3]);
        assert_eq!(f.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_linear() {
        let f = UnaryFactor::from_linear(v(2), vec![1.0, 10.0, 100.0]);

        let expected = ln_array(&[1.0, 10.0, 100.0]);

        assert_eq!(f.var(), v(2));
        assert_eq!(f.scope(), &[v(2)]);
        assert_eq!(f.card(), &[3]);

        for (actual, expected) in f.data().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-12);
        }
    }

    #[test]
    fn test_into_data() {
        let f = UnaryFactor::new(v(7), vec![1.0, 2.0, 3.0]);

        assert_eq!(f.into_data(), vec![1.0, 2.0, 3.0]);
    }

    //
    // reduce Tests
    //

    #[test]
    fn test_reduce_in_scope() {
        let f = UnaryFactor::new(v(0), ln_array(&[1.0, 2.0]));

        let result = <UnaryFactor as FactorOps<LogSumProduct>>::reduce(f, &[v(0)]);

        match result {
            FactorKind::Scalar(s) => {
                let expected = (1.0_f64 + 2.0).ln();
                assert!((s.value() - expected).abs() <= 1e-12);
            }
            _ => panic!("Expected scalar factor"),
        }
    }

    #[test]
    fn test_reduce_out_of_scope() {
        let f = UnaryFactor::new(v(0), vec![1.0, 2.0]);

        let result = <UnaryFactor as FactorOps<LogSumProduct>>::reduce(f, &[v(1)]);

        match result {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[v(0)]);
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
        let f = UnaryFactor::new(v(0), vec![1.0, 2.0]);
        let g = UnaryFactor::new(v(0), vec![10.0, 20.0]);

        let result = <UnaryFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Unary(g));

        match result {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[v(0)]);
                assert_eq!(u.data(), &[11.0, 22.0]);
            }
            _ => panic!("Expected unary factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_unary_intersect_three_states() {
        let f = UnaryFactor::new(v(0), vec![1.0, 2.0, 3.0]);
        let g = UnaryFactor::new(v(0), vec![10.0, 20.0, 30.0]);

        let result = <UnaryFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Unary(g));

        match result {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[v(0)]);
                assert_eq!(u.data(), &[11.0, 22.0, 33.0]);
            }
            _ => panic!("Expected unary factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_unary_disjoint() {
        let f = UnaryFactor::new(v(0), vec![1.0, 2.0]);
        let g = UnaryFactor::new(v(1), vec![10.0, 20.0]);

        let result = <UnaryFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Unary(g));

        match result {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[v(0), v(1)]);

                let expected = array![[11.0, 21.0], [12.0, 22.0],].into_dyn();

                assert_eq!(d.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_unary_unsorted_scope() {
        let f = UnaryFactor::new(v(1), vec![1.0, 2.0]);
        let g = UnaryFactor::new(v(0), vec![10.0, 20.0]);

        let result = <UnaryFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Unary(g));

        match result {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[v(0), v(1)]);

                let expected = array![[11.0, 12.0], [21.0, 22.0],].into_dyn();

                assert_eq!(d.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_scalar() {
        let f = UnaryFactor::new(v(0), vec![1.0, 2.0]);
        let scalar = ScalarFactor::new(10.0);

        let result =
            <UnaryFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Scalar(scalar));

        match result {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[v(0)]);
                assert_eq!(u.data(), &[11.0, 12.0]);
            }
            _ => panic!("Expected unary factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_dense_intersect() {
        let f = UnaryFactor::new(v(1), vec![10.0, 20.0]);

        let dense = DenseFactor::new(vec![v(0), v(1)], array![[1.0, 2.0], [3.0, 4.0],].into_dyn());

        let result =
            <UnaryFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Dense(dense));

        match result {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[v(0), v(1)]);

                let expected = array![[11.0, 22.0], [13.0, 24.0],].into_dyn();

                assert_eq!(d.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_dense_disjoint() {
        let f = UnaryFactor::new(v(2), vec![10.0, 20.0]);

        let dense = DenseFactor::new(vec![v(0), v(1)], array![[1.0, 2.0], [3.0, 4.0],].into_dyn());

        let result =
            <UnaryFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Dense(dense));
        match result {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[v(0), v(1), v(2)]);

                let expected =
                    array![[[11.0, 12.0], [13.0, 14.0]], [[21.0, 22.0], [23.0, 24.0]],].into_dyn();

                assert_eq!(d.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }

    #[test]
    fn test_combine_unary_x_dense_unsorted_scope() {
        let f = UnaryFactor::new(v(1), vec![10.0, 20.0]);

        let dense = DenseFactor::new(vec![v(1), v(0)], array![[1.0, 2.0], [3.0, 4.0],].into_dyn());

        let result =
            <UnaryFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Dense(dense));

        match result {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[v(0), v(1)]);

                let expected = array![[11.0, 23.0], [12.0, 24.0],].into_dyn();

                assert_eq!(d.data(), &expected);
            }
            _ => panic!("Expected dense factor"),
        }
    }

    //
    // Normalize tests
    //

    #[test]
    fn test_normalize_log_sum_product() {
        let factor = UnaryFactor::from_linear(v(0), vec![2.0, 3.0]);

        let normalized =
            <UnaryFactor as FactorNormalize<LogSumProduct>>::normalize(factor);

        let expected = [(2.0_f64 / 5.0).ln(), (3.0_f64 / 5.0).ln()];

        for (actual, expected) in normalized.data().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_normalize_log_max_product() {
        let factor = UnaryFactor::from_linear(v(0), vec![2.0, 4.0]);

        let normalized =
            <UnaryFactor as FactorNormalize<LogMaxProduct>>::normalize(factor);

        let expected = [(2.0_f64 / 4.0).ln(), 0.0];

        for (actual, expected) in normalized.data().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
