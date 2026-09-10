//! Dense discrete factor representation.
//!
//! A [`DenseFactor`] represents a factor over finite discrete variables using
//! an [`ndarray::ArrayD`] of log-space values.
//!
//! Each variable in the factor's scope corresponds to one axis of the data
//! array. The variable at `scope[i]` is represented by axis `i`, whose length
//! is the cardinality of that variable.
//!
//! Factor combination is performed by addition in log-space. Reduction is
//! provided for both [`LogSumProduct`] and [`LogMaxProduct`].

mod combine;
mod reduce;

use ndarray::ArrayD;

use crate::semiring::{LogMaxProduct, LogSumProduct};
use crate::variable::VariableId;

use super::log_utils::lse;
use super::{
    DiscreteFactor, Factor, FactorDistance, FactorKind, FactorNormalize, FactorOps, ScalarFactor,
    UnaryFactor,
};

/// A dense factor over finite discrete variables.
///
/// Factor values are stored in log-space. Each variable in [`Factor::scope`]
/// corresponds to the array axis at the same position.
///
/// For example, a factor with scope `[x, y]` and shape `[2, 3]` represents a
/// binary variable `x` and a three-state variable `y`.
#[derive(Clone, Debug)]
pub struct DenseFactor {
    scope: Vec<VariableId>,
    data: ArrayD<f64>, // log-potentials
}

impl DenseFactor {
    /// Creates a dense factor from log-space values.
    ///
    /// # Panics
    /// Panics if the number of variables in `scope` does not equal the number
    /// of dimensions in `data`.
    pub fn new(scope: Vec<VariableId>, data: ArrayD<f64>) -> Self {
        assert_eq!(
            scope.len(),
            data.ndim(),
            "scope length must match number of data dimensions"
        );

        Self { scope, data }
    }

    /// Construct a dense factor from linear-space values.
    ///
    /// # Panics
    /// Panics if the number of variables in `scope` does not equal the number
    /// of dimensions in `linear`.
    pub fn from_linear(scope: Vec<VariableId>, linear: ArrayD<f64>) -> Self {
        assert_eq!(
            scope.len(),
            linear.ndim(),
            "scope length must match number of data dimensions"
        );

        let data = linear.mapv(|x| x.ln());

        Self { scope, data }
    }

    /// Returns the factor values in log-space.
    pub fn data(&self) -> &ArrayD<f64> {
        &self.data
    }

    /// Consumes the factor and returns its scope and log-space data.
    pub fn into_parts(self) -> (Vec<VariableId>, ArrayD<f64>) {
        (self.scope, self.data)
    }

    /// Consumes the factor and returns its log-space data.
    pub fn into_data(self) -> ArrayD<f64> {
        self.data
    }

    /// Consumes the factor and returns its scope.
    pub fn into_scope(self) -> Vec<VariableId> {
        self.scope
    }
}

impl Factor for DenseFactor {
    fn scope(&self) -> &[VariableId] {
        &self.scope
    }
}

impl DiscreteFactor for DenseFactor {
    fn card(&self) -> &[usize] {
        self.data.shape()
    }
}

impl FactorDistance for DenseFactor {
    fn distance(&self, other: &Self) -> f64 {
        assert_eq!(self.scope, other.scope, "residual requires matching scopes");

        assert_eq!(
            self.data.shape(),
            other.data.shape(),
            "residual requires matching shapes"
        );

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max)
    }
}

impl FactorOps<LogSumProduct> for DenseFactor {
    fn reduce(self, vars: &[VariableId]) -> FactorKind {
        let reduced = self.reduce_sum_kernel(vars);
        match reduced.scope.len() {
            0 => {
                let val = reduced.data().iter().copied().next().unwrap();
                FactorKind::Scalar(ScalarFactor::new(val))
            }
            1 => FactorKind::Unary(crate::factor::utils::dense_into_unary(reduced)),
            _ => FactorKind::Dense(reduced),
        }
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        match other {
            FactorKind::Dense(other) => FactorKind::Dense(self.combine_dense(&other)),
            FactorKind::Unary(unary) => FactorKind::Dense(self.combine_unary(&unary)),

            FactorKind::Scalar(s) => {
                let data = self.data.mapv(|x| x + s.value());
                FactorKind::Dense(DenseFactor::new(self.scope, data))
            }
        }
    }
}

impl FactorOps<LogMaxProduct> for DenseFactor {
    fn reduce(self, vars: &[VariableId]) -> FactorKind {
        let reduced = self.reduce_max_kernel(vars);
        match reduced.scope.len() {
            0 => {
                let val = reduced.data().iter().copied().next().unwrap();
                FactorKind::Scalar(ScalarFactor::new(val))
            }
            1 => FactorKind::Unary(crate::factor::utils::dense_into_unary(reduced)),
            _ => FactorKind::Dense(reduced),
        }
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        match other {
            FactorKind::Dense(other) => FactorKind::Dense(self.combine_dense(&other)),
            FactorKind::Unary(unary) => FactorKind::Dense(self.combine_unary(&unary)),

            FactorKind::Scalar(s) => {
                let data = self.data.mapv(|x| x + s.value());
                FactorKind::Dense(DenseFactor::new(self.scope, data))
            }
        }
    }
}

impl FactorNormalize<LogSumProduct> for DenseFactor {
    fn normalize(self) -> Self {
        let normalizer = lse(self.data().iter().copied());

        let scope = self.scope().to_vec();
        let data = self.data().mapv(|value| value - normalizer);

        DenseFactor::new(scope, data)
    }
}

impl FactorNormalize<LogMaxProduct> for DenseFactor {
    fn normalize(self) -> Self {
        let normalizer = self
            .data()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let scope = self.scope().to_vec();
        let data = self.data().mapv(|value| value - normalizer);

        DenseFactor::new(scope, data)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{IxDyn, arr1, arr2, array};

    fn v(id: usize) -> VariableId {
        VariableId::new(id)
    }

    fn ln_array1(v: &[f64]) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(&[v.len()]), v.iter().map(|x| x.ln()).collect()).unwrap()
    }

    fn arrays_approx_equal(a: &ArrayD<f64>, b: &ArrayD<f64>, tol: f64) -> bool {
        a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn test_scope() {
        let f = DenseFactor::new(
            vec![v(0), v(1)],
            array![[1.0_f64, 2.0], [3.0, 4.0]]
                .mapv(|x| x.ln())
                .into_dyn(),
        );
        assert_eq!(f.scope(), &[v(0), v(1)]);
        assert_eq!(f.data().ndim(), 2);
    }

    #[test]
    fn test_reduce_single_in_scope() {
        let data = array![[1.0_f64, 2.0], [3.0, 4.0]]
            .mapv(|x| x.ln())
            .into_dyn();

        let f = DenseFactor::new(vec![v(0), v(1)], data);

        let g = <DenseFactor as FactorOps<LogSumProduct>>::reduce(f, &[v(0)]);
        let expected = ln_array1(&[4.0, 6.0]);

        match g {
            FactorKind::Dense(d) => {
                panic!("Unexpected dense factor: {:?}", d);
            }
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[v(1)]);
                let expected_vec = expected.iter().cloned().collect::<Vec<_>>();
                assert_eq!(u.data().len(), expected_vec.len());
                for (a, b) in u.data().iter().zip(expected_vec.iter()) {
                    assert!((a - b).abs() <= 1e-12);
                }
            }
            FactorKind::Scalar(s) => {
                panic!("Unexpected scalar factor: {:?}", s);
            }
        }
    }

    #[test]
    fn test_reduce_single_out_of_scope() {
        let data = array![[1.0_f64, 2.0], [3.0, 4.0]]
            .mapv(|x| x.ln())
            .into_dyn();

        let f = DenseFactor::new(vec![v(0), v(1)], data);

        let g = <DenseFactor as FactorOps<LogSumProduct>>::reduce(f, &[v(2)]);
        let expected = array![[1.0_f64, 2.0], [3.0, 4.0]]
            .mapv(|x| x.ln())
            .into_dyn();

        match g {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[v(0), v(1)]);
                assert!(
                    arrays_approx_equal(d.data(), &expected, 1e-12),
                    "Data changed when marginalizing an out-of-scope variable"
                );
            }
            FactorKind::Unary(u) => {
                panic!("Unexpected unary factor: {:?}", u);
            }
            FactorKind::Scalar(s) => {
                panic!("Unexpected scalar factor: {:?}", s);
            }
        }
    }

    #[test]
    fn test_reduce_multiple_vars_logspace() {
        let data = array![[[1.0_f64, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
            .mapv(|x| x.ln())
            .into_dyn();

        let f = DenseFactor::new(vec![v(0), v(1), v(2)], data);

        let g = <DenseFactor as FactorOps<LogSumProduct>>::reduce(f, &[v(0), v(2)]);
        let expected = ln_array1(&[14.0, 22.0]);

        match g {
            FactorKind::Dense(d) => {
                panic!("Unexpected dense factor: {:?}", d);
            }
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[v(1)]);
                let expected_vec = expected.iter().cloned().collect::<Vec<_>>();
                assert_eq!(u.data().len(), expected_vec.len());
                for (a, b) in u.data().iter().zip(expected_vec.iter()) {
                    assert!((a - b).abs() <= 1e-12);
                }
            }
            FactorKind::Scalar(s) => {
                panic!("Unexpected scalar factor: {:?}", s);
            }
        }
    }

    #[test]
    fn test_reduce_to_scalar() {
        let data = array![1.0_f64, 2.0].mapv(|x| x.ln()).into_dyn();

        let f = DenseFactor::new(vec![v(0)], data);

        let g = <DenseFactor as FactorOps<LogSumProduct>>::reduce(f, &[v(0)]);

        match g {
            FactorKind::Scalar(s) => {
                let expected = (1.0_f64 + 2.0_f64).ln();
                assert!((s.value() - expected).abs() <= 1e-12);
            }
            _ => panic!("Expected scalar factor"),
        }
    }

    #[test]
    fn test_reduce_unsorted_scope() {
        let data = array![[1.0_f64, 2.0], [3.0, 4.0]]
            .mapv(|x| x.ln())
            .into_dyn();

        let f = DenseFactor::new(vec![v(1), v(0)], data);

        let g = <DenseFactor as FactorOps<LogSumProduct>>::reduce(f, &[v(0)]);

        match g {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[v(1)]);

                let expected = [3.0_f64.ln(), 7.0_f64.ln()];

                for (actual, expected) in u.data().iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            _ => panic!("Expected unary factor"),
        }
    }

    #[test]
    fn test_reduce_unsorted_3d_scope() {
        let data = array![[[1.0_f64, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],]
            .mapv(|x| x.ln())
            .into_dyn();

        // axis 0 -> v2
        // axis 1 -> v0
        // axis 2 -> v1
        let f = DenseFactor::new(vec![v(2), v(0), v(1)], data);

        // Eliminate v0 and v2, leaving v1.
        let g = <DenseFactor as FactorOps<LogSumProduct>>::reduce(f, &[v(0), v(2)]);

        match g {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[v(1)]);

                // v1 = 0:
                //   1 + 3 + 5 + 7 = 16
                //
                // v1 = 1:
                //   2 + 4 + 6 + 8 = 20
                let expected = [16.0_f64.ln(), 20.0_f64.ln()];

                for (actual, expected) in u.data().iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            _ => panic!("Expected unary factor"),
        }
    }

    #[test]
    fn test_reduce_max_single_in_scope() {
        let data = array![[1.0_f64, 2.0], [3.0, 4.0]]
            .mapv(|x| x.ln())
            .into_dyn();

        let f = DenseFactor::new(vec![v(0), v(1)], data);

        let g = <DenseFactor as FactorOps<LogMaxProduct>>::reduce(f, &[v(0)]);

        match g {
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[v(1)]);

                let expected = [3.0_f64.ln(), 4.0_f64.ln()];

                for (actual, expected) in u.data().iter().zip(expected.iter()) {
                    assert!(
                        (actual - expected).abs() < 1e-12,
                        "Expected {}, got {}",
                        expected,
                        actual
                    );
                }
            }
            FactorKind::Dense(d) => {
                panic!("Expected unary factor, got dense factor: {:?}", d);
            }
            FactorKind::Scalar(s) => {
                panic!("Expected unary factor, got scalar factor: {:?}", s);
            }
        }
    }

    #[test]
    fn test_combine_dense_x_dense_intersect() {
        let f = DenseFactor::new(vec![v(0), v(1)], arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());

        let g = DenseFactor::new(
            vec![v(1), v(2)],
            arr2(&[[10.0, 20.0], [30.0, 40.0]]).into_dyn(),
        );

        let out = match <DenseFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Dense(g))
        {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense"),
        };

        // Expected shape: [0,1,2] -> [2,2,2]
        let expected =
            array![[[11.0, 21.0], [32.0, 42.0]], [[13.0, 23.0], [34.0, 44.0]]].into_dyn();

        assert_eq!(out.scope(), &[v(0), v(1), v(2)]);
        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn test_combine_dense_x_dense_disjoint() {
        let f = DenseFactor::new(vec![v(0)], arr1(&[1.0, 2.0]).into_dyn());

        let g = DenseFactor::new(vec![v(1)], arr1(&[10.0, 20.0]).into_dyn());

        let out = match <DenseFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Dense(g))
        {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense"),
        };

        let expected = array![[11.0, 21.0], [12.0, 22.0]].into_dyn();

        assert_eq!(out.scope(), &[v(0), v(1)]);
        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn test_combine_dense_x_unary_intersect() {
        let f = DenseFactor::new(vec![v(0), v(1)], arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());

        let u = UnaryFactor::new(v(1), vec![10.0, 20.0]);

        let out = match <DenseFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Unary(u))
        {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense"),
        };

        let expected = arr2(&[[11.0, 22.0], [13.0, 24.0]]).into_dyn();

        assert_eq!(out.scope(), &[v(0), v(1)]);
        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn test_combine_dense_x_unary_disjoint() {
        let f = DenseFactor::new(vec![v(0), v(1)], arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());

        let u = UnaryFactor::new(v(2), vec![10.0, 20.0]);

        let out = match <DenseFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Unary(u))
        {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense"),
        };

        let expected =
            array![[[11.0, 12.0], [13.0, 14.0]], [[21.0, 22.0], [23.0, 24.0]],].into_dyn();

        assert_eq!(out.scope(), &[v(0), v(1), v(2)]);
        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn test_combine_dense_x_dense_unsorted_scopes() {
        // Scope [1, 0] means:
        //   axis 0 -> variable 1
        //   axis 1 -> variable 0
        let f = DenseFactor::new(vec![v(1), v(0)], array![[1.0, 2.0], [3.0, 4.0],].into_dyn());

        let g = DenseFactor::new(
            vec![v(2), v(1)],
            array![[10.0, 20.0], [30.0, 40.0],].into_dyn(),
        );

        let out = match <DenseFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Dense(g))
        {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense"),
        };

        assert_eq!(out.scope(), &[v(0), v(1), v(2)]);

        let expected =
            array![[[11.0, 31.0], [23.0, 43.0]], [[12.0, 32.0], [24.0, 44.0]],].into_dyn();

        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn test_combine_dense_x_unary_unsorted_scope() {
        // Dense scope [1, 0]:
        //   axis 0 -> variable 1
        //   axis 1 -> variable 0
        let f = DenseFactor::new(vec![v(1), v(0)], array![[1.0, 2.0], [3.0, 4.0],].into_dyn());

        let u = UnaryFactor::new(v(1), vec![10.0, 20.0]);

        let out = match <DenseFactor as FactorOps<LogSumProduct>>::combine(f, FactorKind::Unary(u))
        {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense"),
        };

        assert_eq!(out.scope(), &[v(0), v(1)]);

        let expected = array![[11.0, 23.0], [12.0, 24.0],].into_dyn();

        assert_eq!(out.data(), &expected);
    }
}
