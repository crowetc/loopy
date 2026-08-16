//! Dense table-based discrete factor implementation (log-space).
//!
//! A `DenseFactor` stores:
//! - a `scope`: variable IDs
//! - a dense `ndarray::ArrayD<f64>` data containing log-potentials

use std::f64;
use ndarray::{ArrayD, IxDyn};

use super::{DiscreteFactor, Factor, FactorKind, ScalarFactor};
use super::log_utils::{lse_finalize, lse_update};

/// Dense table-based factor over discrete variables (log-space).
#[derive(Clone, Debug)]
pub struct DenseFactor {
    scope: Vec<usize>,
    data: ArrayD<f64>, // log-potentials
}

impl DenseFactor {
    /// Create a new dense factor.
    ///
    /// # Panics
    /// Panics if `scope.len() != data.ndim()`.
    pub fn new(scope: Vec<usize>, data: ArrayD<f64>) -> Self {
        assert_eq!(scope.len(), data.ndim());
        Self { scope, data }
    }

    /// Construct a dense factor from **linear-space** values.
    ///
    /// # Panics
    /// Panics if `scope.len() != linear.ndim()`.
    pub fn from_linear(scope: Vec<usize>, linear: ArrayD<f64>) -> Self {
        assert_eq!(
            scope.len(),
            linear.ndim(),
            "scope length must match number of data dimensions"
        );

        // Map all values to log-space without altering shape.
        let data = linear.mapv(|x| x.ln());
        Self { scope, data }
    }

    /// Access the underlying data (log-space).
    pub fn data(&self) -> &ArrayD<f64> {
        &self.data
    }

    /// Consume the factor and return `(scope, data)`.
    pub fn into_parts(self) -> (Vec<usize>, ArrayD<f64>) {
        (self.scope, self.data)
    }

    /// Consume and return the owned data array.
    pub fn into_data(self) -> ArrayD<f64> {
        self.data
    }

    /// Consume and return the owned scope vector.
    pub fn into_scope(self) -> Vec<usize> {
        self.scope
    }

    /// Core marginalization kernel.
    ///
    /// This method:
    /// 1. Partitions axes into kept vs. marginalized.
    /// 2. Permutes so kept axes come first.
    /// 3. Iterates contiguous blocks corresponding to marginalized axes.
    /// 4. Performs log-sum-exp reduction over each block.
    pub(crate) fn marginalize_kernel(&self, vars: &[usize]) -> DenseFactor {
        // ---- Partition axes -------------------------------------------------
        let mut vars_sorted = vars.to_vec();
        vars_sorted.sort_unstable();

        let mut idx_marg = Vec::with_capacity(vars_sorted.len());
        let mut idx_keep = Vec::with_capacity(self.scope.len());
        let mut new_scope = Vec::with_capacity(self.scope.len());

        let mut j = 0;
        for (i, &v) in self.scope.iter().enumerate() {
            while j < vars_sorted.len() && vars_sorted[j] < v {
                j += 1;
            }

            let eliminate = j < vars_sorted.len() && vars_sorted[j] == v;

            if eliminate {
                idx_marg.push(i);
            } else {
                idx_keep.push(i);
                new_scope.push(v);
            }
        }

        // ---- Fast path: nothing to eliminate --------------------------------
        if idx_marg.is_empty() {
            return self.clone();
        }

        // ---- Permute axes ----------------------------------------------------
        let mut order = Vec::with_capacity(idx_keep.len() + idx_marg.len());
        order.extend_from_slice(&idx_keep);
        order.extend_from_slice(&idx_marg);

        let permuted = self.data.view().permuted_axes(order);

        // ---- Compute shapes --------------------------------------------------
        let shape = self.data.shape();
        let kept_shape: Vec<usize> = idx_keep.iter().map(|&i| shape[i]).collect();
        let kept_size = if kept_shape.is_empty() { 1 } else { kept_shape.iter().product::<usize>() };

        // ---- Iterate contiguous blocks --------------------------------------
        let elim_size = idx_marg.iter().map(|&i| shape[i]).product::<usize>();

        let mut out_vec = Vec::with_capacity(kept_size);
        let mut it = permuted.iter();

        for _ in 0..kept_size {
            let mut cur_max = f64::NEG_INFINITY;
            let mut cur_sum = 0.0;

            for _ in 0..elim_size {
                let &x = it.next().expect("iterator length must match kept_size * elim_size");
                lse_update(x, &mut cur_max, &mut cur_sum);
            }

            out_vec.push(lse_finalize(cur_max, cur_sum));
        }

        debug_assert_eq!(out_vec.len(), kept_size);

        let out = ArrayD::from_shape_vec(IxDyn(&kept_shape), out_vec)
            .expect("shape and data length must match");

        DenseFactor::new(new_scope, out)
    }
}


/// Implement `Factor` trait
impl Factor for DenseFactor {
    fn scope(&self) -> &[usize] {
        &self.scope
    }

    fn marginalize(self, vars: &[usize]) -> FactorKind {
        let reduced = self.marginalize_kernel(vars);
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
            FactorKind::Dense(d2) => {
                // -----------------------------
                // 1. Partition scopes
                // -----------------------------
                let f_scope = &self.scope;
                let g_scope = d2.scope();

                let mut f_only = Vec::new();
                let mut g_only = Vec::new();
                let mut shared = Vec::new();

                for &v in f_scope {
                    if g_scope.contains(&v) {
                        shared.push(v);
                    } else {
                        f_only.push(v);
                    }
                }
                for &v in g_scope {
                    if !shared.contains(&v) {
                        g_only.push(v);
                    }
                }

                // -----------------------------
                // 2. Compute block sizes
                // -----------------------------
                let card_f = self.data.shape().to_vec();
                let card_g = d2.data().shape().to_vec();

                let f_only_size: usize = f_only.iter()
                    .map(|v| card_f[f_scope.iter().position(|x| *x == *v).unwrap()])
                    .product();

                let shared_size: usize = shared.iter()
                    .map(|v| card_f[f_scope.iter().position(|x| *x == *v).unwrap()])
                    .product();

                let g_only_size: usize = g_only.iter()
                    .map(|v| card_g[g_scope.iter().position(|x| *x == *v).unwrap()])
                    .product();

                // -----------------------------
                // 3. Output scope + shape
                // -----------------------------
                let mut new_scope = Vec::new();
                new_scope.extend(&f_only);
                new_scope.extend(&shared);
                new_scope.extend(&g_only);

                let mut out_shape = Vec::new();
                for &v in &new_scope {
                    if let Some(idx) = f_scope.iter().position(|x| *x == v) {
                        out_shape.push(card_f[idx]);
                    } else {
                        let idx = g_scope.iter().position(|x| *x == v).unwrap();
                        out_shape.push(card_g[idx]);
                    }
                }

                let mut out = ArrayD::<f64>::zeros(IxDyn(&out_shape));

                // -----------------------------
                // 4. Triple-loop combine kernel
                // -----------------------------
                let mut f_iter = self.data.iter();
                let mut out_iter = out.iter_mut();

                for _ in 0..f_only_size {
                    let mut g_iter = d2.data().iter();
                    for _ in 0..shared_size {
                        for _ in 0..g_only_size {
                            let f_val = *f_iter.next().unwrap();
                            let g_val = *g_iter.next().unwrap();
                            *out_iter.next().unwrap() = f_val + g_val;
                        }
                    }
                }

                FactorKind::Dense(DenseFactor::new(new_scope, out))
            }

            FactorKind::Unary(u) => {
                let f_scope = &self.scope;
                let u_var = u.scope()[0];

                // -----------------------------
                // 1. Partition scopes
                // -----------------------------
                let mut f_only = Vec::new();
                let mut shared = Vec::new();
                let mut u_only = Vec::new();

                for &v in f_scope {
                    if v == u_var {
                        shared.push(v);
                    } else {
                        f_only.push(v);
                    }
                }

                if !shared.contains(&u_var) {
                    u_only.push(u_var);
                }

                // -----------------------------
                // 2. Compute block sizes
                // -----------------------------
                let card_f = self.data.shape().to_vec();
                let card_u = u.data().len();

                // f_only_size = product of cardinals of f_only
                let f_only_size: usize = f_only.iter()
                    .map(|v| card_f[f_scope.iter().position(|x| *x == *v).unwrap()])
                    .product();

                // shared_size = product of cardinals of shared
                // unary has only one variable, so this is either card_u or 1
                let shared_size: usize = if shared.is_empty() {
                    1
                } else {
                    card_u
                };

                // u_only_size = product of cardinals of u_only
                // unary has only one variable, so this is either card_u or 1
                let u_only_size: usize = if u_only.is_empty() {
                    1
                } else {
                    card_u
                };

                // -----------------------------
                // 3. Output scope + shape
                // -----------------------------
                let mut new_scope = Vec::new();
                new_scope.extend(&f_only);
                new_scope.extend(&shared);
                new_scope.extend(&u_only);

                let mut out_shape = Vec::new();
                for &v in &new_scope {
                    if let Some(idx) = f_scope.iter().position(|x| *x == v) {
                        out_shape.push(card_f[idx]);
                    } else {
                        out_shape.push(card_u);
                    }
                }

                let mut out = ArrayD::<f64>::zeros(IxDyn(&out_shape));

                // -----------------------------
                // 4. Triple-loop combine kernel
                // -----------------------------
                let mut f_iter = self.data.iter();
                let mut out_iter = out.iter_mut();

                for _ in 0..f_only_size {
                    for j in 0..shared_size {
                        let u_val = if shared.is_empty() {
                            // unary var not in dense scope
                            0.0
                        } else {
                            u.data()[j]
                        };

                        for k in 0..u_only_size {
                            let f_val = *f_iter.next().unwrap();
                            let add_val = if u_only.is_empty() {
                                u_val
                            } else {
                                u.data()[k]
                            };

                            *out_iter.next().unwrap() = f_val + add_val;
                        }
                    }
                }

                FactorKind::Dense(DenseFactor::new(new_scope, out))
            }

            FactorKind::Scalar(s) => {
                let data = self.data.mapv(|x| x + s.value());
                FactorKind::Dense(DenseFactor::new(self.scope.clone(), data))
            }
        }
    }
}

/// Implement `DiscreteFactor` trait.
impl DiscreteFactor for DenseFactor {
    fn card(&self) -> &[usize] {
        self.data.shape()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn ln_array1(v: &[f64]) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(&[v.len()]), v.iter().map(|x| x.ln()).collect()).unwrap()
    }

    fn arrays_approx_equal(a: &ArrayD<f64>, b: &ArrayD<f64>, tol: f64) -> bool {
        a.shape() == b.shape()
            && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn test_scope() {
        let f = DenseFactor::new(
            vec![0, 1],
            array![[1.0_f64, 2.0], [3.0, 4.0]].mapv(|x| x.ln()).into_dyn(),
        );
        assert_eq!(f.scope(), &[0, 1]);
        assert_eq!(f.data().ndim(), 2);
    }

    #[test]
    fn test_marginalize_single_in_scope() {
        let data = array![[1.0_f64, 2.0], [3.0, 4.0]]
            .mapv(|x| x.ln())
            .into_dyn();

        let f = DenseFactor::new(vec![0, 1], data);

        let g = f.marginalize(&[0]);
        let expected = ln_array1(&[4.0, 6.0]);

        match g {
            FactorKind::Dense(d) => {
                panic!("Unexpected dense factor: {:?}", d);
            }
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[1]);
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
    fn test_marginalize_single_out_of_scope() {
        let data = array![[1.0_f64, 2.0], [3.0, 4.0]]
            .mapv(|x| x.ln())
            .into_dyn();

        let f = DenseFactor::new(vec![0, 1], data);

        let g = f.marginalize(&[2]);
        let expected = array![[1.0_f64, 2.0], [3.0, 4.0]]
            .mapv(|x| x.ln())
            .into_dyn();

        match g {
            FactorKind::Dense(d) => {
                assert_eq!(d.scope(), &[0, 1]);
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
    fn test_marginalize_multiple_vars_logspace() {
        let data = array![
            [[1.0_f64, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]
        .mapv(|x| x.ln())
        .into_dyn();

        let f = DenseFactor::new(vec![0, 1, 2], data);

        let g = f.marginalize(&[0, 2]);
        let expected = ln_array1(&[14.0, 22.0]);

        match g {
            FactorKind::Dense(d) => {
                panic!("Unexpected dense factor: {:?}", d);
            }
            FactorKind::Unary(u) => {
                assert_eq!(u.scope(), &[1]);
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
    fn test_marginalize_to_scalar() {
        let data = array![1.0_f64, 2.0]
            .mapv(|x| x.ln())
            .into_dyn();

        let f = DenseFactor::new(vec![0], data);

        let g = f.marginalize(&[0]);

        match g {
            FactorKind::Scalar(s) => {
                let expected = (1.0_f64 + 2.0_f64).ln();
                assert!((s.value() - expected).abs() <= 1e-12);
            }
            _ => panic!("Expected scalar factor"),
        }
    }
}
