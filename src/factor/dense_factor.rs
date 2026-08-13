//! Dense table-based discrete factor implementation (log-space).
//!
//! A `DenseFactor` stores:
//! - a `scope`: variable IDs
//! - a dense `ndarray::ArrayD<f64>` data containing log-potentials
//!
//! Marginalization is implemented by permuting axes so eliminated axes are fastest-
//! changing, then summing (in log-space via LSE) over contiguous blocks.

use std::f64;
use ndarray::{ArrayD, IxDyn};

use super::Factor;
use super::DiscreteFactor;
use super::log_utils::{lse_update, lse_finalize};

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

    /// Access the underlying data (log-space).
    pub fn data(&self) -> &ArrayD<f64> {
        &self.data
    }

    /// Private kernel: permute so kept axes come first, eliminated axes last,
    /// make contiguous, then iterate contiguous blocks of length `elim_size`
    /// and reduce each block with LSE.
    fn marginalize_kernel(&self, vars: &[usize]) -> DenseFactor {
        // 1) partition axes
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
            if j < vars_sorted.len() && vars_sorted[j] == v {
                idx_marg.push(i);
            } else {
                idx_keep.push(i);
                new_scope.push(v);
            }
        }

        // 2) fast path: nothing to eliminate
        if idx_marg.is_empty() {
            return DenseFactor::new(self.scope.clone(), self.data.clone());
        }

        // 3) permutation: KEPT axes first, then MARGINALIZED axes last
        let mut order = Vec::with_capacity(idx_keep.len() + idx_marg.len());
        order.extend_from_slice(&idx_keep);
        order.extend_from_slice(&idx_marg);

        // 4) construct permuted view
        let permuted = self.data.view().permuted_axes(order);

        // 5) shapes and sizes (derive cardinalities from data.shape())
        let kept_shape: Vec<usize> = idx_keep.iter().map(|&i| self.data.shape()[i]).collect();
        let kept_size = if kept_shape.is_empty() { 1 } else { kept_shape.iter().product::<usize>() };

        // 6) all eliminated -> scalar DenseFactor (log-space)
        if kept_shape.is_empty() {
            let mut cur_max = f64::NEG_INFINITY;
            let mut cur_sum = 0.0;
            for &x in permuted.iter() {
                lse_update(x, &mut cur_max, &mut cur_sum);
            }
            let total = lse_finalize(cur_max, cur_sum);
            let out = ArrayD::from_elem(IxDyn(&[]), total);
            return DenseFactor::new(Vec::new(), out);
        }

        // 7) compute eliminated size and reshape into (kept_size, elim_size)
        let elim_size = idx_marg.iter().map(|&i| self.data.shape()[i]).product::<usize>();

        // 8) iterate over permuted view without copying:
        //    permuted.iter() traverses elements in logical order where the last axis
        //    is the fastest-changing. Because we placed marginalized axes last,
        //    marginalized axes are the fastest-changing and thus contiguous blocks
        //    of length `elim_size` correspond to one kept-assignment.
        let mut out_vec = Vec::with_capacity(kept_size);
        let mut it = permuted.iter();

        for _ in 0..kept_size {
            let mut cur_max = f64::NEG_INFINITY;
            let mut cur_sum = 0.0;
            for _ in 0..elim_size {
                // unwrap is safe because sizes are consistent
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


/// Implement Factor trait
impl Factor for DenseFactor {
    fn scope(&self) -> &[usize] {
        &self.scope
    }

    fn marginalize(&self, vars: &[usize]) -> Self {
        self.marginalize_kernel(vars)
    }
}

/// Implement DiscreteFactor trait without storing cards explicitly:
/// return the ndarray shape slice which corresponds to axis cardinalities.
impl DiscreteFactor for DenseFactor {
    fn card(&self) -> &[usize] {
        self.data.shape()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Helpers to build log-space arrays for tests
    fn ln_array1(v: &[f64]) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(&[v.len()]), v.iter().map(|x| x.ln()).collect()).unwrap()
    }

    // Elementwise absolute tolerance comparison for ArrayD<f64>
    fn arrays_close(a: &ArrayD<f64>, b: &ArrayD<f64>, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn test_scope() {
        let f = DenseFactor::new(
            vec![0, 1],
            array![[1.0_f64, 2.0], [3.0, 4.0]].mapv(|x| x.ln()).into_dyn(),
        );
        assert_eq!(f.scope(), &[0, 1]);
        assert_eq!(f.ndim(), 2);
    }

    #[test]
    fn test_marginalize_single_var_logspace() {
        // data in linear: [[1,2],[3,4]] -> log-space stored
        let data = array![[1.0_f64, 2.0], [3.0, 4.0]].mapv(|x| x.ln()).into_dyn();
        let f = DenseFactor::new(vec![0, 1], data);

        // marginalize var 0: sums are [1+3, 2+4] = [4,6] -> log([4,6])
        let g = f.marginalize(&[0]);
        let expected = ln_array1(&[4.0, 6.0]);

        assert_eq!(g.scope(), &[1]);
        assert!(arrays_close(g.data(), &expected, 1e-12));
    }

    #[test]
    fn test_marginalize_multiple_vars_logspace() {
        // data in linear:
        // [
        //   [[1,2],[3,4]],
        //   [[5,6],[7,8]]
        // ]
        let data = array![
            [[1.0_f64, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]
        .mapv(|x| x.ln())
        .into_dyn();

        let f = DenseFactor::new(vec![0, 1, 2], data);

        // marginalize 0 and 2 -> sums per kept axis 1:
        // for axis1=0: 1+2+5+6 = 14
        // for axis1=1: 3+4+7+8 = 22
        let g = f.marginalize(&[0, 2]);
        let expected = ln_array1(&[14.0, 22.0]);

        assert_eq!(g.scope(), &[1]);
        assert!(arrays_close(g.data(), &expected, 1e-12));
    }
}
