//! Dense table-based discrete factor implementation (log-space).
//!
//! A `DenseFactor` stores:
//! - a `scope`: variable IDs
//! - a dense `ndarray::ArrayD<f64>` data containing log-potentials

use std::f64;
use ndarray::{ArrayD, IxDyn};

use super::{
    DiscreteFactor, Factor, FactorKind, ScalarFactor, UnaryFactor
};
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
        assert_eq!(
            scope.len(),
            data.ndim(),
            "scope length must match number of data dimensions"
        );

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

    //
    // Marginalization
    //

    /// Core marginalization kernel.
    ///
    /// This method:
    /// 1. Partitions axes into kept vs. marginalized.
    /// 2. Permutes so kept axes come first.
    /// 3. Iterates contiguous blocks corresponding to marginalized axes.
    /// 4. Performs log-sum-exp reduction over each block.
    pub(crate) fn marginalize_kernel(&self, vars: &[usize]) -> DenseFactor {
        //
        // Partition Axes
        //
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

        // Nothing to eliminate
        if idx_marg.is_empty() {
            return self.clone();
        }

        //
        // Permute Axes
        //
        let mut order = Vec::with_capacity(self.scope.len());
        order.extend_from_slice(&idx_keep);
        order.extend_from_slice(&idx_marg);

        let permuted = self.data.view().permuted_axes(order);

        //
        // Block Sizes
        //
        let shape = self.data.shape();
        let kept_shape: Vec<usize> =
            idx_keep.iter().map(|&i| shape[i]).collect();

        let kept_size = if kept_shape.is_empty() { 1 } else { kept_shape.iter().product::<usize>() };
        let elim_size = idx_marg.iter().map(|&i| shape[i]).product::<usize>();

        //
        // Log-Sum-Exp Reduction
        //
        let mut out_vec = Vec::with_capacity(kept_size);
        let mut it = permuted.iter();

        for _ in 0..kept_size {
            let mut cur_max = f64::NEG_INFINITY;
            let mut cur_sum = 0.0;

            for _ in 0..elim_size {
                let &x = it
                    .next()
                    .expect("iterator length must match expected size");
                lse_update(x, &mut cur_max, &mut cur_sum);
            }

            out_vec.push(lse_finalize(cur_max, cur_sum));
        }

        debug_assert_eq!(out_vec.len(), kept_size);

        let out = ArrayD::from_shape_vec(IxDyn(&kept_shape), out_vec)
            .expect("shape and data length must match");

        DenseFactor::new(new_scope, out)
    }

    //
    // Combination
    //

    /// Combine two dense factors in log-space.
    ///
    /// The resulting scope is the sorted union of both scopes.
    ///
    /// The views are arranged as:
    ///
    /// ```text
    /// f1: [common, f1-only]
    /// f2: [f2-only, common]
    /// f3: [f2-only, common, f1-only]
    /// ```
    ///
    /// This makes the Cartesian-product combination a simple nested
    /// iteration over contiguous logical blocks.
    fn combine_dense(&self, other: &DenseFactor) -> DenseFactor {
        let alignment = build_alignment(&self.scope, &other.scope);

        //
        // Validate cardinalities for shared variables
        //

        for (&f1_axis, &f2_axis) in alignment
            .f1_common
            .iter()
            .zip(alignment.f2_common.iter())
        {
            assert_eq!(
                self.card()[f1_axis],
                other.card()[f2_axis],
                "Cardinality mismatch in factors"
            );
        }

        //
        // Axis Orders
        //

        // f1: [common, f1-only]
        let mut f1_order = Vec::with_capacity(self.scope.len());
        f1_order.extend_from_slice(&alignment.f1_common);
        f1_order.extend_from_slice(&alignment.f1_only);

        // f2: [f2-only, common]
        let mut f2_order = Vec::with_capacity(other.scope.len());
        f2_order.extend_from_slice(&alignment.f2_only);
        f2_order.extend_from_slice(&alignment.f2_common);

        // f3: [f2-only, common, f1-only]
        let mut f3_order = Vec::with_capacity(alignment.union_vars.len());

        f3_order.extend(
            alignment
                .f2_only
                .iter()
                .map(|&axis| alignment.f2_to_f3[axis]),
        );

        f3_order.extend(
            alignment
                .f1_common
                .iter()
                .map(|&axis| alignment.f1_to_f3[axis]),
        );

        f3_order.extend(
            alignment
                .f1_only
                .iter()
                .map(|&axis| alignment.f1_to_f3[axis]),
        );

        //
        // Block Sizes
        //

        let common_size = product_or_one(
            alignment
                .f1_common
                .iter()
                .map(|&axis| self.card()[axis]),
        );

        let f1_only_size = product_or_one(
            alignment
                .f1_only
                .iter()
                .map(|&axis| self.card()[axis]),
        );

        let f2_only_size = product_or_one(
            alignment
                .f2_only
                .iter()
                .map(|&axis| other.card()[axis]),
        );

        //
        // Output
        //

        let f3_card = alignment.output_card(self.card(), other.card());
        let mut f3_data = ArrayD::<f64>::zeros(IxDyn(&f3_card));

        let f1_view = self.data.view().permuted_axes(f1_order);
        let f2_view = other.data.view().permuted_axes(f2_order);
        let mut f3_view = f3_data.view_mut().permuted_axes(f3_order);

        //
        // Kernel
        //

        let mut it2 = f2_view.iter();
        let mut it3 = f3_view.iter_mut();

        for _ in 0..f2_only_size {

            let mut it1 = f1_view.iter();

            for _ in 0..common_size {
                let b = *it2
                    .next()
                    .expect("f2 iterator length must match shape");

                for _ in 0..f1_only_size {
                    let a = *it1
                        .next()
                        .expect("f1 iterator length must match shape");

                    *it3
                        .next()
                        .expect("f3 iterator length must match shape") = a + b;
                }
            }
        }

        debug_assert!(it2.next().is_none());
        debug_assert!(it3.next().is_none());

        DenseFactor::new(alignment.union_vars, f3_data)
    }

    /// Combine a dense factor with a unary factor.
    ///
    /// For a shared variable:
    ///
    /// ```text
    /// dense:  [common, dense-only]
    /// output: [common, dense-only]
    /// ```
    ///
    /// For a disjoint variable:
    ///
    /// ```text
    /// dense:  [dense-only...]
    /// output: [unary-only, dense-only...]
    /// ```
    ///
    /// The latter case is a straightforward Cartesian product.
    fn combine_unary(&self, unary: &UnaryFactor) -> DenseFactor {
        debug_assert_eq!(
            unary.scope().len(),
            1,
            "UnaryFactor must have exactly one variable"
        );

        let alignment = build_alignment(&self.scope, unary.scope());

        const UNARY_AXIS: usize = 0;

        //
        // Case 1: unary variable is shared with the dense factor.
        //

        if alignment.f2_common.len() == 1 {
            let dense_common_axis = alignment.f1_common[0];

            assert_eq!(
                self.card()[dense_common_axis],
                unary.card()[UNARY_AXIS],
                "Cardinality mismatch in factors"
            );

            // Dense: [common, dense-only]
            let mut f1_order = Vec::with_capacity(self.scope.len());
            f1_order.extend_from_slice(&alignment.f1_common);
            f1_order.extend_from_slice(&alignment.f1_only);

            // Output: [common, dense-only]
            let mut f3_order = Vec::with_capacity(alignment.union_vars.len());

            f3_order.extend(
                alignment
                    .f1_common
                    .iter()
                    .map(|&axis| alignment.f1_to_f3[axis]),
            );

            f3_order.extend(
                alignment
                    .f1_only
                    .iter()
                    .map(|&axis| alignment.f1_to_f3[axis]),
            );

           let dense_view = self.data.view().permuted_axes(f1_order);

            let f3_card = alignment.output_card(self.card(), unary.card());
            let mut f3_data = ArrayD::<f64>::zeros(IxDyn(&f3_card));

            let mut f3_view = f3_data.view_mut().permuted_axes(f3_order);

            let dense_only_size = product_or_one(
                alignment
                    .f1_only
                    .iter()
                    .map(|&axis| self.card()[axis]),
            );

            let unary_data = unary.data();

            let mut dense_iter = dense_view.iter();
            let mut out_iter = f3_view.iter_mut();

            for &u in unary_data.iter() {
                for _ in 0..dense_only_size {
                    let x = *dense_iter
                        .next()
                        .expect("dense iterator length must match shape");

                    *out_iter
                        .next()
                        .expect("output iterator length must match shape") =
                        x + u;
                }
            }

            debug_assert!(dense_iter.next().is_none());
            debug_assert!(out_iter.next().is_none());

            return DenseFactor::new(alignment.union_vars, f3_data);
        } else {

            //
            // Case 2: unary variable is disjoint from the dense factor.
            //

            debug_assert_eq!(alignment.f2_only.len(), 1);

            // Dense: [dense-only...]
            let mut f1_order = Vec::with_capacity(self.scope.len());
            f1_order.extend_from_slice(&alignment.f1_only);

            // Output: [unary-only, dense-only...]
            let mut f3_order = Vec::with_capacity(alignment.union_vars.len());

            f3_order.extend(
                alignment
                    .f2_only
                    .iter()
                    .map(|&axis| alignment.f2_to_f3[axis]),
            );

            f3_order.extend(
                alignment
                    .f1_only
                    .iter()
                    .map(|&axis| alignment.f1_to_f3[axis]),
            );

            let dense_view = self.data.view().permuted_axes(f1_order);

            let f3_card = alignment.output_card(self.card(), unary.card());
            let mut f3_data = ArrayD::<f64>::zeros(IxDyn(&f3_card));

            let unary_data = unary.data();

            let dense_iter = dense_view.iter();
            let mut out_iter = f3_data.iter_mut();

            for &u in unary_data.iter() {
                for &x in dense_iter.clone() {
                    *out_iter
                        .next()
                        .expect("output iterator length must match shape") = x + u;
                }
            }

            debug_assert!(out_iter.next().is_none());

            DenseFactor::new(alignment.union_vars, f3_data)
        }
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
            FactorKind::Dense(other) => {
                if self.data.is_empty() {
                    return FactorKind::Dense(other);
                }

                if other.data.is_empty() {
                    return FactorKind::Dense(self);
                }

                FactorKind::Dense(self.combine_dense(&other))
            }

            FactorKind::Unary(unary) => {
                if self.data.is_empty() {
                    return FactorKind::Unary(unary);
                }

                if unary.data().is_empty() {
                    return FactorKind::Dense(self);
                }

                FactorKind::Dense(self.combine_unary(&unary))
            }

            FactorKind::Scalar(s) => {
                let data = self.data.mapv(|x| x + s.value());
                FactorKind::Dense(DenseFactor::new(self.scope, data))
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

/// Information needed to align two factor scopes.
///
/// The output scope is always the sorted union of the two input scopes.
///
/// The `*_to_f3` mappings map an input axis to its corresponding output axis.
struct Alignment {
    /// The union of variables from f1 and f2, in aligned order
    union_vars: Vec<usize>,

    /// Indices of variables that appear only in f1
    f1_only: Vec<usize>,

    /// Indices of variables that appear only in f2
    f2_only: Vec<usize>,

    /// Indices of variables common to both f1 and f2
    f1_common: Vec<usize>,
    f2_common: Vec<usize>,

    /// Mapping: f1 axis → f3 axis
    f1_to_f3: Vec<usize>,

    /// Mapping: f2 axis → f3 axis
    f2_to_f3: Vec<usize>,
}

impl Alignment {
    /// Construct the output cardinalities from the input cardinalities.
    fn output_card(
        &self,
        f1_card: &[usize],
        f2_card: &[usize],
    ) -> Vec<usize> {
        let mut card = vec![0usize; self.union_vars.len()];

        for (axis, &out_axis) in self.f1_to_f3.iter().enumerate() {
            if out_axis != usize::MAX {
                card[out_axis] = f1_card[axis];
            }
        }

        for (axis, &out_axis) in self.f2_to_f3.iter().enumerate() {
            if out_axis != usize::MAX {
                card[out_axis] = f2_card[axis];
            }
        }

        card
    }
}

/// Build the alignment between two factor scopes.
fn build_alignment(
    f1_scope: &[usize],
    f2_scope: &[usize]
) -> Alignment {
    //
    // Sort Scopes
    //
    let mut f1_sorted: Vec<(usize, usize)> = f1_scope
        .iter()
        .enumerate()
        .map(|(axis, &var)| (var, axis))
        .collect();

    let mut f2_sorted: Vec<(usize, usize)> = f2_scope
        .iter()
        .enumerate()
        .map(|(axis, &var)| (var, axis))
        .collect();

    f1_sorted.sort_unstable_by_key(|&(var, _)| var);
    f2_sorted.sort_unstable_by_key(|&(var, _)| var);

    let n1 = f1_sorted.len();
    let n2 = f2_sorted.len();
    let n_common = n1.min(n2);

    let mut union_vars = Vec::with_capacity(n1 + n2);

    let mut f1_only = Vec::with_capacity(n1);
    let mut f2_only = Vec::with_capacity(n2);

    let mut f1_common = Vec::with_capacity(n_common);
    let mut f2_common = Vec::with_capacity(n_common);

    let mut f1_to_f3 = vec![usize::MAX; n1];
    let mut f2_to_f3 = vec![usize::MAX; n2];

    let mut i = 0;
    let mut j = 0;
    let mut out_axis = 0;

    //
    // Merge walk
    //
    while i < n1 && j < n2 {
        let (v1, axis1) = f1_sorted[i];
        let (v2, axis2) = f2_sorted[j];

        match v1.cmp(&v2) {
            std::cmp::Ordering::Equal => {
                f1_common.push(axis1);
                f2_common.push(axis2);
                union_vars.push(v1);

                f1_to_f3[axis1] = out_axis;
                f2_to_f3[axis2] = out_axis;

                out_axis += 1;
                i += 1;
                j += 1;
            }

            std::cmp::Ordering::Less => {
                f1_only.push(axis1);
                union_vars.push(v1);

                f1_to_f3[axis1] = out_axis;

                out_axis += 1;
                i += 1;
            }

            std::cmp::Ordering::Greater => {
                f2_only.push(axis2);
                union_vars.push(v2);

                f2_to_f3[axis2] = out_axis;

                out_axis += 1;
                j += 1;
            }
        }
    }

    // remaining f1-only
    while i < n1 {
        let (var, axis) = f1_sorted[i];

        f1_only.push(axis);
        union_vars.push(var);
        f1_to_f3[axis] = out_axis;

        out_axis += 1;
        i += 1;
    }

    // remaining f2-only
    while j < n2 {
        let (var, axis) = f2_sorted[j];

        f2_only.push(axis);
        union_vars.push(var);
        f2_to_f3[axis] = out_axis;

        out_axis += 1;
        j += 1;
    }

    Alignment {
        union_vars,
        f1_only,
        f2_only,
        f1_common,
        f2_common,
        f1_to_f3,
        f2_to_f3,
    }
}

/// Product of dimensions, treating an empty product as one.
///
/// This is convenient for Cartesian-product kernels where an empty group of
/// axes still represents one logical assignment.
fn product_or_one<I>(values: I) -> usize
where
    I: IntoIterator<Item = usize>,
{
    values.into_iter().product::<usize>().max(1)
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, array};

    fn ln_array1(v: &[f64]) -> ArrayD<f64> {
        ArrayD::from_shape_vec(
            IxDyn(&[v.len()]),
            v.iter().map(|x| x.ln()).collect()
        )
        .unwrap()
    }

    fn arrays_approx_equal(
        a: &ArrayD<f64>,
        b: &ArrayD<f64>,
        tol: f64
    ) -> bool {
        a.shape() == b.shape()
            && a.iter()
                .zip(b.iter())
                .all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn test_scope() {
        let f = DenseFactor::new(
            vec![0, 1],
            array![[1.0_f64, 2.0], [3.0, 4.0]]
                .mapv(|x| x.ln()).into_dyn(),
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

    #[test]
    fn test_combine_dense_x_dense_intersect() {
        let f = DenseFactor::new(
            vec![0, 1],
            arr2(&[
                [1.0, 2.0],
                [3.0, 4.0]
            ])
            .into_dyn()
        );

        let g = DenseFactor::new(
            vec![1, 2],
            arr2(&[
                [10.0, 20.0],
                [30.0, 40.0]
            ])
            .into_dyn()
        );

        let out = match f.combine(FactorKind::Dense(g)) {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense")
        };

        // Expected shape: [0,1,2] -> [2,2,2]
        let expected = array![
            [[11.0, 21.0], [32.0, 42.0]],
            [[13.0, 23.0], [34.0, 44.0]]
        ]
        .into_dyn();

        assert_eq!(out.scope(), &[0, 1, 2]);
        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn test_combine_dense_x_dense_disjoint() {
        let f = DenseFactor::new(
            vec![0],
            arr1(&[1.0, 2.0]).into_dyn(),
        );

        let g = DenseFactor::new(
            vec![1],
            arr1(&[10.0, 20.0]).into_dyn(),
        );

        let out = match f.combine(FactorKind::Dense(g)) {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense"),
        };

        let expected = array![
            [11.0, 21.0],
            [12.0, 22.0],
        ]
        .into_dyn();

        assert_eq!(out.scope(), &[0, 1]);
        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn test_combine_dense_x_unary_intersect() {
        let f = DenseFactor::new(
            vec![0, 1],
            arr2(&[
                [1.0, 2.0],
                [3.0, 4.0]
            ])
            .into_dyn()
        );

        let u = UnaryFactor::new(1, vec![10.0, 20.0]);

        let out = match f.combine(FactorKind::Unary(u)) {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense")
        };

        let expected = arr2(&[
            [11.0, 22.0],
            [13.0, 24.0]
        ])
        .into_dyn();

        assert_eq!(out.scope(), &[0, 1]);
        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn test_combine_dense_x_unary_disjoint() {
        let f = DenseFactor::new(
            vec![0, 1],
            arr2(&[
                [1.0, 2.0],
                [3.0, 4.0],
            ])
            .into_dyn(),
        );

        let u = UnaryFactor::new(2, vec![10.0, 20.0]);

        let out = match f.combine(FactorKind::Unary(u)) {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense"),
        };

        let expected = array![
            [[11.0, 12.0], [13.0, 14.0]],
            [[21.0, 22.0], [23.0, 24.0]],
        ]
        .into_dyn();

        assert_eq!(out.scope(), &[0, 1, 2]);
        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn test_combine_dense_x_dense_unsorted_scopes() {
        // Scope [1, 0] means:
        //   axis 0 -> variable 1
        //   axis 1 -> variable 0
        let f = DenseFactor::new(
            vec![1, 0],
            array![
                [1.0, 2.0],
                [3.0, 4.0],
            ]
            .into_dyn(),
        );

        let g = DenseFactor::new(
            vec![2, 1],
            array![
                [10.0, 20.0],
                [30.0, 40.0],
            ]
            .into_dyn(),
        );

        let out = match f.combine(FactorKind::Dense(g)) {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense"),
        };

        assert_eq!(out.scope(), &[0, 1, 2]);

        let expected = array![
            [[11.0, 31.0], [23.0, 43.0]],
            [[12.0, 32.0], [24.0, 44.0]],
        ]
        .into_dyn();

        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn test_combine_dense_x_unary_unsorted_scope() {
        // Dense scope [1, 0]:
        //   axis 0 -> variable 1
        //   axis 1 -> variable 0
        let f = DenseFactor::new(
            vec![1, 0],
            array![
                [1.0, 2.0],
                [3.0, 4.0],
            ]
            .into_dyn(),
        );

        let u = UnaryFactor::new(1, vec![10.0, 20.0]);

        let out = match f.combine(FactorKind::Unary(u)) {
            FactorKind::Dense(d) => d,
            _ => panic!("Expected dense"),
        };

        assert_eq!(out.scope(), &[0, 1]);

        let expected = array![
            [11.0, 23.0],
            [12.0, 24.0],
        ]
        .into_dyn();

        assert_eq!(out.data(), &expected);
    }
}
