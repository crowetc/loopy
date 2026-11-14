//! Dense factor combination kernels.
//!
//! This module implements combination between dense factors and other
//! supported discrete factor representations. Factor scopes are aligned by
//! variable identifier, and values are combined by addition in log-space.

use ndarray::{ArrayD, IxDyn};

use super::DenseFactor;
use crate::factor::{DiscreteFactor, Factor, UnaryFactor};
use crate::variable::VariableId;

impl DenseFactor {
    /// Combines two dense factors in log-space.
    ///
    /// The resulting scope is the sorted union of both scopes.
    ///
    /// The views are arranged as:
    ///
    /// ```text
    /// lhs: [common, lhs-only]
    /// rhs: [rhs-only, common]
    /// out: [rhs-only, common, lhs-only]
    /// ```
    ///
    /// This makes the Cartesian-product combination a simple nested
    /// iteration over contiguous logical blocks.
    pub(crate) fn combine_dense(&self, other: &DenseFactor) -> DenseFactor {
        let alignment = build_alignment(&self.scope, &other.scope);

        //
        // Validate cardinalities for shared variables
        //

        for (&lhs_axis, &rhs_axis) in alignment.lhs_common.iter().zip(alignment.rhs_common.iter()) {
            assert_eq!(
                self.card()[lhs_axis],
                other.card()[rhs_axis],
                "Cardinality mismatch in factors"
            );
        }

        //
        // Axis Orders
        //

        // lhs: [common, lhs-only]
        let mut lhs_order = Vec::with_capacity(self.scope.len());
        lhs_order.extend_from_slice(&alignment.lhs_common);
        lhs_order.extend_from_slice(&alignment.lhs_only);

        // rhs: [rhs-only, common]
        let mut rhs_order = Vec::with_capacity(other.scope.len());
        rhs_order.extend_from_slice(&alignment.rhs_only);
        rhs_order.extend_from_slice(&alignment.rhs_common);

        // out: [rhs-only, common, lhs-only]
        let mut out_order = Vec::with_capacity(alignment.union_vars.len());

        out_order.extend(
            alignment
                .rhs_only
                .iter()
                .map(|&axis| alignment.rhs_to_out[axis]),
        );

        out_order.extend(
            alignment
                .lhs_common
                .iter()
                .map(|&axis| alignment.lhs_to_out[axis]),
        );

        out_order.extend(
            alignment
                .lhs_only
                .iter()
                .map(|&axis| alignment.lhs_to_out[axis]),
        );

        //
        // Block Sizes
        //

        let common_size = alignment
            .lhs_common
            .iter()
            .map(|&axis| self.card()[axis])
            .product::<usize>();

        let lhs_only_size = alignment
            .lhs_only
            .iter()
            .map(|&axis| self.card()[axis])
            .product::<usize>();

        let rhs_only_size = alignment
            .rhs_only
            .iter()
            .map(|&axis| other.card()[axis])
            .product::<usize>();

        //
        // Output
        //

        let out_card = alignment.output_card(self.card(), other.card());
        let mut out_data = ArrayD::<f64>::zeros(IxDyn(&out_card));

        let lhs_view = self.data.view().permuted_axes(lhs_order);
        let rhs_view = other.data.view().permuted_axes(rhs_order);
        let mut out_view = out_data.view_mut().permuted_axes(out_order);

        //
        // Kernel
        //

        let mut rhs_iter = rhs_view.iter();
        let mut out_iter = out_view.iter_mut();

        for _ in 0..rhs_only_size {
            let mut lhs_iter = lhs_view.iter();

            for _ in 0..common_size {
                let rhs = *rhs_iter
                    .next()
                    .expect("rhs iterator length must match shape");

                for _ in 0..lhs_only_size {
                    let lhs = *lhs_iter
                        .next()
                        .expect("lhs iterator length must match shape");

                    *out_iter
                        .next()
                        .expect("out iterator length must match shape") = lhs + rhs;
                }
            }
        }

        debug_assert!(rhs_iter.next().is_none());
        debug_assert!(out_iter.next().is_none());

        DenseFactor::new(alignment.union_vars, out_data)
    }

    /// Combines a dense factor with a unary factor.
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
    pub(crate) fn combine_unary(self, unary: &UnaryFactor) -> DenseFactor {
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

        if alignment.rhs_common.len() == 1 {
            let dense_common_axis = alignment.lhs_common[0];

            assert_eq!(
                self.card()[dense_common_axis],
                unary.card()[UNARY_AXIS],
                "Cardinality mismatch in factors"
            );

            // Dense: [common, dense-only]
            let mut lhs_order = Vec::with_capacity(self.scope.len());
            lhs_order.extend_from_slice(&alignment.lhs_common);
            lhs_order.extend_from_slice(&alignment.lhs_only);

            // Output: [common, dense-only]
            let mut out_order = Vec::with_capacity(alignment.union_vars.len());

            out_order.extend(
                alignment
                    .lhs_common
                    .iter()
                    .map(|&axis| alignment.lhs_to_out[axis]),
            );

            out_order.extend(
                alignment
                    .lhs_only
                    .iter()
                    .map(|&axis| alignment.lhs_to_out[axis]),
            );

            let dense_view = self.data.view().permuted_axes(lhs_order);

            let out_card = alignment.output_card(self.card(), unary.card());
            let mut out_data = ArrayD::<f64>::zeros(IxDyn(&out_card));

            let mut out_view = out_data.view_mut().permuted_axes(out_order);

            let dense_only_size = alignment
                .lhs_only
                .iter()
                .map(|&axis| self.card()[axis])
                .product::<usize>();

            let unary_data = unary.data();

            let mut dense_iter = dense_view.iter();
            let mut out_iter = out_view.iter_mut();

            for &u in unary_data.iter() {
                for _ in 0..dense_only_size {
                    let x = *dense_iter
                        .next()
                        .expect("dense iterator length must match shape");

                    *out_iter
                        .next()
                        .expect("output iterator length must match shape") = x + u;
                }
            }

            debug_assert!(dense_iter.next().is_none());
            debug_assert!(out_iter.next().is_none());

            DenseFactor::new(alignment.union_vars, out_data)
        } else {
            //
            // Case 2: unary variable is disjoint from the dense factor.
            //

            debug_assert_eq!(alignment.rhs_only.len(), 1);

            // Dense: [dense-only...]
            let mut lhs_order = Vec::with_capacity(self.scope.len());
            lhs_order.extend_from_slice(&alignment.lhs_only);

            // Output: [unary-only, dense-only...]
            let mut out_order = Vec::with_capacity(alignment.union_vars.len());

            out_order.extend(
                alignment
                    .rhs_only
                    .iter()
                    .map(|&axis| alignment.rhs_to_out[axis]),
            );

            out_order.extend(
                alignment
                    .lhs_only
                    .iter()
                    .map(|&axis| alignment.lhs_to_out[axis]),
            );

            let dense_view = self.data.view().permuted_axes(lhs_order);

            let out_card = alignment.output_card(self.card(), unary.card());
            let mut out_data = ArrayD::<f64>::zeros(IxDyn(&out_card));

            let unary_data = unary.data();

            let dense_iter = dense_view.iter();
            let mut out_view = out_data.view_mut().permuted_axes(out_order);
            let mut out_iter = out_view.iter_mut();

            for &u in unary_data.iter() {
                for &x in dense_iter.clone() {
                    *out_iter
                        .next()
                        .expect("output iterator length must match shape") = x + u;
                }
            }

            debug_assert!(out_iter.next().is_none());

            DenseFactor::new(alignment.union_vars, out_data)
        }
    }
}

/// Information needed to align two factor scopes.
///
/// The output scope is always the sorted union of the two input scopes.
///
/// The `*_to_out` mappings map an input axis to its corresponding output axis.
struct Alignment {
    /// Sorted union of the input scopes.
    union_vars: Vec<VariableId>,

    /// Axes appearing only in the left-hand factor.
    lhs_only: Vec<usize>,

    /// Axes appearing only in the right-hand factor.
    rhs_only: Vec<usize>,

    /// Axes corresponding to variables shared by both factors.
    lhs_common: Vec<usize>,
    rhs_common: Vec<usize>,

    /// Maps left-hand factor axes to output axes.
    lhs_to_out: Vec<usize>,

    /// Maps right-hand factor axes to output axes.
    rhs_to_out: Vec<usize>,
}

impl Alignment {
    /// Constructs the output cardinalities from the input cardinalities.
    fn output_card(&self, lhs_card: &[usize], rhs_card: &[usize]) -> Vec<usize> {
        let mut card = vec![0usize; self.union_vars.len()];

        for (axis, &out_axis) in self.lhs_to_out.iter().enumerate() {
            if out_axis != usize::MAX {
                card[out_axis] = lhs_card[axis];
            }
        }

        for (axis, &out_axis) in self.rhs_to_out.iter().enumerate() {
            if out_axis != usize::MAX {
                card[out_axis] = rhs_card[axis];
            }
        }

        card
    }
}

/// Builds the alignment between two canonically ordered factor scopes.
///
/// Both input scopes must be sorted in ascending variable order.
fn build_alignment(lhs_scope: &[VariableId], rhs_scope: &[VariableId]) -> Alignment {
    debug_assert!(lhs_scope.windows(2).all(|w| w[0] < w[1]));
    debug_assert!(rhs_scope.windows(2).all(|w| w[0] < w[1]));

    let lhs_len = lhs_scope.len();
    let rhs_len = rhs_scope.len();

    let mut union_vars = Vec::with_capacity(lhs_len + rhs_len);

    let mut lhs_only = Vec::with_capacity(lhs_len);
    let mut rhs_only = Vec::with_capacity(rhs_len);

    let mut lhs_common = Vec::with_capacity(lhs_len.min(rhs_len));
    let mut rhs_common = Vec::with_capacity(lhs_len.min(rhs_len));

    let mut lhs_to_out = vec![usize::MAX; lhs_len];
    let mut rhs_to_out = vec![usize::MAX; rhs_len];

    let mut i = 0;
    let mut j = 0;

    // Merge the two sorted scopes.
    while i < lhs_len && j < rhs_len {
        let out_axis = union_vars.len();

        match lhs_scope[i].cmp(&rhs_scope[j]) {
            std::cmp::Ordering::Equal => {
                lhs_common.push(i);
                rhs_common.push(j);
                union_vars.push(lhs_scope[i]);

                lhs_to_out[i] = out_axis;
                rhs_to_out[j] = out_axis;

                i += 1;
                j += 1;
            }

            std::cmp::Ordering::Less => {
                lhs_only.push(i);
                union_vars.push(lhs_scope[i]);

                lhs_to_out[i] = out_axis;
                i += 1;
            }

            std::cmp::Ordering::Greater => {
                rhs_only.push(j);
                union_vars.push(rhs_scope[j]);

                rhs_to_out[j] = out_axis;
                j += 1;
            }
        }
    }

    // Append any remaining left-hand variables.
    while i < lhs_len {
        lhs_to_out[i] = union_vars.len();
        lhs_only.push(i);
        union_vars.push(lhs_scope[i]);
        i += 1;
    }

    // Append any remaining right-hand variables.
    while j < rhs_len {
        rhs_to_out[j] = union_vars.len();
        rhs_only.push(j);
        union_vars.push(rhs_scope[j]);
        j += 1;
    }

    Alignment {
        union_vars,
        lhs_only,
        rhs_only,
        lhs_common,
        rhs_common,
        lhs_to_out,
        rhs_to_out,
    }
}
