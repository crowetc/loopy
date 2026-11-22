//! Dense factor combination kernels.
//!
//! This module implements combination between dense factors and other
//! supported discrete factor representations. Factor scopes are aligned by
//! variable identifier, and values are combined by addition in log-space.

use ndarray::{ArrayD, IxDyn};

use super::DenseFactor;
use crate::factor::{DiscreteFactor, Factor, FactorKind, ScalarFactor, UnaryFactor};
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

    /// Combines two dense factors, writing the result into `output`.
    ///
    /// Reuses the destination's storage when both inputs and the destination
    /// have identical scopes and cardinalities. Otherwise, constructs a new
    /// dense factor and replaces the destination.
    pub(crate) fn combine_dense_into(&self, other: &DenseFactor, output: &mut FactorKind) {
        if self.scope == other.scope {
            assert_eq!(self.card(), other.card(), "Cardinality mismatch in factors");

            if let FactorKind::Dense(result) = output
                && result.scope == self.scope
                && result.card() == self.card()
            {
                for ((out, lhs), rhs) in result
                    .data
                    .iter_mut()
                    .zip(self.data.iter())
                    .zip(other.data.iter())
                {
                    *out = lhs + rhs;
                }

                return;
            }
        }

        *output = FactorKind::Dense(self.combine_dense(other));
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
    pub(crate) fn combine_unary(mut self, unary: &UnaryFactor) -> DenseFactor {
        debug_assert_eq!(
            unary.scope().len(),
            1,
            "UnaryFactor must have exactly one variable"
        );

        let alignment = build_alignment(&self.scope, unary.scope());

        const UNARY_AXIS: usize = 0;

        if alignment.rhs_common.len() == 1 {
            let dense_common_axis = alignment.lhs_common[0];

            assert_eq!(
                self.card()[dense_common_axis],
                unary.card()[UNARY_AXIS],
                "Cardinality mismatch in factors"
            );

            let card = unary.card()[UNARY_AXIS];
            let unary_data = unary.data();

            // Fast path: contiguous physical storage, regardless of
            // whether the array has standard or nonstandard strides.
            //
            // Singleton dimensions are excluded because their strides
            // need not correspond to physical block sizes.
            if self.card().iter().all(|&card| card > 1) {
                let inner = self.data.strides()[dense_common_axis] as usize;

                if let Some(data) = self.data.as_slice_memory_order_mut() {
                    if inner == 1 {
                        // Unary axis is physically innermost.
                        for row in data.chunks_exact_mut(card) {
                            for (value, &u) in row.iter_mut().zip(unary_data.iter()) {
                                *value += u;
                            }
                        }
                    } else {
                        // Unary axis has a larger physical stride.
                        let block_size = card * inner;

                        for block in data.chunks_exact_mut(block_size) {
                            for (state, &u) in unary_data.iter().enumerate() {
                                let start = state * inner;
                                let end = start + inner;

                                for value in &mut block[start..end] {
                                    *value += u;
                                }
                            }
                        }
                    }

                    return self;
                }
            }

            // General path: arbitrary memory layouts.
            for mut lane in self.data.lanes_mut(ndarray::Axis(dense_common_axis)) {
                for (value, &u) in lane.iter_mut().zip(unary_data.iter()) {
                    *value += u;
                }
            }

            self
        } else {
            debug_assert_eq!(alignment.rhs_only.len(), 1);

            let mut lhs_order = Vec::with_capacity(self.scope.len());
            lhs_order.extend_from_slice(&alignment.lhs_only);

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

    /// Combines a dense factor with a unary factor, writing into `output`.
    ///
    /// Reuses compatible destination storage when the unary variable belongs
    /// to the dense factor's scope. The destination's physical layout may be
    /// adjusted to match the source without reallocating.
    ///
    /// Otherwise, replaces the destination with a newly computed factor.
    pub(crate) fn combine_unary_into(&self, unary: &UnaryFactor, output: &mut FactorKind) {
        if let Ok(axis) = self.scope.binary_search(&unary.scope()[0]) {
            assert_eq!(
                self.card()[axis],
                unary.card()[0],
                "Cardinality mismatch in factors"
            );

            if let FactorKind::Dense(result) = output
                && result.scope == self.scope
                && result.card() == self.card()
            {
                // Reinterpret a standard-layout destination to match the
                // source's physical memory order.
                //
                // The destination will be completely overwritten, so its
                // previous physical layout does not need to be preserved.
                if result.data.is_standard_layout()
                    && !self.data.is_standard_layout()
                    && self.data.as_slice_memory_order().is_some()
                    && self.data.strides().iter().all(|&stride| stride > 0)
                    && self.card().iter().all(|&card| card > 1)
                {
                    use ndarray::ShapeBuilder;

                    let shape = self.data.shape().to_vec();
                    let strides = self
                        .data
                        .strides()
                        .iter()
                        .map(|&stride| stride as usize)
                        .collect::<Vec<_>>();

                    let old_data = std::mem::take(&mut result.data);
                    let (values, offset) = old_data.into_raw_vec_and_offset();

                    debug_assert_eq!(offset, Some(0));

                    result.data =
                        ArrayD::from_shape_vec(IxDyn(&shape).strides(IxDyn(&strides)), values)
                            .expect("source strides must describe contiguous storage");
                }

                // Fast path: matching contiguous physical layouts.
                //
                // This handles both standard row-major arrays and
                // nonstandard contiguous arrays.
                if self.data.strides() == result.data.strides()
                    && self.card().iter().all(|&card| card > 1)
                    && let (Some(lhs), Some(out)) = (
                        self.data.as_slice_memory_order(),
                        result.data.as_slice_memory_order_mut(),
                    )
                {
                    let card = unary.card()[0];
                    let inner = self.data.strides()[axis] as usize;
                    let unary_data = unary.data();

                    if inner == 1 {
                        // Unary axis is physically innermost.
                        for (out_row, lhs_row) in
                            out.chunks_exact_mut(card).zip(lhs.chunks_exact(card))
                        {
                            for ((dst, &src), &u) in out_row
                                .iter_mut()
                                .zip(lhs_row.iter())
                                .zip(unary_data.iter())
                            {
                                *dst = src + u;
                            }
                        }
                    } else {
                        // Unary axis has a larger physical stride.
                        let block_size = card * inner;

                        for (lhs_block, out_block) in lhs
                            .chunks_exact(block_size)
                            .zip(out.chunks_exact_mut(block_size))
                        {
                            for (state, &u) in unary_data.iter().enumerate() {
                                let start = state * inner;
                                let end = start + inner;

                                for (&src, dst) in lhs_block[start..end]
                                    .iter()
                                    .zip(out_block[start..end].iter_mut())
                                {
                                    *dst = src + u;
                                }
                            }
                        }
                    }

                    return;
                }

                // General path: arbitrary or mismatched layouts.
                let axis = ndarray::Axis(axis);

                for (mut out_lane, lhs_lane) in result
                    .data
                    .lanes_mut(axis)
                    .into_iter()
                    .zip(self.data.lanes(axis))
                {
                    for ((out, &lhs), &rhs) in out_lane
                        .iter_mut()
                        .zip(lhs_lane.iter())
                        .zip(unary.data().iter())
                    {
                        *out = lhs + rhs;
                    }
                }

                return;
            }
        }

        // Fallback: incompatible destination or disjoint scopes.
        *output = FactorKind::Dense(self.clone().combine_unary(unary));
    }

    /// Combines a dense factor with a scalar factor, writing into `output`.
    ///
    /// Reuses the destination's storage when its scope and shape match
    /// the dense factor. Otherwise, replaces the destination.
    pub(crate) fn combine_scalar_into(&self, scalar: &ScalarFactor, output: &mut FactorKind) {
        if let FactorKind::Dense(result) = output
            && result.scope == self.scope
            && result.card() == self.card()
        {
            for (out, lhs) in result.data.iter_mut().zip(self.data.iter()) {
                *out = lhs + scalar.value();
            }

            return;
        }

        *output = FactorKind::Dense(DenseFactor::new(
            self.scope.clone(),
            self.data.mapv(|x| x + scalar.value()),
        ));
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::VariableId;

    #[test]
    fn combine_unary_into_reuses_destination_with_nonstandard_layout() {
        use ndarray::{ArrayD, IxDyn, ShapeBuilder};

        let scope = vec![VariableId::new(0), VariableId::new(1)];

        // Fortran-order strides [1, 3], rather than row-major [4, 1].
        let data = ArrayD::from_shape_vec(IxDyn(&[3, 4]).f(), (0..12).map(|x| x as f64).collect())
            .unwrap();

        let dense = DenseFactor::new(scope.clone(), data);

        let unary = UnaryFactor::new(scope[1], vec![10.0, 20.0, 30.0, 40.0]);

        let mut output = FactorKind::Dense(DenseFactor::new(scope, ArrayD::zeros(IxDyn(&[3, 4]))));

        let original_ptr = match &output {
            FactorKind::Dense(factor) => factor.data.as_ptr(),
            _ => unreachable!(),
        };

        dense.combine_unary_into(&unary, &mut output);

        let FactorKind::Dense(result) = &output else {
            panic!("expected dense output");
        };

        assert_eq!(result.data.strides(), dense.data.strides());
        assert_eq!(result.data.as_ptr(), original_ptr);

        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(
                    result.data[IxDyn(&[i, j])],
                    dense.data[IxDyn(&[i, j])] + unary.data()[j],
                );
            }
        }

        // The second call should reuse the already-matching layout.
        dense.combine_unary_into(&unary, &mut output);

        let FactorKind::Dense(result) = &output else {
            panic!("expected dense output");
        };

        assert_eq!(result.data.as_ptr(), original_ptr);

        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(
                    result.data[IxDyn(&[i, j])],
                    dense.data[IxDyn(&[i, j])] + unary.data()[j],
                );
            }
        }
    }

    #[test]
    fn combine_unary_handles_singleton_dimensions() {
        use ndarray::{ArrayD, IxDyn};

        let scope = vec![VariableId::new(0), VariableId::new(1)];

        let dense = DenseFactor::new(
            scope.clone(),
            ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
        );

        let unary = UnaryFactor::new(scope[1], vec![10.0, 20.0, 30.0]);

        let expected = [11.0, 22.0, 33.0];

        // Consuming combination.
        let combined = dense.clone().combine_unary(&unary);

        for (actual, expected) in combined.data.iter().zip(expected) {
            assert_eq!(*actual, expected);
        }

        // Combination into an existing destination.
        let mut output = FactorKind::Dense(DenseFactor::new(scope, ArrayD::zeros(IxDyn(&[1, 3]))));

        dense.combine_unary_into(&unary, &mut output);

        let FactorKind::Dense(result) = &output else {
            panic!("expected dense output");
        };

        for (actual, expected) in result.data.iter().zip(expected) {
            assert_eq!(*actual, expected);
        }
    }
}
