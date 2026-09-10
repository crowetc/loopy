//! Dense factor reduction kernels.
//!
//! This module implements reduction over dense discrete factors. Retained and
//! reduced axes are rearranged into contiguous logical blocks before applying
//! the reduction operation.

use ndarray::{ArrayD, IxDyn};

use super::DenseFactor;
use crate::factor::log_utils::{lse_finalize, lse_update};
use crate::variable::VariableId;

impl DenseFactor {
    /// Marginalization with sum.
    ///
    /// Retained axes are moved before eliminated axes so that each assignment of
    /// the retained variables corresponds to one logical reduction block.
    pub(super) fn reduce_sum_kernel(self, vars: &[VariableId]) -> DenseFactor {
        let reduction = match build_reduction(&self.scope, self.data.shape(), vars) {
            Some(reduction) => reduction,
            None => return self,
        };

        let permuted = self.data.view().permuted_axes(reduction.order);

        let mut out_vec = Vec::with_capacity(reduction.kept_size);
        let mut it = permuted.iter();

        for _ in 0..reduction.kept_size {
            let mut cur_max = f64::NEG_INFINITY;
            let mut cur_sum = 0.0;

            for _ in 0..reduction.elim_size {
                let &x = it.next().expect("iterator length must match expected size");

                lse_update(x, &mut cur_max, &mut cur_sum);
            }

            out_vec.push(lse_finalize(cur_max, cur_sum));
        }

        debug_assert_eq!(out_vec.len(), reduction.kept_size);

        let out = ArrayD::from_shape_vec(IxDyn(&reduction.kept_shape), out_vec)
            .expect("shape and data length must match");

        DenseFactor::new(reduction.new_scope, out)
    }

    /// Marginalize with max.
    ///
    /// Retained axes are moved before eliminated axes so that each assignment of
    /// the retained variables corresponds to one logical reduction block.
    pub(super) fn reduce_max_kernel(self, vars: &[VariableId]) -> DenseFactor {
        let reduction = match build_reduction(&self.scope, self.data.shape(), vars) {
            Some(reduction) => reduction,
            None => return self,
        };

        let permuted = self.data.view().permuted_axes(reduction.order);

        let mut out_vec = Vec::with_capacity(reduction.kept_size);
        let mut it = permuted.iter();

        for _ in 0..reduction.kept_size {
            let mut cur_max = f64::NEG_INFINITY;

            for _ in 0..reduction.elim_size {
                let &x = it.next().expect("iterator length must match expected size");

                cur_max = cur_max.max(x);
            }

            out_vec.push(cur_max);
        }

        debug_assert_eq!(out_vec.len(), reduction.kept_size);

        let out = ArrayD::from_shape_vec(IxDyn(&reduction.kept_shape), out_vec)
            .expect("shape and data length must match");

        DenseFactor::new(reduction.new_scope, out)
    }
}

/// Information needed to reduce a factor over a set of variables.
///
/// The reduction axes are rearranged to the end of the factor so that
/// assignments for the retained variables form contiguous logical blocks.
struct Reduction {
    /// Axis permutation: kept axes first, reduced axes second.
    order: Vec<usize>,

    /// Scope of the resulting factor.
    new_scope: Vec<VariableId>,

    /// Shape of the retained dimensions.
    kept_shape: Vec<usize>,

    /// Number of output elements.
    kept_size: usize,

    /// Number of input elements contributing to each output element.
    elim_size: usize,
}

/// Builds the reduction plan for the specified variables.
///
/// Returns `None` when none of the requested variables appear in the factor
/// scope.
fn build_reduction(
    scope: &[VariableId],
    shape: &[usize],
    vars: &[VariableId],
) -> Option<Reduction> {
    //
    // Sort scope by variable ID while retaining original axis indices.
    //

    // Sort by variable ID while retaining each variable's original axis.
    let mut scope_sorted: Vec<(VariableId, usize)> = scope
        .iter()
        .enumerate()
        .map(|(axis, &var)| (var, axis))
        .collect();

    scope_sorted.sort_unstable_by_key(|&(var, _)| var);

    //
    // Sort variables to eliminate.
    //

    let mut vars_sorted = vars.to_vec();
    vars_sorted.sort_unstable();

    let mut idx_marg = Vec::with_capacity(vars_sorted.len());
    let mut idx_keep = Vec::with_capacity(scope.len());
    let mut new_scope = Vec::with_capacity(scope.len());

    let mut j = 0;

    for &(v, axis) in &scope_sorted {
        while j < vars_sorted.len() && vars_sorted[j] < v {
            j += 1;
        }

        let eliminate = j < vars_sorted.len() && vars_sorted[j] == v;

        if eliminate {
            idx_marg.push(axis);
        } else {
            idx_keep.push(axis);
            new_scope.push(v);
        }
    }

    if idx_marg.is_empty() {
        return None;
    }

    //
    // Permutation: kept axes first, eliminated axes second.
    //

    let mut order = Vec::with_capacity(scope.len());
    order.extend_from_slice(&idx_keep);
    order.extend_from_slice(&idx_marg);

    //
    // Output shape and block sizes.
    //

    let kept_shape: Vec<usize> = idx_keep.iter().map(|&axis| shape[axis]).collect();

    let kept_size = kept_shape.iter().product::<usize>();

    let elim_size = idx_marg.iter().map(|&axis| shape[axis]).product();

    Some(Reduction {
        order,
        new_scope,
        kept_shape,
        kept_size,
        elim_size,
    })
}
