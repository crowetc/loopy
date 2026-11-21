//! Dense factor reduction kernels.
//!
//! This module implements reduction over dense discrete factors. Retained and
//! reduced axes are rearranged into contiguous logical blocks before applying
//! the reduction operation.

use ndarray::{ArrayD, IxDyn};

use crate::factor::log_utils::{lse_finalize, lse_update};
use crate::factor::{DenseFactor, FactorKind, ScalarFactor, UnaryFactor};
use crate::variable::VariableId;

impl DenseFactor {
    /// Marginalization with sum.
    ///
    /// Retained axes are moved before eliminated axes so that each assignment of
    /// the retained variables corresponds to one logical reduction block.
    pub(super) fn reduce_sum_kernel(self, vars: &[VariableId]) -> DenseFactor {
        self.reduce_kernel(vars, lse_update, lse_finalize)
    }

    /// Marginalize with max.
    ///
    /// Retained axes are moved before eliminated axes so that each assignment of
    /// the retained variables corresponds to one logical reduction block.
    pub(super) fn reduce_max_kernel(self, vars: &[VariableId]) -> DenseFactor {
        self.reduce_kernel(vars, |value, max, _| *max = max.max(value), |max, _| max)
    }

    pub(super) fn reduce_sum_into(&self, vars: &[VariableId], output: &mut FactorKind) {
        self.reduce_into_kernel(vars, output, lse_update, lse_finalize);
    }

    pub(super) fn reduce_max_into(&self, vars: &[VariableId], output: &mut FactorKind) {
        self.reduce_into_kernel(
            vars,
            output,
            |value, max, _| *max = max.max(value),
            |max, _| max,
        );
    }

    fn reduce_kernel(
        self,
        vars: &[VariableId],
        update: impl Fn(f64, &mut f64, &mut f64) + Copy,
        finalize: impl Fn(f64, f64) -> f64 + Copy,
    ) -> DenseFactor {
        let Some(reduction) = build_reduction(&self.scope, self.data.shape(), vars) else {
            return self;
        };

        let permuted = self.data.view().permuted_axes(reduction.order);

        let mut values = vec![0.0; reduction.kept_size];

        write_reduction(
            permuted.iter(),
            reduction.elim_size,
            values.iter_mut(),
            update,
            finalize,
        );

        let data = ArrayD::from_shape_vec(IxDyn(&reduction.kept_shape), values)
            .expect("reduction output shape mismatch");

        DenseFactor::new(reduction.new_scope, data)
    }

    fn reduce_into_kernel(
        &self,
        vars: &[VariableId],
        output: &mut FactorKind,
        update: impl Fn(f64, &mut f64, &mut f64) + Copy,
        finalize: impl Fn(f64, f64) -> f64 + Copy,
    ) {
        let Some(reduction) = build_reduction(&self.scope, self.data.shape(), vars) else {
            match output {
                FactorKind::Dense(result)
                    if result.scope == self.scope && result.data.shape() == self.data.shape() =>
                {
                    result.data.assign(&self.data);
                }
                _ => *output = FactorKind::Dense(self.clone()),
            }
            return;
        };

        let permuted = self.data.view().permuted_axes(reduction.order);

        match reduction.new_scope.len() {
            0 => {
                let mut value = [0.0];

                write_reduction(
                    permuted.iter(),
                    reduction.elim_size,
                    value.iter_mut(),
                    update,
                    finalize,
                );

                *output = FactorKind::Scalar(ScalarFactor::new(value[0]));
            }

            1 => {
                let var = reduction.new_scope[0];

                if let FactorKind::Unary(result) = output
                    && result.var() == var
                    && result.data().len() == reduction.kept_size
                {
                    write_reduction(
                        permuted.iter(),
                        reduction.elim_size,
                        result.data_mut().iter_mut(),
                        update,
                        finalize,
                    );
                    return;
                }

                let mut values = vec![0.0; reduction.kept_size];

                write_reduction(
                    permuted.iter(),
                    reduction.elim_size,
                    values.iter_mut(),
                    update,
                    finalize,
                );

                *output = FactorKind::Unary(UnaryFactor::new(var, values));
            }

            _ => {
                if let FactorKind::Dense(result) = output
                    && result.scope == reduction.new_scope
                    && result.data.shape() == reduction.kept_shape
                {
                    write_reduction(
                        permuted.iter(),
                        reduction.elim_size,
                        result.data.iter_mut(),
                        update,
                        finalize,
                    );
                    return;
                }

                let mut data = ArrayD::zeros(IxDyn(&reduction.kept_shape));

                write_reduction(
                    permuted.iter(),
                    reduction.elim_size,
                    data.iter_mut(),
                    update,
                    finalize,
                );

                *output = FactorKind::Dense(DenseFactor::new(reduction.new_scope, data));
            }
        }
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
/// The factor scope must be in ascending variable order. Variables to
/// eliminate may be supplied in any order.
///
/// Returns `None` when none of the requested variables appear in the
/// factor scope.
fn build_reduction(
    scope: &[VariableId],
    shape: &[usize],
    vars: &[VariableId],
) -> Option<Reduction> {
    debug_assert!(scope.windows(2).all(|w| w[0] < w[1]));
    debug_assert_eq!(scope.len(), shape.len());

    let mut idx_marg = Vec::with_capacity(vars.len());
    let mut idx_keep = Vec::with_capacity(scope.len());
    let mut new_scope = Vec::with_capacity(scope.len());

    for (axis, &variable) in scope.iter().enumerate() {
        if vars.contains(&variable) {
            idx_marg.push(axis);
        } else {
            idx_keep.push(axis);
            new_scope.push(variable);
        }
    }

    if idx_marg.is_empty() {
        return None;
    }

    // Permutation: retained axes first, eliminated axes second.
    let mut order = Vec::with_capacity(scope.len());
    order.extend_from_slice(&idx_keep);
    order.extend_from_slice(&idx_marg);

    let kept_shape: Vec<usize> = idx_keep.iter().map(|&axis| shape[axis]).collect();

    let kept_size = kept_shape.iter().product();
    let elim_size = idx_marg.iter().map(|&axis| shape[axis]).product();

    Some(Reduction {
        order,
        new_scope,
        kept_shape,
        kept_size,
        elim_size,
    })
}

fn write_reduction<'a>(
    input: impl Iterator<Item = &'a f64>,
    elim_size: usize,
    output: impl Iterator<Item = &'a mut f64>,
    mut update: impl FnMut(f64, &mut f64, &mut f64),
    mut finalize: impl FnMut(f64, f64) -> f64,
) {
    let mut input = input;

    for dst in output {
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;

        for _ in 0..elim_size {
            let value = *input.next().expect("reduction input length mismatch");
            update(value, &mut max, &mut sum);
        }

        *dst = finalize(max, sum);
    }
}
