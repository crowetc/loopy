use ndarray::IxDyn;

use crate::factor::{DenseFactor, UnaryFactor};

/// Try to convert an owned DenseFactor with a single axis into a UnaryFactor
/// without copying. If zero-copy is not possible, fall back to a single copy.
pub(crate) fn dense_into_unary(d: DenseFactor) -> UnaryFactor {
    // consume the DenseFactor to get owned parts
    let (scope, data) = d.into_parts();
    assert_eq!(scope.len(), 1);
    let var = scope[0];
    let card = data.shape()[0];

    // Try zero-copy: attempt to reshape a clone of the owned array into 1-D.
    // If that succeeds and the backing vec has offset == Some(0), reuse it.
    if let Ok(arr1d) = data.clone().into_shape_with_order(IxDyn(&[card])) {
        let (vec, offset_opt) = arr1d.into_raw_vec_and_offset();
        if offset_opt == Some(0) {
            return UnaryFactor::new(var, vec);
        }
        // otherwise fall through to copy from `data`
    }

    // Fallback: copy from the owned `data`
    let vec = data.iter().cloned().collect::<Vec<_>>();
    UnaryFactor::new(var, vec)
}


// /// Reduce an owned DenseFactor to a scalar (log-space) without allocating.
// /// Returns the scalar log-value.
// pub(crate) fn dense_into_scalar_if_possible(d: DenseFactor) -> f64 {
//     let mut cur_max = f64::NEG_INFINITY;
//     let mut cur_sum = 0.0;
//     for &x in d.data.iter() {
//         crate::factor::log_utils::lse_update(x, &mut cur_max, &mut cur_sum);
//     }
//     crate::factor::log_utils::lse_finalize(cur_max, cur_sum)
// }

// /// Add a unary factor into a dense factor in-place (consumes both).
// /// If the unary variable is not present in the dense scope, produce a Dense
// /// factor that is the outer-sum (i.e., broadcast unary across new axis).
// ///
// /// Returns a FactorKind (Dense or Unary) as appropriate.
// pub(crate) fn add_unary_into_dense_inplace(mut dense: DenseFactor, unary: UnaryFactor) -> FactorKind {
//     // find axis index of unary.var in dense.scope
//     if let Some(axis_idx) = dense.scope.iter().position(|&v| v == unary.var()) {
//         // axis exists: add unary values into each lane along that axis
//         let card = dense.data.shape()[axis_idx];
//         assert_eq!(card, unary.data().len(), "cardinalities must match");

//         // iterate mutable over axis and add unary values
//         // axis_iter_mut yields views where the axis is removed; we need to add scalar to each lane
//         // We iterate over index i and add unary.data[i] to the corresponding lane
//         for (i, mut lane) in dense.data.axis_iter_mut(Axis(axis_idx)).enumerate() {
//             let add_val = unary.data()[i];
//             lane += add_val; // ndarray supports adding scalar to view
//         }

//         FactorKind::Dense(dense)
//     } else {
//         // axis not present: produce a Dense factor with the unary axis appended
//         // New scope: dense.scope followed by unary.var
//         let mut new_scope = dense.scope.clone();
//         new_scope.push(unary.var());

//         // new shape is old_shape + [unary_card]
//         let mut new_shape = dense.data.shape().to_vec();
//         new_shape.push(unary.data().len());

//         // allocate result and fill with dense_value + unary_value
//         let kept_size = dense.data.len();
//         let unary_card = unary.data().len();
//         let mut out_vec = Vec::with_capacity(kept_size * unary_card);

//         // iterate over dense elements and for each append unary values
//         // This preserves the dense ordering and appends the unary axis as fastest-changing
//         // (i.e., shape [..., unary_card])
//         // Use dense.data.iter() which iterates in logical order
//         for &dval in dense.data.iter() {
//             for &uval in unary.data().iter() {
//                 out_vec.push(dval + uval);
//             }
//         }

//         let out = ArrayD::from_shape_vec(IxDyn(&new_shape), out_vec)
//             .expect("shape and data length must match");

//         FactorKind::Dense(DenseFactor::new(new_scope, out))
//     }
// }

// /// Multiply two dense factors (consume both) and return a Dense FactorKind.
// /// This builds the union scope (order: d1.scope then any vars from d2 not in d1),
// /// computes the union shape, and fills the result by mapping linear indices to
// /// coordinates and summing corresponding entries from d1 and d2.
// ///
// /// This implementation favors correctness and clarity; for very large tensors
// /// you may want to add specialized fast paths or use ndarray broadcasting tricks.
// pub(crate) fn multiply_dense_dense(d1: DenseFactor, d2: DenseFactor) -> FactorKind {
//     // Build union scope: d1.scope followed by vars in d2.scope not in d1
//     let mut union_scope = d1.scope.clone();
//     for &v in d2.scope.iter() {
//         if !union_scope.contains(&v) {
//             union_scope.push(v);
//         }
//     }

//     // Build union shape
//     let mut union_shape = Vec::with_capacity(union_scope.len());
//     for &v in union_scope.iter() {
//         // find cardinality in d1 or d2
//         if let Some(pos) = d1.scope.iter().position(|&x| x == v) {
//             union_shape.push(d1.data.shape()[pos]);
//         } else if let Some(pos) = d2.scope.iter().position(|&x| x == v) {
//             union_shape.push(d2.data.shape()[pos]);
//         } else {
//             unreachable!("variable must appear in at least one factor");
//         }
//     }

//     // Precompute strides for union shape (row-major / C-order)
//     let union_strides = compute_strides(&union_shape);

//     // Prepare raw data vectors for d1 and d2 (linearized in their native order).
//     // Try to take ownership of raw vecs without copying; otherwise fall back to collect.
//     let (d1_vec, d1_shape, d1_strides) = {
//         let shape = d1.data.shape().to_vec();
//         match d1.data.into_shape(IxDyn(&shape)) {
//             Ok(arr) => {
//                 // arr is owned and contiguous in this shape
//                 let vec = arr.into_raw_vec();
//                 let strides = compute_strides(&shape);
//                 (vec, shape, strides)
//             }
//             Err(arr) => {
//                 let vec = arr.iter().cloned().collect::<Vec<_>>();
//                 let strides = compute_strides(&shape);
//                 (vec, shape, strides)
//             }
//         }
//     };

//     let (d2_vec, d2_shape, d2_strides) = {
//         let shape = d2.data.shape().to_vec();
//         match d2.data.into_shape(IxDyn(&shape)) {
//             Ok(arr) => {
//                 let vec = arr.into_raw_vec();
//                 let strides = compute_strides(&shape);
//                 (vec, shape, strides)
//             }
//             Err(arr) => {
//                 let vec = arr.iter().cloned().collect::<Vec<_>>();
//                 let strides = compute_strides(&shape);
//                 (vec, shape, strides)
//             }
//         }
//     };

//     // Helper: map a union multi-index to an index in d1/d2 linear storage
//     let map_to_factor_index = |coords: &[usize], factor_scope: &[usize], factor_strides: &[usize]| -> usize {
//         // coords are in union order; pick coordinates for axes present in factor_scope
//         let mut idx = 0usize;
//         for (axis_pos, &var) in factor_scope.iter().enumerate() {
//             // find position of var in union_scope
//             let union_pos = union_scope.iter().position(|&x| x == var).unwrap();
//             let c = coords[union_pos];
//             idx += c * factor_strides[axis_pos];
//         }
//         idx
//     };

//     // Allocate output vector and fill
//     let total_len = union_shape.iter().product::<usize>();
//     let mut out_vec = Vec::with_capacity(total_len);

//     // Temporary coords vector
//     let mut coords = vec![0usize; union_shape.len()];

//     for linear in 0..total_len {
//         // compute multi-index coords from linear index
//         let mut rem = linear;
//         for (i, &stride) in union_strides.iter().enumerate() {
//             coords[i] = rem / stride;
//             rem %= stride;
//         }

//         // map to d1 and d2 indices
//         let idx1 = map_to_factor_index(&coords, &d1.scope, &d1_strides);
//         let idx2 = map_to_factor_index(&coords, &d2.scope, &d2_strides);

//         let v1 = d1_vec[idx1];
//         let v2 = d2_vec[idx2];
//         out_vec.push(v1 + v2);
//     }

//     let out = ArrayD::from_shape_vec(IxDyn(&union_shape), out_vec)
//         .expect("shape and data length must match");

//     FactorKind::Dense(DenseFactor::new(union_scope, out))
// }

// /// Compute row-major strides for a shape (i.e., product of subsequent dims).
// fn compute_strides(shape: &[usize]) -> Vec<usize> {
//     let n = shape.len();
//     let mut strides = vec![0usize; n];
//     if n == 0 {
//         return strides;
//     }
//     let mut acc = 1usize;
//     for i in (0..n).rev() {
//         strides[i] = acc;
//         acc = acc.saturating_mul(shape[i]);
//     }
//     strides
// }
