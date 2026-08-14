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
