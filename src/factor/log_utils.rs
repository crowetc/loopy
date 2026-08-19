/// Numerically stable log-sum-exp helpers
#[inline]
pub(crate) fn lse_update(x: f64, cur_max: &mut f64, cur_sum: &mut f64) {
    if x.is_infinite() && x.is_sign_negative() {
        return;
    }
    if *cur_max == f64::NEG_INFINITY {
        *cur_max = x;
        *cur_sum = 1.0;
    } else if x > *cur_max {
        // new max: rescale sum
        let scale = (*cur_max - x).exp();
        *cur_sum = *cur_sum * scale + 1.0;
        *cur_max = x;
    } else {
        *cur_sum += (x - *cur_max).exp();
    }
}

#[inline]
pub(crate) fn lse_finalize(cur_max: f64, cur_sum: f64) -> f64 {
    if cur_max == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        cur_max + cur_sum.ln()
    }
}

/// Two-pass log-sum-exp over a slice of log-values.
/// Returns log(sum(exp(slice))).
#[inline]
pub(crate) fn lse_two_pass(slice: &[f64]) -> f64 {
    // 1) find max
    let mut max = f64::NEG_INFINITY;
    for &x in slice.iter() {
        if x > max {
            max = x;
        }
    }
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    // 2) sum exp(x - max)
    let mut sum = 0.0;
    for &x in slice.iter() {
        sum += (x - max).exp();
    }
    max + sum.ln()
}
