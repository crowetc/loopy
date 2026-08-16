use super::{Factor, DenseFactor, ScalarFactor, UnaryFactor};
use super::log_utils::lse_two_pass;
use super::utils::dense_into_unary;

#[derive(Clone, Debug)]
pub enum FactorKind {
    Dense(DenseFactor),
    Scalar(ScalarFactor),
    Unary(UnaryFactor),
}

impl FactorKind {
    pub fn scope(&self) -> &[usize] {
        match self {
            FactorKind::Dense(d) => d.scope(),
            FactorKind::Scalar(s) => s.scope(),
            FactorKind::Unary(u) => u.scope(),
        }
    }

    // Consuming marginalize: takes ownership and returns a new FactorKind
    pub fn marginalize(self, vars: &[usize]) -> FactorKind {
        match self {
            FactorKind::Dense(d) => {
                let reduced = d.marginalize_kernel(vars); // returns DenseFactor
                match reduced.scope().len() {
                    0 => FactorKind::Scalar(ScalarFactor::new(reduced.data()[[]])),
                    1 => FactorKind::Unary(dense_into_unary(reduced)),
                    _ => FactorKind::Dense(reduced),
                }
            }
            FactorKind::Scalar(s) => {
                FactorKind::Scalar(s)
            }
            FactorKind::Unary(u) => {
                if vars.contains(&u.scope()[0]) {
                    let total = lse_two_pass(&u.data());
                    FactorKind::Unary(UnaryFactor::new(u.scope()[0], vec![total]))
                } else {
                    FactorKind::Unary(u)
                }
            }
        }
    }
}
