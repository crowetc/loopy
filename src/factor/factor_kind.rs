use super::{Factor, DenseFactor, UnaryFactor};
use super::log_utils::lse_two_pass;
use super::utils::dense_into_unary;

#[derive(Clone, Debug)]
pub enum FactorKind {
    Dense(DenseFactor),
    Unary(UnaryFactor),
}

impl FactorKind {
    pub fn scope(&self) -> &[usize] {
        match self {
            FactorKind::Dense(d) => d.scope(),
            FactorKind::Unary(u) => u.scope(),
        }
    }

    // Consuming marginalize: takes ownership and returns a new FactorKind
    pub fn marginalize(self, vars: &[usize]) -> FactorKind {
        match self {
            FactorKind::Dense(d) => {
                let reduced = d.marginalize_kernel(vars); // returns DenseFactor
                match reduced.scope().len() {
                    1 => FactorKind::Unary(dense_into_unary(reduced)),
                    _ => FactorKind::Dense(reduced),
                }
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

    // pub fn combine(self, other: FactorKind) -> FactorKind {
    //     // pattern-match pairs and dispatch to specialized fast paths
    //     // e.g., (Scalar, X) -> add scalar in-place; (Unary, Dense) -> broadcast add in-place
    //     // implement specialized helpers in utils or impl blocks
    //     unimplemented!()
    // }
}
