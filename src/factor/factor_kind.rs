use super::{Factor, DenseFactor, ScalarFactor, UnaryFactor};

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
            FactorKind::Dense(d) => d.marginalize(vars),
            FactorKind::Unary(u) => u.marginalize(vars),
            FactorKind::Scalar(s) => s.marginalize(vars),
        }
    }
}
