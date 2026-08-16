use super::{Factor, DenseFactor, ScalarFactor, UnaryFactor};

#[derive(Clone, Debug)]
pub enum FactorKind {
    Dense(DenseFactor),
    Scalar(ScalarFactor),
    Unary(UnaryFactor),
}

impl Factor for FactorKind {
    fn scope(&self) -> &[usize] {
        match self {
            FactorKind::Dense(d) => d.scope(),
            FactorKind::Scalar(s) => s.scope(),
            FactorKind::Unary(u) => u.scope(),
        }
    }

    // Consuming marginalize: takes ownership and returns a new FactorKind
    fn marginalize(self, vars: &[usize]) -> FactorKind {
        match self {
            FactorKind::Dense(d) => d.marginalize(vars),
            FactorKind::Unary(u) => u.marginalize(vars),
            FactorKind::Scalar(s) => s.marginalize(vars),
        }
    }

    // Consuming combine
    fn combine(self, other: FactorKind) -> FactorKind {
        match self {
            FactorKind::Dense(d) => d.combine(other),
            FactorKind::Unary(u) => u.combine(other),
            FactorKind::Scalar(s) => s.combine(other),
        }
    }
}
