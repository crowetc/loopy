use super::{DenseFactor, Factor, ScalarFactor, UnaryFactor};

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

    // Consuming reduce: takes ownership and returns a new FactorKind
    fn reduce(self, vars: &[usize]) -> FactorKind {
        match self {
            FactorKind::Dense(d) => d.reduce(vars),
            FactorKind::Unary(u) => u.reduce(vars),
            FactorKind::Scalar(s) => s.reduce(vars),
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
