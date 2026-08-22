use super::Semiring;
use super::{DenseFactor, Factor, FactorOps, ScalarFactor, UnaryFactor, VariableId};

#[derive(Clone, Debug)]
pub enum FactorKind {
    Dense(DenseFactor),
    Scalar(ScalarFactor),
    Unary(UnaryFactor),
}

impl Factor for FactorKind {
    fn scope(&self) -> &[VariableId] {
        match self {
            FactorKind::Dense(d) => d.scope(),
            FactorKind::Scalar(s) => s.scope(),
            FactorKind::Unary(u) => u.scope(),
        }
    }
}

impl<S: Semiring> FactorOps<S> for FactorKind
where
    DenseFactor: FactorOps<S>,
    ScalarFactor: FactorOps<S>,
    UnaryFactor: FactorOps<S>,
{
    fn reduce(self, vars: &[VariableId]) -> FactorKind {
        match self {
            FactorKind::Dense(d) => <DenseFactor as FactorOps<S>>::reduce(d, vars),

            FactorKind::Scalar(s) => <ScalarFactor as FactorOps<S>>::reduce(s, vars),

            FactorKind::Unary(u) => <UnaryFactor as FactorOps<S>>::reduce(u, vars),
        }
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        match self {
            FactorKind::Dense(d) => <DenseFactor as FactorOps<S>>::combine(d, other),

            FactorKind::Scalar(s) => <ScalarFactor as FactorOps<S>>::combine(s, other),

            FactorKind::Unary(u) => <UnaryFactor as FactorOps<S>>::combine(u, other),
        }
    }
}
