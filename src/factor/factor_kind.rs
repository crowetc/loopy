use super::Semiring;
use super::{DenseFactor, Factor, FactorOps, ScalarFactor, UnaryFactor};
use crate::variable::VariableId;

/// A concrete factor representation supported by the library.
///
/// `FactorKind` allows factors with different representations to be stored
/// and manipulated through a single type. The variant identifies the
/// representation of the contained factor.
///
/// The available representations are:
///
/// - [`FactorKind::Dense`] - a dense multidimensional discrete factor;
/// - [`FactorKind::Scalar`] - a scalar factor with no variables;
/// - [`FactorKind::Unary`] - a factor over a single discrete variable.
///
/// `FactorKind` delegates [`Factor`] and [`FactorOps`] operations to the
/// contained factor.
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
