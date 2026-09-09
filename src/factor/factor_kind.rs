use crate::semiring::Semiring;
use crate::variable::VariableId;

use super::{DenseFactor, Factor, FactorNormalize, FactorOps, FactorDistance, ScalarFactor, UnaryFactor};

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

impl From<DenseFactor> for FactorKind {
    fn from(factor: DenseFactor) -> Self {
        Self::Dense(factor)
    }
}

impl From<ScalarFactor> for FactorKind {
    fn from(factor: ScalarFactor) -> Self {
        Self::Scalar(factor)
    }
}

impl From<UnaryFactor> for FactorKind {
    fn from(factor: UnaryFactor) -> Self {
        Self::Unary(factor)
    }
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

impl FactorDistance for FactorKind {
    fn distance(&self, other: &Self) -> f64 {
        match (self, other) {
            (Self::Unary(a), Self::Unary(b)) => a.distance(b),
            (Self::Dense(a), Self::Dense(b)) => a.distance(b),
            (Self::Scalar(a), Self::Scalar(b)) => a.distance(b),

            _ => panic!("residual requires matching factor representations"),
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

impl<S: Semiring> FactorNormalize<S> for FactorKind
where
    DenseFactor: FactorNormalize<S>,
    ScalarFactor: FactorNormalize<S>,
    UnaryFactor: FactorNormalize<S>,
{
    fn normalize(self) -> Self {
        match self {
            Self::Dense(factor) => Self::Dense(factor.normalize()),
            Self::Scalar(factor) => Self::Scalar(factor.normalize()),
            Self::Unary(factor) => Self::Unary(factor.normalize()),
        }
    }
}
