use crate::semiring::Semiring;
use crate::variable::VariableId;

use super::{
    DenseFactor, Factor, FactorDistance, FactorNormalize, FactorOps, ScalarFactor, UnaryFactor,
};

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
/// `FactorKind` delegates factor operations to the contained representation.
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
            Self::Dense(factor) => factor.scope(),
            Self::Scalar(factor) => factor.scope(),
            Self::Unary(factor) => factor.scope(),
        }
    }
}

impl FactorDistance for FactorKind {
    fn distance(&self, other: &Self) -> f64 {
        match (self, other) {
            (Self::Unary(a), Self::Unary(b)) => a.distance(b),
            (Self::Dense(a), Self::Dense(b)) => a.distance(b),
            (Self::Scalar(a), Self::Scalar(b)) => a.distance(b),

            _ => panic!("distance requires matching factor representations"),
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
            Self::Dense(factor) => <DenseFactor as FactorOps<S>>::reduce(factor, vars),
            Self::Scalar(factor) => <ScalarFactor as FactorOps<S>>::reduce(factor, vars),
            Self::Unary(factor) => <UnaryFactor as FactorOps<S>>::reduce(factor, vars),
        }
    }

    fn combine(self, other: FactorKind) -> FactorKind {
        match self {
            Self::Dense(factor) => <DenseFactor as FactorOps<S>>::combine(factor, other),
            Self::Scalar(factor) => <ScalarFactor as FactorOps<S>>::combine(factor, other),
            Self::Unary(factor) => <UnaryFactor as FactorOps<S>>::combine(factor, other),
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
