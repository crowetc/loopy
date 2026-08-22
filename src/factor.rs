pub mod dense_factor;
pub mod factor_graph;
pub mod factor_kind;
pub mod scalar_factor;
pub mod unary_factor;
pub mod variable;

mod log_utils;
mod utils;

pub use dense_factor::DenseFactor;
pub use factor_graph::FactorGraph;
pub use factor_kind::FactorKind;
pub use scalar_factor::ScalarFactor;
pub use unary_factor::UnaryFactor;
pub use variable::{Variable, VariableId};

pub use crate::semiring::{LogMaxProduct, LogSumProduct, Semiring};

/// A factor in the sense used in probabilistic graphical models.
pub trait Factor {
    /// Variables in the factor's scope.
    fn scope(&self) -> &[VariableId];

    /// The number of variables in the factor.
    fn ndim(&self) -> usize {
        self.scope().len()
    }
}

pub trait FactorOps<S: Semiring>: Factor {
    /// Consume self and reduce the given variables.
    fn reduce(self, vars: &[VariableId]) -> FactorKind;

    /// Consume self and combine with other.
    fn combine(self, other: FactorKind) -> FactorKind;
}

/// A factor over discrete variables with finite cardinalities.
pub trait DiscreteFactor: Factor {
    /// Cardinalities of all variables in scope.
    fn card(&self) -> &[usize];
}
