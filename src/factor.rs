pub mod factor_kind;
pub mod dense_factor;
pub mod scalar_factor;
pub mod unary_factor;

mod log_utils;
mod utils;

pub use factor_kind::FactorKind;
pub use dense_factor::DenseFactor;
pub use scalar_factor::ScalarFactor;
pub use unary_factor::UnaryFactor;

/// A factor in the sense used in probabilistic graphical models.
pub trait Factor {
    /// Variables in the factor's scope.
    fn scope(&self) -> &[usize];

    /// The number of variables in the factor.
    fn ndim(&self) -> usize { return self.scope().len(); }

    /// Consume self and marginalize the given variables.
    fn marginalize(self, vars: &[usize]) -> FactorKind;

    /// Consume self and combine with other.
    fn combine(self, other: FactorKind) -> FactorKind;
}

/// A factor over discrete variables with finite cardinalities.
pub trait DiscreteFactor: Factor {
    /// Cardinalities of all variables in scope.
    fn card(&self) -> &[usize];
}
