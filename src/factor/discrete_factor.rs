//! Trait for factors over discrete variables with finite cardinalities.

use super::Factor;

/// A factor over discrete variables with finite cardinalities.
pub trait DiscreteFactor: Factor {
    /// Cardinalities of all variables in scope.
    fn card(&self) -> &[usize];
}
