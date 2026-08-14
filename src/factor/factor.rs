use crate::factor::FactorKind;

pub trait Factor {
    /// Variables in the factor's scope.
    fn scope(&self) -> &[usize];

    /// The number of variables in the factor.
    fn ndim(&self) -> usize {
        return self.scope().len();
    }

    /// Consume self and marginalize the given variables.
    fn marginalize(self, vars: &[usize]) -> FactorKind;

    // /// Consume self and combine with another factor.
    // fn combine(self, other: FactorKind) -> FactorKind;
}
