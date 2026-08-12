//! Core Factor trait.

pub trait Factor {
    /// Variables in the factor's scope.
    fn scope(&self) -> &[usize];

    /// The number of variables in the factor.
    fn ndim(&self) -> usize {
        return self.scope().len();
    }

    /// Marginalize out the given variables.
    fn marginalize(&self, vars: &[usize]) -> Self where Self: Sized;
}
