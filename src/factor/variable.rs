use std::fmt;

/// A graph-local identifier for a variable.
///
/// A `VariableId` has meaning only in the context of the
/// [`FactorGraph`](super::FactorGraph) that created it.
///
/// The underlying value is an implementation detail and should not
/// generally be used directly outside the crate.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VariableId(usize);

impl VariableId {
    /// Create a variable ID from its graph-local index.
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the graph-local index represented by this ID.
    pub(crate) fn index(self) -> usize {
        self.0
    }
}

impl fmt::Debug for VariableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for VariableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A variable that can be placed in a factor graph.
///
/// A variable does not own an identifier. Its identity is assigned
/// by the [`FactorGraph`](super::FactorGraph) that contains it.
#[derive(Clone, Debug)]
pub struct Variable {
    name: String,
}

impl Variable {
    /// Create a variable with the given name.
    ///
    /// Variable names are not required to be unique when constructing a
    /// variable. Uniqueness is enforced by [`FactorGraph`](super::FactorGraph)
    /// when the variable is added to a graph.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Return the variable's name.
    pub fn name(&self) -> &str {
        &self.name
    }
}
