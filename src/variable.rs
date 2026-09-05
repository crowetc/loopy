//! Variables and domains used by factor graphs.
//!
//! A [`Variable`] represents a quantity whose possible values are described
//! by a domain. Variables are assigned graph-local [`VariableId`] values when
//! they are added to a factor graph.
//!
//! The current implementation supports finite discrete domains through
//! [`DiscreteDomain`].

use std::fmt;

mod discrete_domain;

pub use discrete_domain::DiscreteDomain;

/// A graph-local identifier for a variable.
///
/// A `VariableId` has meaning only in the context of the factor graph that
/// created it. The underlying index is an implementation detail.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VariableId(usize);

impl VariableId {
    /// Creates a variable identifier from a graph-local index.
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the graph-local index represented by this identifier.
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

/// The domain of values that a variable may take.
#[derive(Clone, Debug)]
pub enum Domain {
    /// A finite set of discrete values.
    Discrete(DiscreteDomain),
}

/// A variable that can be placed in a factor graph.
///
/// A variable has a name and a domain describing the values it may take.
/// Its graph-local identity is assigned when it is added to a factor graph.
#[derive(Clone, Debug)]
pub struct Variable {
    name: String,
    domain: Domain,
}

impl Variable {
    /// Creates a variable with the given name and domain.
    ///
    /// Variable names are not required to be unique during construction.
    /// Uniqueness is enforced when the variable is added to a factor graph.
    pub fn new(name: impl Into<String>, domain: Domain) -> Self {
        Self {
            name: name.into(),
            domain,
        }
    }

    /// Creates a variable with a finite discrete domain.
    pub fn discrete<I, V>(name: impl Into<String>, values: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<String>,
    {
        Self::new(name, Domain::Discrete(DiscreteDomain::new(values)))
    }

    /// Returns the variable's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the variable's domain.
    pub fn domain(&self) -> &Domain {
        &self.domain
    }
}
