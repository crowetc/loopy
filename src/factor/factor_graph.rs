use std::collections::HashMap;

use super::{FactorKind, Variable, VariableId};

/// A factor graph containing variables and factors.
///
/// The graph owns variable identity and enforces graph-level invariants
/// such as unique variable names.
#[derive(Clone, Debug)]
pub struct FactorGraph {
    variables: Vec<Variable>,
    variable_names: HashMap<String, VariableId>,
    factors: Vec<FactorKind>,
}

impl FactorGraph {
    /// Create an empty factor graph.
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            variable_names: HashMap::new(),
            factors: Vec::new(),
        }
    }

    /// Add a variable to the graph.
    ///
    /// Variable names must be unique within the graph. The returned
    /// [`VariableId`] can be used to refer to the variable from factors
    /// and other graph operations.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::DuplicateVariableName`] if a variable with
    /// the same name already exists in the graph.
    pub fn add_variable(&mut self, variable: Variable) -> Result<VariableId, GraphError> {
        let name = variable.name().to_owned();

        if self.variable_names.contains_key(&name) {
            return Err(GraphError::DuplicateVariableName(name));
        }

        let id = VariableId::new(self.variables.len());

        self.variables.push(variable);
        self.variable_names.insert(name, id);

        Ok(id)
    }

    /// Return the number of variables in the graph.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Return the variable identified by `id`.
    ///
    /// A [`VariableId`] is meaningful only within the graph that created it.
    /// Returns `None` if the ID does not refer to a variable in this graph.
    pub fn variable(&self, id: VariableId) -> Option<&Variable> {
        self.variables.get(id.index())
    }

    /// Return the ID of the variable with the given name.
    ///
    /// Variable names are unique within a graph, so a successful lookup
    /// identifies exactly one variable.
    pub fn variable_id(&self, name: &str) -> Option<VariableId> {
        self.variable_names.get(name).copied()
    }

    /// Add a factor to the graph.
    ///
    /// TODO: Validate that every variable in the factor's scope belongs
    /// to this graph before accepting the factor.
    pub fn add_factor(&mut self, factor: FactorKind) -> usize {
        let id = self.factors.len();
        self.factors.push(factor);
        id
    }

    /// Get a factor by its index.
    pub fn factor(&self, id: usize) -> Option<&FactorKind> {
        self.factors.get(id)
    }

    /// All variables in the graph.
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// All factors in the graph.
    pub fn factors(&self) -> &[FactorKind] {
        &self.factors
    }
}

/// Errors that can occur while constructing a factor graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// A variable with the given name already exists in the graph.
    DuplicateVariableName(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_graph_is_empty() {
        let graph = FactorGraph::new();

        assert_eq!(graph.num_variables(), 0);
        assert!(graph.variables().is_empty());
    }

    #[test]
    fn add_variable_returns_id() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(Variable::new("x")).unwrap();

        assert_eq!(x.index(), 0);
        assert_eq!(graph.num_variables(), 1);
    }

    #[test]
    fn variable_can_be_retrieved_by_id() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(Variable::new("x")).unwrap();

        let variable = graph.variable(x).unwrap();

        assert_eq!(variable.name(), "x");
    }

    #[test]
    fn variable_id_can_be_looked_up_by_name() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(Variable::new("x")).unwrap();

        assert_eq!(graph.variable_id("x"), Some(x));
    }

    #[test]
    fn unknown_variable_name_returns_none() {
        let graph = FactorGraph::new();

        assert_eq!(graph.variable_id("x"), None);
    }

    #[test]
    fn invalid_variable_id_returns_none() {
        let graph = FactorGraph::new();

        let id = VariableId::new(0);

        assert!(graph.variable(id).is_none());
    }

    #[test]
    fn variables_are_stored_in_id_order() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(Variable::new("x")).unwrap();
        let y = graph.add_variable(Variable::new("y")).unwrap();
        let z = graph.add_variable(Variable::new("z")).unwrap();

        assert_eq!(x.index(), 0);
        assert_eq!(y.index(), 1);
        assert_eq!(z.index(), 2);

        assert_eq!(graph.variables()[x.index()].name(), "x");
        assert_eq!(graph.variables()[y.index()].name(), "y");
        assert_eq!(graph.variables()[z.index()].name(), "z");
    }

    #[test]
    fn variable_names_must_be_unique() {
        let mut graph = FactorGraph::new();

        graph.add_variable(Variable::new("x")).unwrap();

        let result = graph.add_variable(Variable::new("x"));

        assert_eq!(
            result,
            Err(GraphError::DuplicateVariableName("x".to_owned()))
        );
    }

    #[test]
    fn different_variables_get_different_ids() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(Variable::new("x")).unwrap();
        let y = graph.add_variable(Variable::new("y")).unwrap();

        assert_ne!(x, y);
    }
}
