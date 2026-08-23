use std::collections::HashMap;

use super::{Factor, FactorKind, Variable, VariableId};

/// A node representing a variable in a [`FactorGraph`].
///
/// The node stores the variable itself and the factors connected to it.
#[derive(Clone, Debug)]
pub struct VariableNode {
    variable: Variable,
    factors: Vec<FactorId>,
}

impl VariableNode {
    fn new(variable: Variable) -> Self {
        Self {
            variable,
            factors: Vec::new(),
        }
    }

    /// Return the variable represented by this node.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }

    /// Return the factors connected to this variable.
    pub fn factors(&self) -> &[FactorId] {
        &self.factors
    }
}

/// A node representing a factor in a [`FactorGraph`].
///
/// The node stores the factor and exposes the variables in its scope as
/// its neighboring variable nodes.
#[derive(Clone, Debug)]
pub struct FactorNode {
    factor: FactorKind,
}

impl FactorNode {
    fn new(factor: FactorKind) -> Self {
        Self { factor }
    }

    /// Return the factor represented by this node.
    pub fn factor(&self) -> &FactorKind {
        &self.factor
    }

    /// Return the variables connected to this factor.
    pub fn scope(&self) -> &[VariableId] {
        self.factor.scope()
    }
}

/// An identifier for a factor within a [`FactorGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FactorId(usize);

impl FactorId {
    fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the index underlying this identifier.
    pub fn index(self) -> usize {
        self.0
    }
}

/// A factor graph is a bipartite graphical model that decomposes a global
/// function into a collection of local functions (factors) over subsets of
/// variables.
#[derive(Clone, Debug)]
pub struct FactorGraph {
    variables: Vec<VariableNode>,
    factors: Vec<FactorNode>,
    variable_registry: HashMap<String, VariableId>,
}

impl FactorGraph {
    /// Create an empty factor graph.
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            variable_registry: HashMap::new(),
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

        if self.variable_registry.contains_key(&name) {
            return Err(GraphError::DuplicateVariableName(name));
        }

        let id = VariableId::new(self.variables.len());

        self.variables.push(VariableNode::new(variable));
        self.variable_registry.insert(name, id);

        Ok(id)
    }

    /// Return the number of variables in the graph.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Return the variable identified by `id`.
    ///
    /// This provides access to the underlying variable without exposing
    /// its graph connectivity. Use [`FactorGraph::variable_node`] when
    /// graph adjacency is also needed.
    pub fn variable(&self, id: VariableId) -> Option<&Variable> {
        self.variables.get(id.index()).map(|node| node.variable())
    }

    /// Return the variable node identified by `id`.
    ///
    /// The node provides access to both the variable and the factors
    /// connected to it.
    pub fn variable_node(&self, id: VariableId) -> Option<&VariableNode> {
        self.variables.get(id.index())
    }

    /// Return the ID of the variable with the given name.
    ///
    /// Variable names are unique within a graph, so a successful lookup
    /// identifies exactly one variable.
    pub fn variable_id(&self, name: &str) -> Option<VariableId> {
        self.variable_registry.get(name).copied()
    }

    /// Add a factor to the graph.
    ///
    /// Every variable in the factor's scope must belong to this graph.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::UnknownVariableId`] if the factor references
    /// a variable that does not belong to this graph.
    pub fn add_factor(&mut self, factor: FactorKind) -> Result<FactorId, GraphError> {
        for &variable_id in factor.scope() {
            if self.variable(variable_id).is_none() {
                return Err(GraphError::UnknownVariableId(variable_id));
            }
        }

        let id = FactorId::new(self.factors.len());

        for &variable_id in factor.scope() {
            self.variables[variable_id.index()].factors.push(id);
        }

        self.factors.push(FactorNode::new(factor));

        Ok(id)
    }

    /// Return the number of factors in the graph.
    pub fn num_factors(&self) -> usize {
        self.factors.len()
    }

    /// Get a factor by its index.
    pub fn factor(&self, id: FactorId) -> Option<&FactorKind> {
        self.factors.get(id.index()).map(|node| node.factor())
    }

    /// Return the factor node identified by `id`.
    pub fn factor_node(&self, id: FactorId) -> Option<&FactorNode> {
        self.factors.get(id.index())
    }

    /// Return all variable nodes in the graph.
    pub fn variables(&self) -> &[VariableNode] {
        &self.variables
    }

    /// Return all factor nodes in the graph.
    pub fn factors(&self) -> &[FactorNode] {
        &self.factors
    }
}

/// Errors that can occur while constructing a factor graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// A variable with the given name already exists in the graph.
    DuplicateVariableName(String),

    /// A factor references a variable that does not belong to this graph.
    UnknownVariableId(VariableId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor::UnaryFactor;

    #[test]
    fn new_graph_is_empty() {
        let graph = FactorGraph::new();

        assert_eq!(graph.num_variables(), 0);
        assert_eq!(graph.num_factors(), 0);
        assert!(graph.variables().is_empty());
        assert!(graph.factors().is_empty());
    }

    #[test]
    fn add_variable_creates_variable_node() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(Variable::new("x")).unwrap();

        let node = graph.variable_node(x).unwrap();

        assert_eq!(node.variable().name(), "x");
        assert!(node.factors().is_empty());
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
        assert_eq!(graph.variable_id("y"), None);
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
    fn variable_ids_correspond_to_storage_order() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(Variable::new("x")).unwrap();
        let y = graph.add_variable(Variable::new("y")).unwrap();
        let z = graph.add_variable(Variable::new("z")).unwrap();

        assert_eq!(x.index(), 0);
        assert_eq!(y.index(), 1);
        assert_eq!(z.index(), 2);

        assert_eq!(graph.variable_node(x).unwrap().variable().name(), "x");
        assert_eq!(graph.variable_node(y).unwrap().variable().name(), "y");
        assert_eq!(graph.variable_node(z).unwrap().variable().name(), "z");
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

    #[test]
    fn factor_scope_must_contain_known_variables() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(Variable::new("x")).unwrap();

        let factor = FactorKind::Unary(UnaryFactor::new(x, vec![0.0, 1.0]));

        assert!(graph.add_factor(factor).is_ok());
    }

    #[test]
    fn factor_scope_cannot_contain_unknown_variable() {
        let mut graph = FactorGraph::new();

        let factor = FactorKind::Unary(UnaryFactor::new(VariableId::new(0), vec![0.0, 1.0]));

        assert_eq!(
            graph.add_factor(factor),
            Err(GraphError::UnknownVariableId(VariableId::new(0)))
        );
    }

    #[test]
    fn invalid_factor_is_not_added() {
        let mut graph = FactorGraph::new();

        let factor = FactorKind::Unary(UnaryFactor::new(VariableId::new(0), vec![0.0, 1.0]));

        assert!(graph.add_factor(factor).is_err());
        assert_eq!(graph.num_factors(), 0);
    }

    #[test]
    fn factor_can_be_retrieved_after_adding() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(Variable::new("x")).unwrap();
        let factor = FactorKind::Unary(UnaryFactor::new(x, vec![0.0, 1.0]));

        let id = graph.add_factor(factor).unwrap();

        assert!(graph.factor(id).is_some());
        assert_eq!(graph.factor(id).unwrap().scope(), &[x]);
    }
}
