use super::{DirectedEdge, Endpoint, MessageId, MessageKind};
use crate::factor::{FactorGraph, FactorId, VariableId};

/// Storage for messages associated with the directed edges of a factor graph.
///
/// Each [`DirectedEdge`] in the store has a corresponding message slot,
/// identified by a [`MessageId`]. Message slots are stored contiguously and
/// can be accessed directly by their identifiers.
///
/// For each factor graph edge, the store contains one directed edge in each
/// direction:
/// - from a variable to a factor;
/// - from a factor to a variable.
///
/// In addition to storing edges and their associated messages, the store
/// maintains adjacency indices for variables and factors. These indices
/// provide direct access to messages entering and leaving each graph node.
///
/// The adjacency indices preserve the ordering of the corresponding graph
/// node adjacency lists. For a factor, the message at position `i` in
/// `factor_in` or `factor_out` corresponds to the variable at position `i`
/// in the factor's scope. For a variable, the message at position `i` in
/// `variable_in` or `variable_out` corresponds to the factor at position `i`
/// in the variable's factor adjacency list.
///
/// This positional correspondence allows message-passing algorithms to
/// associate messages with neighboring nodes without performing additional
/// topology lookups.
pub struct MessageStore {
    messages: Vec<MessageKind>,
    edges: Vec<DirectedEdge>,

    variable_out: Vec<Vec<MessageId>>,
    variable_in: Vec<Vec<MessageId>>,

    factor_out: Vec<Vec<MessageId>>,
    factor_in: Vec<Vec<MessageId>>,
}

impl MessageStore {
    /// Creates an empty message store.
    ///
    /// The returned store contains no messages or directed edges.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            edges: Vec::new(),
            variable_out: Vec::new(),
            variable_in: Vec::new(),
            factor_out: Vec::new(),
            factor_in: Vec::new(),
        }
    }

    /// Creates a message store for the edges of `graph`.
    ///
    /// Each factor graph edge produces two directed edges in the store:
    /// - one from the variable to the factor;
    /// - one from the factor to the variable.
    ///
    /// Each directed edge receives an associated message slot initialized to
    /// [`MessageKind::Empty`].
    pub fn from_graph(graph: &FactorGraph) -> Self {
        let message_count: usize = graph
            .factors()
            .iter()
            .map(|factor| factor.scope().len() * 2)
            .sum();

        let mut store = Self {
            messages: Vec::with_capacity(message_count),
            edges: Vec::with_capacity(message_count),

            variable_out: Vec::new(),
            variable_in: Vec::new(),

            factor_out: vec![Vec::new(); graph.factors().len()],
            factor_in: vec![Vec::new(); graph.factors().len()],
        };

        for (factor_index, factor_node) in graph.factors().iter().enumerate() {
            let factor_id = FactorId::new(factor_index);

            for &variable_id in factor_node.scope() {
                store.add(DirectedEdge::variable_to_factor(variable_id, factor_id));

                store.add(DirectedEdge::factor_to_variable(factor_id, variable_id));
            }
        }

        store
    }

    /// Returns the number of message slots in the store.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns `true` if the store contains no message slots.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Adds a directed edge to the store.
    ///
    /// A message slot initialized to [`MessageKind::Empty`] is created for the
    /// edge, ensuring that every edge in the store has an associated message.
    /// The appropriate incoming and outgoing adjacency indices are also updated.
    ///
    /// Returns the [`MessageId`] assigned to the new edge's message slot.
    pub fn add(&mut self, edge: DirectedEdge) -> MessageId {
        let id = MessageId::new(self.messages.len());

        self.edges.push(edge);
        self.messages.push(MessageKind::Empty);

        match (edge.from, edge.to) {
            (Endpoint::Variable(variable), Endpoint::Factor(factor)) => {
                self.ensure_variable(variable.index());
                self.ensure_factor(factor.index());
                self.variable_out[variable.index()].push(id);
                self.factor_in[factor.index()].push(id);
            }

            (Endpoint::Factor(factor), Endpoint::Variable(variable)) => {
                self.ensure_variable(variable.index());
                self.ensure_factor(factor.index());
                self.factor_out[factor.index()].push(id);
                self.variable_in[variable.index()].push(id);
            }

            _ => unreachable!("factor graph edges must connect variables and factors"),
        }

        id
    }

    /// Returns the directed edge associated with `id`.
    pub fn edge(&self, id: MessageId) -> DirectedEdge {
        self.edges[id.index()]
    }

    /// Returns the message associated with `id`.
    pub fn get(&self, id: MessageId) -> &MessageKind {
        &self.messages[id.index()]
    }

    /// Returns a mutable reference to the message associated with `id`.
    pub fn get_mut(&mut self, id: MessageId) -> &mut MessageKind {
        &mut self.messages[id.index()]
    }

    /// Returns the messages directed from `variable` to neighboring factors.
    ///
    /// The returned slice contains the [`MessageId`] for every message whose
    /// source is `variable`.
    pub fn variable_out(&self, variable: VariableId) -> &[MessageId] {
        self.variable_out
            .get(variable.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the messages directed to `variable` from neighboring factors.
    ///
    /// The returned slice contains the [`MessageId`] for every message whose
    /// destination is `variable`.
    pub fn variable_in(&self, variable: VariableId) -> &[MessageId] {
        self.variable_in
            .get(variable.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the messages directed from `factor` to neighboring variables.
    ///
    /// The returned slice contains the [`MessageId`] for every message whose
    /// source is `factor`.
    pub fn factor_out(&self, factor: FactorId) -> &[MessageId] {
        self.factor_out
            .get(factor.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the messages directed to `factor` from neighboring variables.
    ///
    /// The returned slice contains the [`MessageId`] for every message whose
    /// destination is `factor`.
    pub fn factor_in(&self, factor: FactorId) -> &[MessageId] {
        self.factor_in
            .get(factor.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Ensures adjacency storage exists for `index`.
    fn ensure_variable(&mut self, index: usize) {
        let required_len = index + 1;

        if self.variable_out.len() < required_len {
            self.variable_out.resize_with(required_len, Vec::new);
            self.variable_in.resize_with(required_len, Vec::new);
        }
    }

    /// Ensures adjacency storage exists for `index`.
    fn ensure_factor(&mut self, index: usize) {
        let required_len = index + 1;

        if self.factor_out.len() < required_len {
            self.factor_out.resize_with(required_len, Vec::new);
            self.factor_in.resize_with(required_len, Vec::new);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor::{DenseFactor, FactorKind, Variable, VariableId};
    use ndarray::array;

    #[test]
    fn new_store_is_empty() {
        let store = MessageStore::new();

        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn add_variable_to_factor_indexes_message() {
        let mut store = MessageStore::new();

        let variable = VariableId::new(0);
        let factor = FactorId::new(0);

        let id = store.add(DirectedEdge::variable_to_factor(variable, factor));

        assert_eq!(id, MessageId::new(0));
        assert_eq!(store.len(), 1);

        assert_eq!(store.variable_out(variable), &[id]);
        assert_eq!(store.factor_in(factor), &[id]);

        assert!(store.variable_in(variable).is_empty());
        assert!(store.factor_out(factor).is_empty());
    }

    #[test]
    fn add_factor_to_variable_indexes_message() {
        let mut store = MessageStore::new();

        let variable = VariableId::new(0);
        let factor = FactorId::new(0);

        let id = store.add(DirectedEdge::factor_to_variable(factor, variable));

        assert_eq!(id, MessageId::new(0));
        assert_eq!(store.len(), 1);

        assert_eq!(store.factor_out(factor), &[id]);
        assert_eq!(store.variable_in(variable), &[id]);

        assert!(store.factor_in(factor).is_empty());
        assert!(store.variable_out(variable).is_empty());
    }

    #[test]
    fn add_both_directions_indexes_both_messages() {
        let mut store = MessageStore::new();

        let variable = VariableId::new(0);
        let factor = FactorId::new(0);

        let variable_to_factor = store.add(DirectedEdge::variable_to_factor(variable, factor));

        let factor_to_variable = store.add(DirectedEdge::factor_to_variable(factor, variable));

        assert_eq!(store.len(), 2);

        assert_eq!(store.variable_out(variable), &[variable_to_factor]);

        assert_eq!(store.variable_in(variable), &[factor_to_variable]);

        assert_eq!(store.factor_in(factor), &[variable_to_factor]);

        assert_eq!(store.factor_out(factor), &[factor_to_variable]);
    }

    #[test]
    fn adjacency_order_matches_graph_topology() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(Variable::new("x")).unwrap();
        let y = graph.add_variable(Variable::new("y")).unwrap();
        let z = graph.add_variable(Variable::new("z")).unwrap();

        let factor = FactorKind::Dense(DenseFactor::new(
            vec![x, y, z],
            array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],].into_dyn(),
        ));

        let f = graph.add_factor(factor).unwrap();

        let store = MessageStore::from_graph(&graph);

        let factor_in = store.factor_in(f);
        let factor_out = store.factor_out(f);

        assert_eq!(factor_in.len(), 3);
        assert_eq!(factor_out.len(), 3);

        for (i, &variable) in graph.factor_node(f).unwrap().scope().iter().enumerate() {
            assert_eq!(
                store.edge(factor_in[i]),
                DirectedEdge::variable_to_factor(variable, f)
            );

            assert_eq!(
                store.edge(factor_out[i]),
                DirectedEdge::factor_to_variable(f, variable)
            );
        }

        for &variable in &[x, y, z] {
            let variable_node = graph.variable_node(variable).unwrap();
            let factors = variable_node.factors();

            assert_eq!(factors.len(), 1);
            assert_eq!(factors[0], f);

            let variable_in = store.variable_in(variable);
            let variable_out = store.variable_out(variable);

            assert_eq!(
                store.edge(variable_in[0]),
                DirectedEdge::factor_to_variable(f, variable)
            );

            assert_eq!(
                store.edge(variable_out[0]),
                DirectedEdge::variable_to_factor(variable, f)
            );
        }
    }
}
