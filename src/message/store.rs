use super::{DirectedEdge, Endpoint, Message, MessageId};
use crate::factor::FactorId;
use crate::factor_graph::FactorGraph;
use crate::variable::VariableId;

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
#[derive(Debug)]
pub(crate) struct MessageStore {
    messages: Vec<Option<Message>>,
    edges: Vec<DirectedEdge>,

    variable_out: Vec<Vec<MessageId>>,
    variable_in: Vec<Vec<MessageId>>,

    factor_out: Vec<Vec<MessageId>>,
    factor_in: Vec<Vec<MessageId>>,
}

impl MessageStore {
    //
    // Construction
    //

    /// Creates an empty message store.
    ///
    /// The returned store contains no messages or directed edges.
    #[cfg(test)]
    fn new() -> Self {
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
    /// `None`, indicating that no message has been computed for the edge.
    ///
    /// The adjacency indices are populated in the same order as the graph's
    /// factor scopes and variable factor-adjacency lists.
    pub(crate) fn from_graph(graph: &FactorGraph) -> Self {
        let message_count: usize = graph
            .factors()
            .iter()
            .map(|factor| factor.variable_ids().len() * 2)
            .sum();

        let mut store = Self {
            messages: Vec::with_capacity(message_count),
            edges: Vec::with_capacity(message_count),
            variable_out: vec![Vec::new(); graph.variables().len()],
            variable_in: vec![Vec::new(); graph.variables().len()],
            factor_out: vec![Vec::new(); graph.factors().len()],
            factor_in: vec![Vec::new(); graph.factors().len()],
        };

        for variable_index in 0..graph.variables().len() {
            store.add_variable(VariableId::new(variable_index));
        }

        for (factor_index, factor_node) in graph.factors().iter().enumerate() {
            store.add_factor(FactorId::new(factor_index), factor_node.variable_ids());
        }

        store
    }

    //
    // Topology
    //

    /// Extends adjacency storage for a newly added variable.
    pub(crate) fn add_variable(&mut self, variable: VariableId) {
        self.ensure_variable(variable.index());
    }

    /// Adds message slots for a newly added factor.
    ///
    /// One message is created in each direction for every variable in `scope`.
    /// The order of the messages matches the order of variables in `scope`.
    pub(crate) fn add_factor(&mut self, factor: FactorId, scope: &[VariableId]) {
        self.ensure_factor(factor.index());

        for &variable in scope {
            self.add(DirectedEdge::variable_to_factor(variable, factor));
            self.add(DirectedEdge::factor_to_variable(factor, variable));
        }
    }

    //
    // Messages
    //

    /// Returns the directed edge associated with `id`.
    pub(crate) fn edge(&self, id: MessageId) -> DirectedEdge {
        self.edges[id.index()]
    }

    /// Returns the message associated with `id`, if one has been computed.
    pub(crate) fn get(&self, id: MessageId) -> Option<&Message> {
        self.messages[id.index()].as_ref()
    }

    /// Stores `message` in the slot associated with `id`.
    ///
    /// Any previously computed message for the edge is replaced.
    pub(crate) fn set(&mut self, id: MessageId, message: Message) {
        self.messages[id.index()] = Some(message);
    }

    //
    // Adjacency
    //

    /// Returns the messages directed from `variable` to neighboring factors.
    ///
    /// The returned slice contains the [`MessageId`] for every message whose
    /// source is `variable`.
    #[allow(dead_code)]
    pub(crate) fn variable_out(&self, variable: VariableId) -> &[MessageId] {
        self.variable_out
            .get(variable.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the messages directed to `variable` from neighboring factors.
    ///
    /// The returned slice contains the [`MessageId`] for every message whose
    /// destination is `variable`.
    pub(crate) fn variable_in(&self, variable: VariableId) -> &[MessageId] {
        self.variable_in
            .get(variable.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the messages directed from `factor` to neighboring variables.
    ///
    /// The returned slice contains the [`MessageId`] for every message whose
    /// source is `factor`.
    pub(crate) fn factor_out(&self, factor: FactorId) -> &[MessageId] {
        self.factor_out
            .get(factor.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the messages directed to `factor` from neighboring variables.
    ///
    /// The returned slice contains the [`MessageId`] for every message whose
    /// destination is `factor`.
    pub(crate) fn factor_in(&self, factor: FactorId) -> &[MessageId] {
        self.factor_in
            .get(factor.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    //
    // Internal Helpers
    //

    /// Adds a directed edge to the store.
    ///
    /// A message slot initialized to `None` is created for the edge, ensuring
    /// that every edge in the store has an associated message slot. The
    /// appropriate incoming and outgoing adjacency indices are also updated.
    ///
    /// Returns the [`MessageId`] assigned to the new edge's message slot.
    fn add(&mut self, edge: DirectedEdge) -> MessageId {
        let id = MessageId::new(self.messages.len());

        self.edges.push(edge);
        self.messages.push(None);

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

    /// Returns the number of message slots in the store.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns `true` if the store contains no message slots.
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor::{DenseFactor, FactorKind};
    use crate::variable::{Variable, VariableId};
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

        let x = graph
            .add_variable(Variable::discrete("x", ["0", "1"]))
            .unwrap();

        let y = graph
            .add_variable(Variable::discrete("y", ["0", "1"]))
            .unwrap();

        let z = graph
            .add_variable(Variable::discrete("z", ["0", "1"]))
            .unwrap();

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

        for (i, &variable) in graph
            .factor_node(f)
            .unwrap()
            .variable_ids()
            .iter()
            .enumerate()
        {
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
            let factors = variable_node.factor_ids();

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

    #[test]
    fn variable_adjacency_order_matches_multiple_factors() {
        let mut graph = FactorGraph::new();

        let x = graph
            .add_variable(Variable::discrete("x", ["0", "1"]))
            .unwrap();

        let y = graph
            .add_variable(Variable::discrete("y", ["0", "1"]))
            .unwrap();

        let f0 = FactorKind::Dense(DenseFactor::new(
            vec![x, y],
            array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
        ));

        let f1 = FactorKind::Dense(DenseFactor::new(vec![x], array![5.0, 6.0].into_dyn()));

        let f0 = graph.add_factor(f0).unwrap();
        let f1 = graph.add_factor(f1).unwrap();

        let store = MessageStore::from_graph(&graph);

        let factors = graph.variable_node(x).unwrap().factor_ids();

        assert_eq!(factors, &[f0, f1]);

        let variable_out = store.variable_out(x);
        let variable_in = store.variable_in(x);

        assert_eq!(variable_out.len(), 2);
        assert_eq!(variable_in.len(), 2);

        for (i, &factor) in factors.iter().enumerate() {
            assert_eq!(
                store.edge(variable_out[i]),
                DirectedEdge::variable_to_factor(x, factor)
            );

            assert_eq!(
                store.edge(variable_in[i]),
                DirectedEdge::factor_to_variable(factor, x)
            );
        }

        // y is connected only to f0.
        assert_eq!(
            store.edge(store.variable_out(y)[0]),
            DirectedEdge::variable_to_factor(y, f0)
        );

        assert_eq!(
            store.edge(store.variable_in(y)[0]),
            DirectedEdge::factor_to_variable(f0, y)
        );
    }

    #[test]
    fn isolated_variable_has_no_messages() {
        let mut graph = FactorGraph::new();

        let x = graph
            .add_variable(Variable::discrete("x", ["0", "1"]))
            .unwrap();

        let store = MessageStore::from_graph(&graph);

        assert!(store.variable_in(x).is_empty());
        assert!(store.variable_out(x).is_empty());
    }

    #[test]
    fn new_message_slot_is_empty() {
        let mut store = MessageStore::new();

        let variable = VariableId::new(0);
        let factor = FactorId::new(0);

        let id = store.add(DirectedEdge::variable_to_factor(variable, factor));

        assert!(store.get(id).is_none());
    }
}
