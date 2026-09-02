use super::{DirectedEdge, Endpoint, MessageId, MessageKind};
use crate::factor::{FactorGraph, FactorId, VariableId};

/// Stores messages associated with directed factor-graph edges.
///
/// Messages are stored in a contiguous vector and addressed by a stable
/// message index. This avoids hash-based lookup during message passing and
/// provides a layout that can be used efficiently by parallel algorithms.
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

    /// Creates a message store from a factor graph.
    ///
    /// Two message slots are created for every factor-graph edge:
    ///
    /// - variable → factor
    /// - factor → variable
    ///
    /// Messages are initially empty.
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

    /// Returns the number of message slots.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns whether the store contains no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Adds a directed edge and returns its message index.
    ///
    /// The corresponding variable/factor adjacency indices are updated
    /// automatically.
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

    /// Returns the edge associated with a message id.
    pub fn edge(&self, id: MessageId) -> DirectedEdge {
        self.edges[id.index()]
    }

    /// Returns a reference to a message.
    pub fn get(&self, id: MessageId) -> &MessageKind {
        &self.messages[id.index()]
    }

    /// Returns a mutable reference to a message.
    pub fn get_mut(&mut self, id: MessageId) -> &mut MessageKind {
        &mut self.messages[id.index()]
    }

    /// Returns the messages directed from a variable to its neighboring factors.
    pub fn variable_out(&self, variable: VariableId) -> &[MessageId] {
        self.variable_out
            .get(variable.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the messages directed into a variable from neighboring factors.
    pub fn variable_in(&self, variable: VariableId) -> &[MessageId] {
        self.variable_in
            .get(variable.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the messages directed from a factor to neighboring variables.
    pub fn factor_out(&self, factor: FactorId) -> &[MessageId] {
        self.factor_out
            .get(factor.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the messages directed into a factor from neighboring variables.
    pub fn factor_in(&self, factor: FactorId) -> &[MessageId] {
        self.factor_in
            .get(factor.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn ensure_variable(&mut self, index: usize) {
        let required_len = index + 1;

        if self.variable_out.len() < required_len {
            self.variable_out.resize_with(required_len, Vec::new);
            self.variable_in.resize_with(required_len, Vec::new);
        }
    }

    fn ensure_factor(&mut self, index: usize) {
        let required_len = index + 1;

        if self.factor_out.len() < required_len {
            self.factor_out.resize_with(required_len, Vec::new);
            self.factor_in.resize_with(required_len, Vec::new);
        }
    }
}

impl Default for MessageStore {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
}
