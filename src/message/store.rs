use super::{DirectedEdge, Message, MessageId};

/// Stores messages associated with directed factor-graph edges.
///
/// Messages are stored in a contiguous vector and addressed by a stable
/// message index. This avoids hash-based lookup during message passing and
/// provides a layout that can be used efficiently by parallel algorithms.
pub struct MessageStore<M> {
    messages: Vec<M>,
    edges: Vec<DirectedEdge>,
}

impl<M> MessageStore<M> {
    /// Creates an empty message store.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            edges: Vec::new(),
        }
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
    pub fn add(&mut self, edge: DirectedEdge, message: M) -> MessageId {
        let id = MessageId::new(self.messages.len());

        self.edges.push(edge);
        self.messages.push(message);

        id
    }

    /// Returns the edge associated with a message id.
    pub fn edge(&self, id: MessageId) -> DirectedEdge {
        self.edges[id.index()]
    }

    /// Returns a reference to a message.
    pub fn get(&self, id: MessageId) -> &M {
        &self.messages[id.index()]
    }

    /// Returns a mutable reference to a message.
    pub fn get_mut(&mut self, id: MessageId) -> &mut M {
        &mut self.messages[id.index()]
    }
}

impl<M> Default for MessageStore<M> {
    fn default() -> Self {
        Self::new()
    }
}
