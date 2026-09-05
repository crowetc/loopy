pub(crate) mod directed_edge;
pub(crate) mod discrete_message;
pub(crate) mod message_kind;
pub(crate) mod store;

pub(crate) use directed_edge::{DirectedEdge, Endpoint};
pub(crate) use discrete_message::DiscreteMessage;
pub(crate) use message_kind::MessageKind;
pub(crate) use store::MessageStore;

use crate::factor::VariableId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MessageId(usize);

impl MessageId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    pub(crate) fn index(self) -> usize {
        self.0
    }
}

/// Information passed along an edge of a factor graph.
///
/// A message represents information about a single variable, but the
/// concrete representation depends on the factor family and inference
/// algorithm.
pub(crate) trait Message: Send + Sync {
    /// The variable represented by this message.
    fn variable(&self) -> VariableId;
}
