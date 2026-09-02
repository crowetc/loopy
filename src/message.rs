pub mod directed_edge;
pub mod discrete_message;
pub mod message_kind;
pub mod store;

pub use directed_edge::{DirectedEdge, Endpoint};
pub use discrete_message::DiscreteMessage;
pub use message_kind::MessageKind;
pub use store::MessageStore;

use crate::factor::VariableId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MessageId(usize);

impl MessageId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

/// Information passed along an edge of a factor graph.
///
/// A message represents information about a single variable, but the
/// concrete representation depends on the factor family and inference
/// algorithm.
pub trait Message: Send + Sync {
    /// The variable represented by this message.
    fn variable(&self) -> VariableId;
}
