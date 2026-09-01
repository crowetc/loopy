pub mod directed_edge;
pub mod discrete;
pub mod store;

pub use directed_edge::DirectedEdge;
pub use discrete::DiscreteMessage;
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
