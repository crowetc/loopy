//! Message-passing infrastructure for belief propagation.
//!
//! Messages are associated with directed edges of a factor graph and carry
//! information about a single variable. The concrete representation of that
//! information is provided by the factor system.
//!
//! This module also defines stable message identifiers and internal storage
//! used by inference algorithms.

mod directed_edge;
mod message;
mod store;

pub(crate) use directed_edge::{DirectedEdge, Endpoint};
pub(crate) use message::Message;
pub(crate) use store::MessageStore;

/// Stable identifier for a directed message in a belief-propagation state.
///
/// A `MessageId` indexes a message slot in [`MessageStore`]. Message IDs remain
/// stable as long as the underlying inference state is extended append-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MessageId(usize);

impl MessageId {
    /// Creates a message identifier from its storage index.
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the storage index represented by this identifier.
    pub(crate) fn index(self) -> usize {
        self.0
    }
}
