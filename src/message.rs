//! Message-passing infrastructure for belief propagation.
//!
//! Messages are associated with directed edges of a factor graph and carry
//! information about a single variable. The concrete representation of that
//! information is provided by the factor system.
//!
//! This module also defines stable message identifiers and internal storage
//! used by inference algorithms.

mod directed_edge;
mod store;

pub(crate) use directed_edge::{DirectedEdge, Endpoint};
pub(crate) use store::MessageStore;

use crate::factor::{Factor, FactorKind};
use crate::variable::VariableId;

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

/// Information passed along a directed edge of a factor graph.
///
/// A message is represented by a factor over exactly one variable. The
/// underlying [`FactorKind`] determines the concrete representation and the
/// semiring operations supported by the message.
///
/// Construction is restricted so that multivariate factors cannot be stored
/// as belief-propagation messages.
#[derive(Clone, Debug)]
pub(crate) struct Message {
    factor: FactorKind,
}

impl Message {
    /// Returns the variable represented by this message.
    pub(crate) fn variable(&self) -> VariableId {
        self.factor.scope()[0]
    }

    /// Returns the factor carried by this message.
    pub(crate) fn factor(&self) -> &FactorKind {
        &self.factor
    }

    /// Consumes the message and returns its underlying factor.
    pub(crate) fn into_factor(self) -> FactorKind {
        self.factor
    }
}

impl TryFrom<FactorKind> for Message {
    type Error = FactorKind;

    /// Attempts to construct a message from a factor.
    ///
    /// A valid belief-propagation message must represent exactly one variable.
    /// The original factor is returned if this constraint is not satisfied.
    fn try_from(factor: FactorKind) -> Result<Self, Self::Error> {
        if factor.ndim() == 1 {
            Ok(Self { factor })
        } else {
            Err(factor)
        }
    }
}
