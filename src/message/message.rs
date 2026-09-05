use crate::factor::{Factor, FactorKind, VariableId};

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
