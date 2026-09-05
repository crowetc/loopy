use crate::factor::{UnaryFactor, VariableId};

use super::Message;

/// A message over a discrete variable.
#[derive(Clone, Debug)]
pub(crate) struct DiscreteMessage {
    factor: UnaryFactor,
}

impl DiscreteMessage {
    pub(crate) fn new(factor: UnaryFactor) -> Self {
        Self { factor }
    }

    pub(crate) fn factor(&self) -> &UnaryFactor {
        &self.factor
    }

    pub(crate) fn into_factor(self) -> UnaryFactor {
        self.factor
    }
}

impl Message for DiscreteMessage {
    fn variable(&self) -> VariableId {
        self.factor.var()
    }
}
