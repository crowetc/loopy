use crate::factor::{FactorKind, UnaryFactor};

#[derive(Clone, Debug)]
pub(crate) enum MessageKind {
    Empty,
    Discrete(UnaryFactor),
}

impl MessageKind {
    pub(crate) fn into_factor(self) -> Option<FactorKind> {
        match self {
            MessageKind::Empty => None,
            MessageKind::Discrete(factor) => Some(FactorKind::Unary(factor)),
        }
    }
}
