use crate::factor::UnaryFactor;

#[derive(Clone, Debug)]
pub enum MessageKind {
    Empty,
    Discrete(UnaryFactor),
}
