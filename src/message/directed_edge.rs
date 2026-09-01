use crate::factor::{FactorId, VariableId};

/// One endpoint of a directed factor-graph edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Endpoint {
    Variable(VariableId),
    Factor(FactorId),
}

/// A directed edge in a factor graph.
///
/// Each graph edge has two possible message directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DirectedEdge {
    pub from: Endpoint,
    pub to: Endpoint,
}

impl DirectedEdge {
    pub fn variable_to_factor(variable: VariableId, factor: FactorId) -> Self {
        Self {
            from: Endpoint::Variable(variable),
            to: Endpoint::Factor(factor),
        }
    }

    pub fn factor_to_variable(factor: FactorId, variable: VariableId) -> Self {
        Self {
            from: Endpoint::Factor(factor),
            to: Endpoint::Variable(variable),
        }
    }
}
