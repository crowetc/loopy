use crate::factor::{Factor, FactorId, FactorKind, FactorOps};
use crate::factor_graph::{FactorGraph, GraphError};
use crate::message::{MessageStore, combine_message};
use crate::semiring::Semiring;
use crate::variable::{Variable, VariableId};

#[derive(Debug)]
pub struct BeliefState {
    graph: FactorGraph,
    messages: MessageStore,
}

impl BeliefState {
    /// Creates a belief state from an existing factor graph.
    pub fn from_graph(graph: FactorGraph) -> Self {
        let messages = MessageStore::from_graph(&graph);

        Self { graph, messages }
    }

    /// Returns the factor graph associated with this belief state.
    pub fn graph(&self) -> &FactorGraph {
        &self.graph
    }

    /// Extends the inference state with a new variable.
    ///
    /// Unlike [`FactorGraph::add_variable`], this updates both the underlying
    /// factor graph and the associated message state.
    pub fn extend(&mut self, variable: Variable) -> Result<VariableId, GraphError> {
        let variable_id = self.graph.add_variable(variable)?;

        self.messages.add_variable(variable_id);

        Ok(variable_id)
    }

    /// Applies a new factor to the inference state.
    ///
    /// Unlike [`FactorGraph::add_factor`], this updates both the underlying
    /// factor graph and the associated message state while preserving existing
    /// messages.
    pub fn apply(&mut self, factor: FactorKind) -> Result<FactorId, GraphError> {
        let scope = factor.scope().to_vec();
        let factor_id = self.graph.add_factor(factor)?;

        self.messages.add_factor(factor_id, &scope);

        Ok(factor_id)
    }

    pub fn belief<S>(&self, variable: VariableId) -> Option<FactorKind>
    where
        S: Semiring,
        FactorKind: FactorOps<S>,
    {
        let mut accumulator = None;

        for &message_id in self.messages.variable_in(variable) {
            let Some(message) = self.messages.get(message_id) else {
                continue;
            };

            accumulator = Some(combine_message::<S>(accumulator, message));
        }

        accumulator
    }

    pub(crate) fn messages(&self) -> &MessageStore {
        &self.messages
    }

    pub(crate) fn messages_mut(&mut self) -> &mut MessageStore {
        &mut self.messages
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor::{DenseFactor, FactorKind, UnaryFactor};
    use crate::schedule::{Schedule, Synchronous};
    use crate::semiring::LogSumProduct;
    use crate::variable::Variable;
    use ndarray::array;

    fn binary_variable(name: &str) -> Variable {
        Variable::discrete(name, ["0", "1"])
    }

    #[test]
    fn from_graph_builds_message_state() {
        let mut graph = FactorGraph::new();

        let x = graph
            .add_variable(Variable::discrete("x", ["0", "1"]))
            .unwrap();

        let y = graph
            .add_variable(Variable::discrete("y", ["0", "1"]))
            .unwrap();

        let factor = FactorKind::Dense(DenseFactor::new(
            vec![x, y],
            array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
        ));

        let f = graph.add_factor(factor).unwrap();

        let state = BeliefState::from_graph(graph);

        assert_eq!(state.messages().factor_in(f).len(), 2);
        assert_eq!(state.messages().factor_out(f).len(), 2);
    }

    #[test]
    fn extend_adds_variable_without_messages() {
        let graph = FactorGraph::new();
        let mut state = BeliefState::from_graph(graph);

        let x = state.extend(Variable::discrete("x", ["0", "1"])).unwrap();

        assert!(state.messages().variable_in(x).is_empty());
        assert!(state.messages().variable_out(x).is_empty());
    }

    #[test]
    fn apply_adds_factor_messages() {
        let mut graph = FactorGraph::new();

        let x = graph
            .add_variable(Variable::discrete("x", ["0", "1"]))
            .unwrap();

        let y = graph
            .add_variable(Variable::discrete("y", ["0", "1"]))
            .unwrap();

        let mut state = BeliefState::from_graph(graph);

        let factor = FactorKind::Dense(DenseFactor::new(
            vec![x, y],
            array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
        ));

        let f = state.apply(factor).unwrap();

        assert_eq!(state.messages().factor_in(f).len(), 2);
        assert_eq!(state.messages().factor_out(f).len(), 2);

        assert_eq!(state.messages().variable_in(x).len(), 1);
        assert_eq!(state.messages().variable_out(x).len(), 1);

        assert_eq!(state.messages().variable_in(y).len(), 1);
        assert_eq!(state.messages().variable_out(y).len(), 1);
    }

    #[test]
    fn belief_from_single_unary_factor() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();

        graph
            .add_factor(FactorKind::Unary(UnaryFactor::new(x, vec![1.0, 2.0])))
            .unwrap();

        let mut state = BeliefState::from_graph(graph);
        let mut schedule = Synchronous;

        <Synchronous as Schedule<LogSumProduct>>::step(&mut schedule, &mut state);

        let belief = state.belief::<LogSumProduct>(x).expect("expected belief");

        let FactorKind::Unary(belief) = belief else {
            panic!("expected unary belief");
        };

        assert_eq!(belief.data(), &[1.0, 2.0]);
    }

    #[test]
    fn belief_combines_multiple_incoming_messages() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();

        graph
            .add_factor(FactorKind::Unary(UnaryFactor::new(x, vec![1.0, 2.0])))
            .unwrap();

        graph
            .add_factor(FactorKind::Unary(UnaryFactor::new(x, vec![3.0, 4.0])))
            .unwrap();

        let mut state = BeliefState::from_graph(graph);
        let mut schedule = Synchronous;

        <Synchronous as Schedule<LogSumProduct>>::step(&mut schedule, &mut state);

        let belief = state.belief::<LogSumProduct>(x).expect("expected belief");

        let FactorKind::Unary(belief) = belief else {
            panic!("expected unary belief");
        };

        assert_eq!(belief.data(), &[4.0, 6.0]);
    }

    #[test]
    fn belief_is_none_without_incoming_messages() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();

        let state = BeliefState::from_graph(graph);

        assert!(state.belief::<LogSumProduct>(x).is_none());
    }
}
