use std::marker::PhantomData;

use crate::factor::{Factor, FactorId, FactorKind, FactorOps};
use crate::factor_graph::{FactorGraph, GraphError};
use crate::message::MessageStore;
use crate::semiring::Semiring;
use crate::variable::{Variable, VariableId};

/// An error produced when querying an inference belief.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BeliefError {
    UnknownVariableId(VariableId),
}

#[derive(Debug)]
pub struct BeliefState<S> {
    graph: FactorGraph,
    messages: MessageStore,
    _semiring: PhantomData<S>,
}

impl<S> BeliefState<S>
where
    S: Semiring,
{
    /// Creates a belief state from an existing factor graph.
    pub fn from_graph(graph: FactorGraph) -> Self {
        let messages = MessageStore::from_graph(&graph);

        Self {
            graph,
            messages,
            _semiring: PhantomData,
        }
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
    pub fn apply<F>(&mut self, factor: F) -> Result<FactorId, GraphError>
    where
        F: Into<FactorKind>,
    {
        let factor = factor.into();
        let scope = factor.scope().to_vec();
        let factor_id = self.graph.add_factor(factor)?;

        self.messages.add_factor(factor_id, &scope);

        Ok(factor_id)
    }

    /// Returns the current raw belief for a variable.
    ///
    /// The returned factor, when present, has scope exactly `[variable]`.
    ///
    /// Returns `Ok(None)` when the variable is valid but has no concrete
    /// incoming messages.
    pub fn belief(&self, variable: VariableId) -> Result<Option<FactorKind>, BeliefError>
    where
        FactorKind: FactorOps<S>,
    {
        if self.graph.variable(variable).is_none() {
            return Err(BeliefError::UnknownVariableId(variable));
        }

        let mut belief = None;

        for &message_id in self.messages.variable_in(variable) {
            let Some(message) = self.messages.get(message_id) else {
                continue;
            };

            belief = Some(match belief {
                None => message.factor().clone(),

                Some(factor) => {
                    <FactorKind as FactorOps<S>>::combine(
                        factor,
                        message.factor().clone(),
                    )
                }
            });
        }

        Ok(belief)
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
    use crate::semiring::{LogMaxProduct, LogSumProduct};
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

        let state = BeliefState::<LogMaxProduct>::from_graph(graph);

        assert_eq!(state.messages().factor_in(f).len(), 2);
        assert_eq!(state.messages().factor_out(f).len(), 2);
    }

    #[test]
    fn extend_adds_variable_without_messages() {
        let graph = FactorGraph::new();
        let mut state = BeliefState::<LogSumProduct>::from_graph(graph);

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

        let mut state = BeliefState::<LogSumProduct>::from_graph(graph);

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
            .add_factor(UnaryFactor::new(x, vec![1.0, 2.0]))
            .unwrap();

        let mut state = BeliefState::<LogSumProduct>::from_graph(graph);
        let mut schedule = Synchronous::new();

        schedule.step(&mut state);

        let belief = state
            .belief(x)
            .expect("belief query should succeed")
            .expect("expected belief");

        let FactorKind::Unary(belief) = belief else {
            panic!("expected unary belief");
        };

        let normalizer = (1.0_f64.exp() + 2.0_f64.exp()).ln();

        assert!((belief.data()[0] - (1.0 - normalizer)).abs() < 1e-10);

        assert!((belief.data()[1] - (2.0 - normalizer)).abs() < 1e-10);
    }

    #[test]
    fn belief_combines_multiple_incoming_messages() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();

        graph
            .add_factor(UnaryFactor::new(x, vec![1.0, 2.0]))
            .unwrap();

        graph
            .add_factor(UnaryFactor::new(x, vec![3.0, 4.0]))
            .unwrap();

        let mut state = BeliefState::<LogSumProduct>::from_graph(graph);
        let mut schedule = Synchronous::new();

        schedule.step(&mut state);

        let belief = state
            .belief(x)
            .expect("belief query should succeed")
            .expect("expected belief");

        let FactorKind::Unary(belief) = belief else {
            panic!("expected unary belief");
        };

        let normalizer_a = (1.0_f64.exp() + 2.0_f64.exp()).ln();

        let normalizer_b = (3.0_f64.exp() + 4.0_f64.exp()).ln();

        let expected_0 = (1.0 - normalizer_a) + (3.0 - normalizer_b);

        let expected_1 = (2.0 - normalizer_a) + (4.0 - normalizer_b);

        assert!((belief.data()[0] - expected_0).abs() < 1e-10);
        assert!((belief.data()[1] - expected_1).abs() < 1e-10);
    }

    #[test]
    fn belief_is_none_without_incoming_messages() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();

        let state = BeliefState::<LogMaxProduct>::from_graph(graph);

        assert!(
            state
                .belief(x)
                .expect("belief query should succeed")
                .is_none()
        );
    }

    #[test]
    fn belief_rejects_unknown_variable() {
        let graph = FactorGraph::new();
        let state = BeliefState::<LogSumProduct>::from_graph(graph);

        let unknown = VariableId::new(0);

        let result = state.belief(unknown);

        assert!(matches!(
            result,
            Err(BeliefError::UnknownVariableId(id)) if id == unknown
        ));
    }
}
