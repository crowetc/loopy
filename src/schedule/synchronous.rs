use std::marker::PhantomData;

use crate::belief_state::BeliefState;
use crate::factor::{FactorId, FactorKind, FactorOps};
use crate::semiring::Semiring;

use super::Schedule;
use super::updates::compute_message;

#[derive(Clone, Copy, Debug, Default)]
pub struct Synchronous<S> {
    _semiring: PhantomData<S>,
}

impl<S> Synchronous<S> {
    pub fn new() -> Self {
        Self {
            _semiring: PhantomData,
        }
    }
}

impl<S> Schedule for Synchronous<S>
where
    S: Semiring,
    FactorKind: FactorOps<S>,
{
    fn step(&mut self, state: &mut BeliefState) {
        let mut message_ids = Vec::new();

        for index in 0..state.graph().num_factors() {
            let factor = FactorId::new(index);

            message_ids.extend_from_slice(state.messages().factor_in(factor));

            message_ids.extend_from_slice(state.messages().factor_out(factor));
        }

        let updates: Vec<_> = message_ids
            .into_iter()
            .filter_map(|id| compute_message::<S>(state, id).map(|message| (id, message)))
            .collect();

        for (id, message) in updates {
            state.messages_mut().set(id, message);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    use crate::factor::{DenseFactor, FactorKind, UnaryFactor};
    use crate::factor_graph::FactorGraph;
    use crate::semiring::{LogMaxProduct, LogSumProduct};
    use crate::variable::Variable;

    fn binary_variable(name: &str) -> Variable {
        Variable::discrete(name, ["0", "1"])
    }

    #[test]
    fn synchronous_supports_log_max_product() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();
        let y = graph.add_variable(binary_variable("y")).unwrap();

        let factor = graph
            .add_factor(FactorKind::Dense(DenseFactor::new(
                vec![x, y],
                array![[0.0, 1.0], [2.0, 3.0],].into_dyn(),
            )))
            .unwrap();

        let mut state = BeliefState::from_graph(graph);
        let mut schedule = Synchronous::<LogMaxProduct>::new();

        schedule.step(&mut state);

        let factor_to_x = state.messages().factor_out(factor)[0];

        let message = state
            .messages()
            .get(factor_to_x)
            .expect("expected factor-to-variable message");

        let FactorKind::Unary(message) = message.factor() else {
            panic!("expected unary factor");
        };

        assert_eq!(message.data(), &[1.0, 3.0]);
    }

    #[test]
    fn updates_are_visible_only_on_the_next_iteration() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();
        let y = graph.add_variable(binary_variable("y")).unwrap();

        let f = graph
            .add_factor(FactorKind::Unary(UnaryFactor::new(x, vec![1.0, 2.0])))
            .unwrap();

        let g = graph
            .add_factor(FactorKind::Dense(DenseFactor::new(
                vec![x, y],
                array![[0.0, 0.0], [0.0, 0.0],].into_dyn(),
            )))
            .unwrap();

        let mut state = BeliefState::from_graph(graph);
        let mut schedule = Synchronous::<LogSumProduct>::new();

        let f_to_x = state.messages().factor_out(f)[0];

        let x_to_g = state
            .messages()
            .variable_out(x)
            .iter()
            .copied()
            .find(|&id| {
                matches!(
                    state.messages().edge(id).to,
                    crate::message::Endpoint::Factor(destination)
                        if destination == g
                )
            })
            .expect("expected x-to-g message");

        let g_to_y = state.messages().factor_out(g)[1];

        //
        // Iteration 1
        //

        schedule.step(&mut state);

        let f_to_x_message = state
            .messages()
            .get(f_to_x)
            .expect("expected f-to-x message");

        let FactorKind::Unary(f_to_x_factor) = f_to_x_message.factor() else {
            panic!("expected unary factor");
        };

        assert_eq!(f_to_x_factor.data(), &[1.0, 2.0]);

        // x -> g was computed from the message state at the beginning
        // of iteration 1, before f -> x existed.
        assert!(state.messages().get(x_to_g).is_none());

        let g_to_y_message = state
            .messages()
            .get(g_to_y)
            .expect("expected g-to-y message");

        let FactorKind::Unary(g_to_y_factor) = g_to_y_message.factor() else {
            panic!("expected unary factor");
        };

        let expected = f64::ln(2.0);

        assert!((g_to_y_factor.data()[0] - expected).abs() < 1e-10);
        assert!((g_to_y_factor.data()[1] - expected).abs() < 1e-10);

        //
        // Iteration 2
        //

        schedule.step(&mut state);

        let x_to_g_message = state
            .messages()
            .get(x_to_g)
            .expect("expected x-to-g message");

        let FactorKind::Unary(x_to_g_factor) = x_to_g_message.factor() else {
            panic!("expected unary factor");
        };

        // x -> g can now see f -> x from iteration 1.
        assert_eq!(x_to_g_factor.data(), &[1.0, 2.0]);

        let g_to_y_message = state
            .messages()
            .get(g_to_y)
            .expect("expected g-to-y message");

        let FactorKind::Unary(g_to_y_factor) = g_to_y_message.factor() else {
            panic!("expected unary factor");
        };

        // g -> y was computed during iteration 2 from the state at
        // the beginning of iteration 2, when x -> g was still None.
        assert!((g_to_y_factor.data()[0] - expected).abs() < 1e-10);
        assert!((g_to_y_factor.data()[1] - expected).abs() < 1e-10);
    }
}
