use crate::belief_state::BeliefState;
use crate::factor::{FactorDistance, FactorId, FactorKind, FactorOps};
use crate::message::{Message, MessageOps};
use crate::semiring::Semiring;

use super::Schedule;
use super::updates::compute_message;

#[derive(Clone, Copy, Debug, Default)]
pub struct Synchronous;

impl Synchronous {
    pub fn new() -> Self {
        Self
    }
}

impl<S> Schedule<S> for Synchronous
where
    S: Semiring,
    FactorKind: FactorOps<S>,
    Message: MessageOps<S>,
{
    fn step(&mut self, state: &mut BeliefState<S>) -> f64 {
        let mut message_ids = Vec::new();

        for index in 0..state.graph().num_factors() {
            let factor = FactorId::new(index);

            message_ids.extend_from_slice(state.messages().factor_in(factor));

            message_ids.extend_from_slice(state.messages().factor_out(factor));
        }

        let updates: Vec<_> = message_ids
            .into_iter()
            .filter_map(|id| {
                let message = compute_message::<S>(state, id)?;

                let message = <Message as MessageOps<S>>::normalize(message);

                let residual = match state.messages().get(id) {
                    Some(previous) => message.factor().distance(previous.factor()),
                    None => f64::INFINITY,
                };

                Some((id, message, residual))
            })
            .collect();

        let max_residual = updates
            .iter()
            .map(|(_, _, residual)| *residual)
            .fold(0.0, f64::max);

        for (id, message, _) in updates {
            state.messages_mut().set(id, message);
        }

        max_residual
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

    fn assert_values_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            );
        }
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

        let mut state = BeliefState::<LogMaxProduct>::from_graph(graph);
        let mut schedule = Synchronous::new();

        schedule.step(&mut state);

        let factor_to_x = state.messages().factor_out(factor)[0];

        let message = state
            .messages()
            .get(factor_to_x)
            .expect("expected factor-to-variable message");

        let FactorKind::Unary(message) = message.factor() else {
            panic!("expected unary factor");
        };

        // Max-product reduction gives [1.0, 3.0].
        // Message normalization subtracts the maximum value, 3.0.
        assert_values_close(message.data(), &[-2.0, 0.0]);
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
                array![[0.0, 0.0], [0.0, 0.0]].into_dyn(),
            )))
            .unwrap();

        let mut state = BeliefState::<LogSumProduct>::from_graph(graph);
        let mut schedule = Synchronous::new();

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

        let normalizer = (1.0_f64.exp() + 2.0_f64.exp()).ln();

        assert!((f_to_x_factor.data()[0] - (1.0 - normalizer)).abs() < 1e-10);
        assert!((f_to_x_factor.data()[1] - (2.0 - normalizer)).abs() < 1e-10);

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

        // The raw message is [ln(2), ln(2)]. After sum-product
        // normalization it becomes [-ln(2), -ln(2)].
        let expected = -f64::ln(2.0);

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

        // x -> g can now see the normalized f -> x message from
        // iteration 1.
        assert!((x_to_g_factor.data()[0] - (1.0 - normalizer)).abs() < 1e-10);
        assert!((x_to_g_factor.data()[1] - (2.0 - normalizer)).abs() < 1e-10);

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
