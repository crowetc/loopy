//! Synchronous message-passing schedule.
//!
//! All messages in an iteration are computed from the message state at the
//! start of the iteration. Updates are committed only after every message has
//! been computed, preserving synchronous update semantics.

use crate::belief_state::BeliefState;
use crate::factor::{FactorDistance, FactorId, FactorKind, FactorNormalize, FactorOps};
use crate::message::Message;
use crate::semiring::Semiring;

use super::Schedule;
use super::updates::compute_message;

/// A synchronous belief-propagation schedule.
///
/// Message updates are staged for an entire iteration before being committed,
/// so no update is visible to another message computed in the same iteration.
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
    FactorKind: FactorOps<S> + FactorNormalize<S>,
{
    fn step(&mut self, state: &mut BeliefState<S>) -> f64 {
        let mut message_ids = Vec::new();

        for index in 0..state.graph().num_factors() {
            let factor = FactorId::new(index);

            message_ids.extend_from_slice(state.messages().factor_in(factor));
            message_ids.extend_from_slice(state.messages().factor_out(factor));
        }

        let mut max_residual: f64 = 0.0;

        let updates: Vec<_> = message_ids
            .into_iter()
            .filter_map(|id| {
                let message = compute_message::<S>(state, id)?;

                let factor = <FactorKind as FactorNormalize<S>>::normalize(message.into_factor());

                let message = Message::try_from(factor)
                    .expect("normalization must preserve message dimensionality");

                let residual = match state.messages().get(id) {
                    Some(previous) => message.factor().distance(previous.factor()),
                    None => f64::INFINITY,
                };

                max_residual = max_residual.max(residual);

                Some((id, message))
            })
            .collect();

        for (id, message) in updates {
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

    use super::Synchronous;

    use crate::belief_state::BeliefState;
    use crate::factor::{DenseFactor, FactorKind, UnaryFactor};
    use crate::factor_graph::FactorGraph;
    use crate::schedule::{RunOptions, Schedule};
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
            .add_factor(DenseFactor::new(
                vec![x, y],
                array![[0.0, 1.0], [2.0, 3.0],].into_dyn(),
            ))
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
        assert!((message.data()[0] + 2.0).abs() < 1e-10);
        assert!(message.data()[1].abs() < 1e-10);
    }

    #[test]
    fn updates_are_visible_only_on_the_next_iteration() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();
        let y = graph.add_variable(binary_variable("y")).unwrap();

        let f = graph
            .add_factor(UnaryFactor::new(x, vec![1.0, 2.0]))
            .unwrap();

        let g = graph
            .add_factor(DenseFactor::new(
                vec![x, y],
                array![[0.0, 0.0], [0.0, 0.0]].into_dyn(),
            ))
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

        // Iteration 1

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

        // Iteration 2

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

    #[test]
    fn sum_product_matches_exact_marginal_on_tree() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();
        let y = graph.add_variable(binary_variable("y")).unwrap();

        graph
            .add_factor(UnaryFactor::from_linear(x, vec![0.8, 0.2]))
            .unwrap();

        graph
            .add_factor(DenseFactor::from_linear(
                vec![x, y],
                array![[0.9, 0.1], [0.2, 0.8],].into_dyn(),
            ))
            .unwrap();

        let mut state = BeliefState::<LogSumProduct>::from_graph(graph);
        let mut schedule = Synchronous::new();

        schedule.step(&mut state);
        schedule.step(&mut state);
        schedule.step(&mut state);

        let belief = state
            .belief(y)
            .expect("belief query should succeed")
            .expect("expected belief");

        let FactorKind::Unary(belief) = belief else {
            panic!("expected unary belief");
        };

        assert!((belief.data()[0].exp() - 0.76).abs() < 1e-10);
        assert!((belief.data()[1].exp() - 0.24).abs() < 1e-10);
    }

    #[test]
    fn run_converges_to_exact_marginal_on_tree() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();
        let y = graph.add_variable(binary_variable("y")).unwrap();

        graph
            .add_factor(UnaryFactor::from_linear(x, vec![0.8, 0.2]))
            .unwrap();

        graph
            .add_factor(DenseFactor::from_linear(
                vec![x, y],
                array![[0.9, 0.1], [0.2, 0.8],].into_dyn(),
            ))
            .unwrap();

        let mut state = BeliefState::<LogSumProduct>::from_graph(graph);
        let mut schedule = Synchronous::new();

        let result = schedule.run(
            &mut state,
            RunOptions {
                max_iterations: 100,
                tolerance: 1e-10,
            },
        );

        assert!(result.converged);
        assert!(result.iterations > 0);
        assert!(result.residual <= 1e-10);

        let belief = state
            .belief(y)
            .expect("belief query should succeed")
            .expect("expected belief");

        let FactorKind::Unary(belief) = belief else {
            panic!("expected unary belief");
        };

        assert!((belief.data()[0].exp() - 0.76).abs() < 1e-10);
        assert!((belief.data()[1].exp() - 0.24).abs() < 1e-10);
    }

    #[test]
    fn run_converges_to_exact_marginal_on_three_variable_chain() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();
        let y = graph.add_variable(binary_variable("y")).unwrap();
        let z = graph.add_variable(binary_variable("z")).unwrap();

        graph
            .add_factor(UnaryFactor::from_linear(x, vec![0.8, 0.2]))
            .unwrap();

        graph
            .add_factor(DenseFactor::from_linear(
                vec![x, y],
                array![[0.9, 0.1], [0.2, 0.8],].into_dyn(),
            ))
            .unwrap();

        graph
            .add_factor(DenseFactor::from_linear(
                vec![y, z],
                array![[0.7, 0.3], [0.1, 0.9],].into_dyn(),
            ))
            .unwrap();

        let mut state = BeliefState::<LogSumProduct>::from_graph(graph);
        let mut schedule = Synchronous::new();

        let result = schedule.run(
            &mut state,
            RunOptions {
                max_iterations: 100,
                tolerance: 1e-10,
            },
        );

        assert!(result.converged);
        assert!(result.residual <= 1e-10);

        let belief = state
            .belief(z)
            .expect("belief query should succeed")
            .expect("expected belief");

        let FactorKind::Unary(belief) = belief else {
            panic!("expected unary belief");
        };

        assert!((belief.data()[0].exp() - 0.556).abs() < 1e-10);
        assert!((belief.data()[1].exp() - 0.444).abs() < 1e-10);
    }

    #[test]
    fn run_converges_on_three_variable_triangle() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();
        let y = graph.add_variable(binary_variable("y")).unwrap();
        let z = graph.add_variable(binary_variable("z")).unwrap();

        // Evidence favoring x = 0.
        graph
            .add_factor(UnaryFactor::from_linear(x, vec![0.8, 0.2]))
            .unwrap();

        // Pairwise factors favor neighboring variables taking the same state.
        graph
            .add_factor(DenseFactor::from_linear(
                vec![x, y],
                array![[0.8, 0.2], [0.2, 0.8],].into_dyn(),
            ))
            .unwrap();

        graph
            .add_factor(DenseFactor::from_linear(
                vec![y, z],
                array![[0.8, 0.2], [0.2, 0.8],].into_dyn(),
            ))
            .unwrap();

        graph
            .add_factor(DenseFactor::from_linear(
                vec![z, x],
                array![[0.8, 0.2], [0.2, 0.8],].into_dyn(),
            ))
            .unwrap();

        let mut state = BeliefState::<LogSumProduct>::from_graph(graph);
        let mut schedule = Synchronous::new();

        let options = RunOptions {
            max_iterations: 100,
            tolerance: 1e-10,
        };

        let result = schedule.run(&mut state, options);

        assert!(result.converged);
        assert!(result.iterations < options.max_iterations);
        assert!(result.residual.is_finite());
        assert!(result.residual <= options.tolerance);

        let x_belief = state
            .belief(x)
            .expect("belief query should succeed")
            .expect("expected belief for x");

        let y_belief = state
            .belief(y)
            .expect("belief query should succeed")
            .expect("expected belief for y");

        let z_belief = state
            .belief(z)
            .expect("belief query should succeed")
            .expect("expected belief for z");

        let FactorKind::Unary(x_belief) = x_belief else {
            panic!("expected unary belief for x");
        };

        let FactorKind::Unary(y_belief) = y_belief else {
            panic!("expected unary belief for y");
        };

        let FactorKind::Unary(z_belief) = z_belief else {
            panic!("expected unary belief for z");
        };

        // Evidence at x favors state 0, and the attractive pairwise factors
        // should propagate that preference throughout the loop.
        assert!(x_belief.data()[0] > x_belief.data()[1]);
        assert!(y_belief.data()[0] > y_belief.data()[1]);
        assert!(z_belief.data()[0] > z_belief.data()[1]);
    }

    #[test]
    fn run_reports_non_convergence_when_iteration_limit_is_reached() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();
        let y = graph.add_variable(binary_variable("y")).unwrap();
        let z = graph.add_variable(binary_variable("z")).unwrap();

        graph
            .add_factor(UnaryFactor::from_linear(x, vec![0.8, 0.2]))
            .unwrap();

        graph
            .add_factor(DenseFactor::from_linear(
                vec![x, y],
                array![[0.9, 0.1], [0.2, 0.8],].into_dyn(),
            ))
            .unwrap();

        graph
            .add_factor(DenseFactor::from_linear(
                vec![y, z],
                array![[0.7, 0.3], [0.1, 0.9],].into_dyn(),
            ))
            .unwrap();

        let mut state = BeliefState::<LogSumProduct>::from_graph(graph);
        let mut schedule = Synchronous::new();

        let result = schedule.run(
            &mut state,
            RunOptions {
                max_iterations: 1,
                tolerance: 1e-10,
            },
        );

        assert!(!result.converged);
        assert_eq!(result.iterations, 1);
        assert!(result.residual.is_infinite());
    }
}
