//! Message-update computations for belief propagation.
//!
//! Computes variable-to-factor and factor-to-variable messages using the
//! current belief state. Results are written to a separate destination,
//! leaving the source messages unchanged.
//!
//! Existing destination storage is reused where supported by the factor
//! operations.

use crate::belief_state::BeliefState;
use crate::factor::{Factor, FactorKind, FactorOps};
use crate::message::{Endpoint, Message, MessageId};
use crate::semiring::Semiring;

/// Computes the message identified by `message_id` into `output`.
///
/// Dispatches to the appropriate update based on the direction of the edge.
/// The destination may contain a previous message whose storage can be reused.
///
/// An absent message is represented by `None`.
pub(crate) fn compute_message_into<S>(
    state: &BeliefState<S>,
    message_id: MessageId,
    output: &mut Option<Message>,
) where
    S: Semiring,
    FactorKind: FactorOps<S>,
{
    let edge = state.messages().edge(message_id);

    match (edge.from, edge.to) {
        (Endpoint::Variable(_), Endpoint::Factor(_)) => {
            variable_to_factor_into::<S>(state, message_id, output);
        }
        (Endpoint::Factor(_), Endpoint::Variable(_)) => {
            factor_to_variable_into::<S>(state, message_id, output);
        }
        _ => unreachable!("factor-graph edges must connect a variable and a factor"),
    }
}

/// Combines incoming factor-to-variable messages, excluding the destination factor.
fn variable_to_factor_into<S>(
    state: &BeliefState<S>,
    message_id: MessageId,
    output: &mut Option<Message>,
) where
    S: Semiring,
    FactorKind: FactorOps<S>,
{
    let edge = state.messages().edge(message_id);

    let (Endpoint::Variable(variable), Endpoint::Factor(factor)) = (edge.from, edge.to) else {
        unreachable!("expected variable-to-factor edge");
    };

    let mut incoming = state
        .messages()
        .variable_in(variable)
        .iter()
        .filter_map(|&id| {
            let edge = state.messages().edge(id);

            if edge.from == Endpoint::Factor(factor) {
                None
            } else {
                state.messages().get(id)
            }
        });

    let Some(first) = incoming.next() else {
        *output = None;
        return;
    };

    let mut result = first.factor().clone();

    for message in incoming {
        result = <FactorKind as FactorOps<S>>::combine(result, message.factor());
    }

    // A message must have exactly one variable.
    assert_eq!(result.ndim(), 1);

    match output {
        Some(destination) => {
            *destination.factor_mut() = result;
        }
        None => {
            *output =
                Some(Message::try_from(result).expect("variable-to-factor result must be unary"));
        }
    }
}

/// Combines the factor potential with incoming messages, then reduces to the destination variable.
fn factor_to_variable_into<S>(
    state: &BeliefState<S>,
    message_id: MessageId,
    output: &mut Option<Message>,
) where
    S: Semiring,
    FactorKind: FactorOps<S>,
{
    let edge = state.messages().edge(message_id);

    let (Endpoint::Factor(factor), Endpoint::Variable(variable)) = (edge.from, edge.to) else {
        unreachable!("expected factor-to-variable edge");
    };

    let mut result = state
        .graph()
        .factor(factor)
        .expect("message edge must reference a valid factor")
        .clone();

    for &incoming_id in state.messages().factor_in(factor) {
        let incoming_edge = state.messages().edge(incoming_id);

        if incoming_edge.from == Endpoint::Variable(variable) {
            continue;
        }

        let Some(message) = state.messages().get(incoming_id) else {
            continue;
        };

        result = <FactorKind as FactorOps<S>>::combine(result, message.factor());
    }

    // Eliminate every variable except the message destination.
    let reduce_vars: Vec<_> = result
        .scope()
        .iter()
        .copied()
        .filter(|&id| id != variable)
        .collect();

    if let Some(destination) = output.as_mut() {
        <FactorKind as FactorOps<S>>::reduce_into(&result, &reduce_vars, destination.factor_mut());
        assert_eq!(destination.factor().ndim(), 1);
    } else {
        let reduced = <FactorKind as FactorOps<S>>::reduce(result, &reduce_vars);

        *output =
            Some(Message::try_from(reduced).expect("factor-to-variable result must be unary"));
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::belief_state::BeliefState;
    use crate::factor::{DenseFactor, FactorKind, UnaryFactor};
    use crate::factor_graph::FactorGraph;
    use crate::message::Message;
    use crate::semiring::LogSumProduct;
    use crate::variable::Variable;

    fn binary_variable(name: &str) -> Variable {
        Variable::discrete(name, ["0", "1"])
    }

    fn unary_message(variable: crate::variable::VariableId, values: [f64; 2]) -> Message {
        Message::try_from(FactorKind::Unary(UnaryFactor::new(
            variable,
            vec![values[0], values[1]],
        )))
        .expect("unary factor should be a valid message")
    }

    fn variable_to_factor_id<S>(
        state: &BeliefState<S>,
        variable: crate::variable::VariableId,
        factor: crate::factor::FactorId,
    ) -> MessageId
    where
        S: Semiring,
        FactorKind: FactorOps<S>,
    {
        state
            .messages()
            .variable_out(variable)
            .iter()
            .copied()
            .find(|&id| {
                let edge = state.messages().edge(id);

                matches!(
                    (edge.from, edge.to),
                    (
                        Endpoint::Variable(source),
                        Endpoint::Factor(destination),
                    ) if source == variable && destination == factor
                )
            })
            .expect("expected variable-to-factor message edge")
    }

    fn factor_to_variable_id<S>(
        state: &BeliefState<S>,
        factor: crate::factor::FactorId,
        variable: crate::variable::VariableId,
    ) -> MessageId
    where
        S: Semiring,
        FactorKind: FactorOps<S>,
    {
        state
            .messages()
            .factor_out(factor)
            .iter()
            .copied()
            .find(|&id| {
                let edge = state.messages().edge(id);

                matches!(
                    (edge.from, edge.to),
                    (
                        Endpoint::Factor(source),
                        Endpoint::Variable(destination),
                    ) if source == factor && destination == variable
                )
            })
            .expect("expected factor-to-variable message edge")
    }

    #[test]
    fn variable_to_factor_with_no_other_messages_returns_none() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();

        let factor = graph
            .add_factor(FactorKind::Unary(UnaryFactor::new(x, vec![0.0, 1.0])))
            .unwrap();

        let state = BeliefState::from_graph(graph);

        let message_id = variable_to_factor_id(&state, x, factor);

        let mut result = None;

        variable_to_factor_into::<LogSumProduct>(&state, message_id, &mut result);

        assert!(result.is_none());
    }

    #[test]
    fn variable_to_factor_forwards_single_other_message() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();

        let f = graph
            .add_factor(FactorKind::Unary(UnaryFactor::new(x, vec![0.0, 0.0])))
            .unwrap();

        let g = graph
            .add_factor(FactorKind::Unary(UnaryFactor::new(x, vec![0.0, 0.0])))
            .unwrap();

        let mut state = BeliefState::from_graph(graph);

        let incoming_id = factor_to_variable_id(&state, g, x);

        state
            .messages_mut()
            .set(incoming_id, unary_message(x, [1.0, 2.0]));

        let outgoing_id = variable_to_factor_id(&state, x, f);

        let mut result = None;

        variable_to_factor_into::<LogSumProduct>(&state, outgoing_id, &mut result);

        let result = result.expect("expected a message");

        let FactorKind::Unary(result) = result.factor() else {
            panic!("expected unary factor");
        };

        assert_eq!(result.var(), x);
        assert_eq!(result.data(), &[1.0, 2.0]);
    }

    #[test]
    fn variable_to_factor_combines_multiple_other_messages() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();

        let f = graph
            .add_factor(FactorKind::Unary(UnaryFactor::new(x, vec![0.0, 0.0])))
            .unwrap();

        let g = graph
            .add_factor(FactorKind::Unary(UnaryFactor::new(x, vec![0.0, 0.0])))
            .unwrap();

        let h = graph
            .add_factor(FactorKind::Unary(UnaryFactor::new(x, vec![0.0, 0.0])))
            .unwrap();

        let mut state = BeliefState::from_graph(graph);

        for (factor, values) in [(g, [1.0, 2.0]), (h, [3.0, 4.0])] {
            let incoming_id = factor_to_variable_id(&state, factor, x);

            state
                .messages_mut()
                .set(incoming_id, unary_message(x, values));
        }

        let outgoing_id = variable_to_factor_id(&state, x, f);

        let mut result = None;

        variable_to_factor_into::<LogSumProduct>(&state, outgoing_id, &mut result);

        let result = result.expect("expected a message");

        let FactorKind::Unary(result) = result.factor() else {
            panic!("expected unary factor");
        };

        assert_eq!(result.data(), &[4.0, 6.0]);
    }

    #[test]
    fn factor_to_variable_bootstraps_from_factor() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();
        let y = graph.add_variable(binary_variable("y")).unwrap();

        let factor = FactorKind::Dense(DenseFactor::new(
            vec![x, y],
            array![[0.0, 1.0], [2.0, 3.0],].into_dyn(),
        ));

        let factor_id = graph.add_factor(factor).unwrap();

        let state = BeliefState::from_graph(graph);

        let message_id = factor_to_variable_id(&state, factor_id, x);

        let mut result = None;

        factor_to_variable_into::<LogSumProduct>(&state, message_id, &mut result);

        let result = result.expect("expected a message");

        let FactorKind::Unary(result) = result.factor() else {
            panic!("expected unary factor");
        };

        assert_eq!(result.var(), x);
    }

    #[test]
    fn factor_to_variable_combines_incoming_message_before_reduction() {
        let mut graph = FactorGraph::new();

        let x = graph.add_variable(binary_variable("x")).unwrap();
        let y = graph.add_variable(binary_variable("y")).unwrap();

        let factor_id = graph
            .add_factor(FactorKind::Dense(DenseFactor::new(
                vec![x, y],
                array![[0.0, 0.0], [0.0, 0.0],].into_dyn(),
            )))
            .unwrap();

        let mut state = BeliefState::from_graph(graph);

        let incoming_id = variable_to_factor_id(&state, y, factor_id);

        state
            .messages_mut()
            .set(incoming_id, unary_message(y, [1.0, 2.0]));

        let outgoing_id = factor_to_variable_id(&state, factor_id, x);

        let mut result = None;

        factor_to_variable_into::<LogSumProduct>(&state, outgoing_id, &mut result);

        let result = result.expect("expected a message");

        let FactorKind::Unary(result) = result.factor() else {
            panic!("expected unary factor");
        };

        assert_eq!(result.var(), x);
    }
}
