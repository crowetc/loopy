use crate::belief_state::BeliefState;
use crate::factor::{Factor, FactorKind, FactorOps};
use crate::message::{Endpoint, Message, MessageId};
use crate::semiring::Semiring;

pub(crate) fn compute_message<S>(state: &BeliefState<S>, message_id: MessageId) -> Option<Message>
where
    S: Semiring,
    FactorKind: FactorOps<S>,
{
    let edge = state.messages().edge(message_id);

    match (edge.from, edge.to) {
        (Endpoint::Variable(_), Endpoint::Factor(_)) => variable_to_factor::<S>(state, message_id),

        (Endpoint::Factor(_), Endpoint::Variable(_)) => factor_to_variable::<S>(state, message_id),

        _ => unreachable!("factor-graph edges must connect a variable and a factor"),
    }
}

fn variable_to_factor<S>(state: &BeliefState<S>, message_id: MessageId) -> Option<Message>
where
    S: Semiring,
    FactorKind: FactorOps<S>,
{
    let edge = state.messages().edge(message_id);

    let (Endpoint::Variable(variable), Endpoint::Factor(factor)) = (edge.from, edge.to) else {
        unreachable!("expected variable-to-factor edge");
    };

    let mut accumulator = None;

    for &incoming_id in state.messages().variable_in(variable) {
        let incoming_edge = state.messages().edge(incoming_id);

        let (Endpoint::Factor(source_factor), Endpoint::Variable(_)) =
            (incoming_edge.from, incoming_edge.to)
        else {
            unreachable!("expected factor-to-variable edge");
        };

        if source_factor == factor {
            continue;
        }

        let Some(message) = state.messages().get(incoming_id) else {
            continue;
        };

        accumulator = Some(match accumulator {
            None => message.factor().clone(),
            Some(factor) => <FactorKind as FactorOps<S>>::combine(factor, message.factor().clone()),
        });
    }

    accumulator.map(|factor| {
        Message::try_from(factor).expect("variable-to-factor update must produce a unary factor")
    })
}

fn factor_to_variable<S>(state: &BeliefState<S>, message_id: MessageId) -> Option<Message>
where
    S: Semiring,
    FactorKind: FactorOps<S>,
{
    let edge = state.messages().edge(message_id);

    let (Endpoint::Factor(factor), Endpoint::Variable(variable)) = (edge.from, edge.to) else {
        unreachable!("expected factor-to-variable edge");
    };

    // Start with the factor represented by this node.
    let mut result = state
        .graph()
        .factor(factor)
        .expect("message edge must reference a valid factor")
        .clone();

    // Incorporate incoming variable -> factor messages, excluding the
    // destination variable.
    for &incoming_id in state.messages().factor_in(factor) {
        let incoming_edge = state.messages().edge(incoming_id);

        let (Endpoint::Variable(source_variable), Endpoint::Factor(_)) =
            (incoming_edge.from, incoming_edge.to)
        else {
            unreachable!("expected variable-to-factor edge");
        };

        if source_variable == variable {
            continue;
        }

        let Some(message) = state.messages().get(incoming_id) else {
            continue;
        };

        result = <FactorKind as FactorOps<S>>::combine(result, message.factor().clone());
    }

    // Marginalize every variable except the destination.
    let reduce_vars: Vec<_> = result
        .scope()
        .iter()
        .copied()
        .filter(|id| *id != variable)
        .collect();

    let result = <FactorKind as FactorOps<S>>::reduce(result, &reduce_vars);

    Some(Message::try_from(result).expect("factor-to-variable update must produce a unary factor"))
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

        let result = variable_to_factor::<LogSumProduct>(&state, message_id);

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

        let result =
            variable_to_factor::<LogSumProduct>(&state, outgoing_id).expect("expected a message");

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

        let result =
            variable_to_factor::<LogSumProduct>(&state, outgoing_id).expect("expected a message");

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

        let result = factor_to_variable::<LogSumProduct>(&state, message_id)
            .expect("factor should produce a message");

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

        let result = factor_to_variable::<LogSumProduct>(&state, outgoing_id)
            .expect("expected factor-to-variable message");

        let FactorKind::Unary(result) = result.factor() else {
            panic!("expected unary factor");
        };

        assert_eq!(result.var(), x);
    }
}
