use crate::factor::{FactorKind, FactorOps};
use crate::semiring::Semiring;

use super::Message;

/// Accumulates a message into an optional factor.
///
/// If no factor has been accumulated yet, the message's factor initializes
/// the accumulator. Otherwise, the message is combined with the accumulated
/// factor using the operations defined for the selected semiring.
///
/// The message is borrowed from message storage. Its underlying factor is
/// cloned only when ownership is required by [`FactorOps::combine`].
pub(crate) fn combine_message<S>(accumulator: Option<FactorKind>, message: &Message) -> FactorKind
where
    S: Semiring,
    FactorKind: FactorOps<S>,
{
    match accumulator {
        None => message.factor().clone(),

        Some(factor) => <FactorKind as FactorOps<S>>::combine(factor, message.factor().clone()),
    }
}

#[cfg(test)]
mod tests {
    use crate::factor::{FactorKind, UnaryFactor};
    use crate::message::Message;
    use crate::semiring::LogSumProduct;
    use crate::variable::VariableId;

    use super::combine_message;

    #[test]
    fn initializes_accumulator_from_message() {
        let variable = VariableId::new(0);

        let factor = FactorKind::Unary(UnaryFactor::new(variable, vec![0.0, 1.0]));

        let message = Message::try_from(factor).expect("unary factor should be a valid message");

        let result = combine_message::<LogSumProduct>(None, &message);

        match result {
            FactorKind::Unary(result) => {
                assert_eq!(result.var(), variable);
                assert_eq!(result.data(), &[0.0, 1.0]);
            }

            _ => panic!("expected unary factor"),
        }
    }

    #[test]
    fn combines_message_with_existing_factor() {
        let variable = VariableId::new(0);

        let accumulator = FactorKind::Unary(UnaryFactor::new(variable, vec![1.0, 2.0]));

        let message_factor = FactorKind::Unary(UnaryFactor::new(variable, vec![3.0, 4.0]));

        let message =
            Message::try_from(message_factor).expect("unary factor should be a valid message");

        let result = combine_message::<LogSumProduct>(Some(accumulator), &message);

        match result {
            FactorKind::Unary(result) => {
                assert_eq!(result.var(), variable);
                assert_eq!(result.data(), &[4.0, 6.0]);
            }

            _ => panic!("expected unary factor"),
        }
    }
}
