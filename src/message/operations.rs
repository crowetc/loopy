use crate::factor::{DenseFactor, Factor, FactorKind, FactorOps, FactorResidual, UnaryFactor};
use crate::semiring::{LogMaxProduct, LogSumProduct, Semiring};

use super::Message;

/// Operations used to manipulate messages under a particular semiring.
pub(crate) trait MessageOps<S>
where
    S: Semiring,
    FactorKind: FactorOps<S>,
{
    /// Combines a message with an accumulated factor.
    ///
    /// If no factor has been accumulated, the message factor becomes the
    /// initial value.
    fn combine(accumulator: Option<FactorKind>, message: &Message) -> FactorKind {
        match accumulator {
            None => message.factor().clone(),
            Some(factor) => <FactorKind as FactorOps<S>>::combine(factor, message.factor().clone()),
        }
    }

    /// Normalizes the message under this semiring.
    fn normalize(self) -> Self;

    /// Returns the maximum absolute difference between corresponding
    /// log-values in two messages.
    fn residual(&self, other: &Self) -> f64;
}

fn log_sum_exp(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<_> = values.collect();

    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if max == f64::NEG_INFINITY {
        return max;
    }

    max + values
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f64>()
        .ln()
}

impl MessageOps<LogSumProduct> for Message {
    fn normalize(self) -> Self {
        let factor = match self.into_factor() {
            FactorKind::Unary(factor) => {
                let normalizer = log_sum_exp(factor.data().iter().copied());

                let data = factor
                    .data()
                    .iter()
                    .map(|value| value - normalizer)
                    .collect();

                FactorKind::Unary(UnaryFactor::new(factor.var(), data))
            }

            FactorKind::Dense(factor) => {
                let normalizer = log_sum_exp(factor.data().iter().copied());

                let scope = factor.scope().to_vec();
                let data = factor.data().mapv(|value| value - normalizer);

                FactorKind::Dense(DenseFactor::new(scope, data))
            }

            FactorKind::Scalar(_) => {
                unreachable!("messages must have exactly one variable")
            }
        };

        Message::try_from(factor).expect("normalization must preserve message dimensionality")
    }

    fn residual(&self, other: &Self) -> f64 {
        self.factor().residual(other.factor())
    }
}

impl MessageOps<LogMaxProduct> for Message {
    fn normalize(self) -> Self {
        let factor = match self.into_factor() {
            FactorKind::Unary(factor) => {
                let normalizer = factor
                    .data()
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);

                let data = factor
                    .data()
                    .iter()
                    .map(|value| value - normalizer)
                    .collect();

                FactorKind::Unary(UnaryFactor::new(factor.var(), data))
            }

            FactorKind::Dense(factor) => {
                let normalizer = factor
                    .data()
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);

                let scope = factor.scope().to_vec();
                let data = factor.data().mapv(|value| value - normalizer);

                FactorKind::Dense(DenseFactor::new(scope, data))
            }

            FactorKind::Scalar(_) => {
                unreachable!("messages must have exactly one variable")
            }
        };

        Message::try_from(factor).expect("normalization must preserve message dimensionality")
    }

    fn residual(&self, other: &Self) -> f64 {
        self.factor().residual(other.factor())
    }
}

#[cfg(test)]
mod tests {
    use crate::factor::{FactorKind, UnaryFactor};
    use crate::message::Message;
    use crate::semiring::LogSumProduct;
    use crate::variable::VariableId;

    use super::MessageOps;

    #[test]
    fn initializes_accumulator_from_message() {
        let variable = VariableId::new(0);

        let factor = FactorKind::Unary(UnaryFactor::new(variable, vec![0.0, 1.0]));

        let message = Message::try_from(factor).expect("unary factor should be a valid message");

        let result = <Message as MessageOps<LogSumProduct>>::combine(None, &message);

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

        let result = <Message as MessageOps<LogSumProduct>>::combine(Some(accumulator), &message);

        match result {
            FactorKind::Unary(result) => {
                assert_eq!(result.var(), variable);
                assert_eq!(result.data(), &[4.0, 6.0]);
            }

            _ => panic!("expected unary factor"),
        }
    }
}
