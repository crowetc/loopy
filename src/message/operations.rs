use crate::factor::{DenseFactor, Factor, FactorKind, FactorOps, FactorDistance, UnaryFactor};
use crate::semiring::{LogMaxProduct, LogSumProduct, Semiring};

use super::Message;

/// Operations used to manipulate messages under a particular semiring.
pub(crate) trait MessageOps<S>
where
    S: Semiring,
    FactorKind: FactorOps<S>,
{
    /// Normalizes the message under this semiring.
    fn normalize(self) -> Self;

    /// Returns the maximum absolute difference between corresponding
    /// log-values in two messages.
    fn distance(&self, other: &Self) -> f64;
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

    fn distance(&self, other: &Self) -> f64 {
        self.factor().distance(other.factor())
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

    fn distance(&self, other: &Self) -> f64 {
        self.factor().distance(other.factor())
    }
}
