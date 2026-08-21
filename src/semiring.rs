/// An inference regime used by message-passing algorithms.
///
/// A semiring identifies the type of inference being performed.
/// Factor implementations provide the concrete operations for
/// each supported semiring.
pub trait Semiring {}

/// Max-product inference in log space.
///
/// Combining factors corresponds to addition of log-potentials.
/// Reducing variables corresponds to taking a maximum.
#[derive(Clone, Copy, Debug, Default)]
pub struct LogMaxProduct;

impl Semiring for LogMaxProduct {}

/// Sum-product inference in log space.
///
/// Combining factors corresponds to addition of log-potentials.
/// Reducing variables corresponds to log-sum-exp.
#[derive(Clone, Copy, Debug, Default)]
pub struct LogSumProduct;

impl Semiring for LogSumProduct {}
