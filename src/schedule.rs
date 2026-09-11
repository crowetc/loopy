//! Message-passing schedules and convergence control.

mod synchronous;
mod updates;

use crate::belief_state::BeliefState;
use crate::semiring::Semiring;

pub use synchronous::Synchronous;

/// Controls iterative belief propagation.
#[derive(Clone, Copy, Debug)]
pub struct RunOptions {
    /// Maximum number of message-passing iterations.
    pub max_iterations: usize,

    /// Residual threshold used to determine convergence.
    pub tolerance: f64,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
        }
    }
}

/// Summary of a belief-propagation run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunResult {
    /// Number of iterations executed.
    pub iterations: usize,

    /// Whether the final residual satisfied the convergence tolerance.
    pub converged: bool,

    /// Maximum message residual from the final iteration.
    pub residual: f64,
}

/// Defines the order and visibility of message updates during belief propagation.
pub trait Schedule<S>
where
    S: Semiring,
{
    /// Executes one message-passing iteration.
    ///
    /// Returns the maximum message residual produced by the iteration.
    fn step(&mut self, state: &mut BeliefState<S>) -> f64;

    /// Runs message passing until convergence or the iteration limit is reached.
    ///
    /// Convergence is reached when the maximum message residual produced by an
    /// iteration is less than or equal to `options.tolerance`.
    fn run(&mut self, state: &mut BeliefState<S>, options: RunOptions) -> RunResult {
        let mut residual = f64::INFINITY;

        for iteration in 1..=options.max_iterations {
            residual = self.step(state);

            if residual <= options.tolerance {
                return RunResult {
                    iterations: iteration,
                    converged: true,
                    residual,
                };
            }
        }

        RunResult {
            iterations: options.max_iterations,
            converged: false,
            residual,
        }
    }
}
