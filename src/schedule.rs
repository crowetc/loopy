mod synchronous;
mod updates;

use crate::belief_state::BeliefState;
use crate::semiring::Semiring;

pub use synchronous::Synchronous;

#[derive(Clone, Copy, Debug)]
pub struct RunOptions {
    pub max_iterations: usize,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunResult {
    pub iterations: usize,
    pub converged: bool,
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
