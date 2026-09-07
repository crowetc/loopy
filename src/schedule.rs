mod synchronous;
mod updates;

use crate::belief_state::BeliefState;
use crate::semiring::Semiring;

pub use synchronous::Synchronous;

/// Defines the order and visibility of message updates during belief propagation.
pub trait Schedule<S: Semiring> {
    /// Execute one message-passing iteration.
    fn step(&mut self, state: &mut BeliefState);
}
