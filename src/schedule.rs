mod synchronous;
mod updates;

use crate::belief_state::BeliefState;

pub use synchronous::Synchronous;

/// Defines the order and visibility of message updates during belief propagation.
pub trait Schedule {
    /// Execute one message-passing iteration.
    fn step(&mut self, state: &mut BeliefState);
}
