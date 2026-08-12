//! Unary factor: a factor over a single discrete variable.

use super::Factor;
use super::DiscreteFactor;

/// A unary factor over a single variable.
///
/// Stores:
/// - `var`: the variable ID
/// - `data`: the factor values for each domain element
#[derive(Clone, Debug)]
pub struct UnaryFactor {
    var: usize,
    data: Vec<f64>,
    card: [usize; 1], // stable slice
}

impl UnaryFactor {
    /// Create a unary factor for a variable with the given data.
    pub fn new(var: usize, data: Vec<f64>) -> Self {
        assert!(!data.is_empty());
        let card = [data.len()];
        Self { var, data, card }
    }

    /// Access the underlying data.
    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

impl Factor for UnaryFactor {
    fn scope(&self) -> &[usize] {
        std::slice::from_ref(&self.var)
    }

    fn marginalize(&self, vars: &[usize]) -> Self {
        if vars.contains(&self.var) {
            let sum: f64 = self.data.iter().sum();
            UnaryFactor::new(self.var, vec![sum])
        } else {
            self.clone()
        }
    }
}

impl DiscreteFactor for UnaryFactor {
    fn card(&self) -> &[usize] {
        &self.card
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope() {
        let f = UnaryFactor::new(7, vec![0.1, 0.9]);
        assert_eq!(f.scope(), &[7]);
    }

    #[test]
    fn test_card() {
        let f = UnaryFactor::new(3, vec![1.0, 2.0, 3.0]);
        assert_eq!(f.card(), &[3]);
    }

    #[test]
    fn test_marginalize_kept() {
        let f = UnaryFactor::new(2, vec![1.0, 2.0, 3.0]);
        let g = f.marginalize(&[]);
        assert_eq!(g.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_marginalize_eliminated() {
        let f = UnaryFactor::new(2, vec![1.0, 2.0, 3.0]);
        let g = f.marginalize(&[2]);
        assert_eq!(g.data(), &[6.0]);
        assert_eq!(g.card(), &[1]);
    }
}
