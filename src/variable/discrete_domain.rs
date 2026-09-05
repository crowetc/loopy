/// A finite domain of discrete values.
///
/// Values are stored in a fixed order. That ordering defines the mapping
/// between domain values and the indices used by discrete factor
/// representations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscreteDomain {
    values: Vec<String>,
}

impl DiscreteDomain {
    /// Creates a discrete domain from an ordered collection of values.
    pub fn new<I, V>(values: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<String>,
    {
        Self {
            values: values.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns the values in the domain.
    pub fn values(&self) -> &[String] {
        &self.values
    }

    /// Returns the number of values in the domain.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the domain contains no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
