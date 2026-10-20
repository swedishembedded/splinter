// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Mutations of a reference implementation: one exact text replacement each.

use serde::Deserialize;

/// One candidate bug: replace `find` (which occurs exactly once in the
/// reference source) by `replace`.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Mutation {
    /// Stable identifier within the family.
    pub id: String,
    /// The text to replace; must occur exactly once.
    pub find: String,
    /// What it becomes; must differ from `find`.
    pub replace: String,
    /// The property the mutant must violate: an invariant name, or `trace`
    /// when only trace validation against the specification catches it.
    pub violates: String,
    /// What the change does, for the people reading the task catalog.
    #[serde(default)]
    pub note: String,
}

/// Why a mutation does not apply to a source.
#[derive(Debug, PartialEq, Eq)]
pub enum MutationError {
    /// `find` is not in the source.
    NoMatch,
    /// `find` occurs more than once, so the change is ambiguous.
    Ambiguous(usize),
    /// `replace` equals `find`, so nothing would change.
    NoChange,
}

impl std::fmt::Display for MutationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMatch => write!(f, "the text to replace does not occur in the source"),
            Self::Ambiguous(n) => write!(f, "the text to replace occurs {n} times, not once"),
            Self::NoChange => write!(f, "the replacement equals the text it replaces"),
        }
    }
}

impl std::error::Error for MutationError {}

impl Mutation {
    /// The mutated source.
    pub fn apply(&self, source: &str) -> Result<String, MutationError> {
        if self.find == self.replace {
            return Err(MutationError::NoChange);
        }
        match source.matches(&self.find).count() {
            0 => Err(MutationError::NoMatch),
            1 => Ok(source.replacen(&self.find, &self.replace, 1)),
            n => Err(MutationError::Ambiguous(n)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mutation(find: &str, replace: &str) -> Mutation {
        Mutation {
            id: "m".into(),
            find: find.into(),
            replace: replace.into(),
            violates: "trace".into(),
            note: String::new(),
        }
    }

    #[test]
    fn replaces_the_single_occurrence() {
        assert_eq!(
            mutation("a < b", "a <= b").apply("if a < b then"),
            Ok("if a <= b then".into())
        );
    }

    #[test]
    fn refuses_ambiguous_missing_and_empty_changes() {
        assert_eq!(
            mutation("x", "y").apply("x x"),
            Err(MutationError::Ambiguous(2))
        );
        assert_eq!(mutation("x", "y").apply("z"), Err(MutationError::NoMatch));
        assert_eq!(mutation("x", "x").apply("x"), Err(MutationError::NoChange));
    }
}
