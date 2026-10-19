// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements evaluations that tell a model update from a
// knowledge update, for its clients. If your team needs expertise in
// attributing an agent's improvement to its weights or to the knowledge it
// retrieves, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What an evaluation arm is: the weights it runs and the knowledge it may
//! retrieve from.
//!
//! A model can get better at a task because it was trained or because it was
//! given something to look up. The two are updated, audited and rolled back
//! differently, so a comparison states which varied between its arms and
//! refuses to attribute an effect when both did.
//!
//! Knowledge is named by the snapshot of the memory the arm read
//! (`sven_sdk::memory::MemoryScope::snapshot_id`), frozen for the evaluation,
//! so an arm's knowledge cannot change while it is measured.

use serde::{Deserialize, Serialize};

/// One arm of a comparison.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arm {
    /// The weights: the identity of the model checkpoint that answers.
    pub weights: String,
    /// The frozen memory snapshot it retrieves from; `None` when it retrieves
    /// nothing.
    pub knowledge: Option<String>,
}

/// What differs between two arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Varied {
    /// The arms are the same treatment.
    Nothing,
    /// Only the weights.
    Weights,
    /// Only the knowledge.
    Knowledge,
    /// Both.
    Both,
}

/// What an effect between two arms is attributed to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Attribution {
    /// A change of weights.
    Weights,
    /// A change of retrieved knowledge.
    Knowledge,
}

/// Why no attribution can be made.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NotAttributable {
    /// The arms are the same treatment: there is no effect to attribute.
    #[error("the arms are the same treatment")]
    Identical,
    /// Weights and knowledge both differ, so an effect could be either.
    #[error("weights and knowledge both differ, so the effect is confounded")]
    Confounded,
}

impl Arm {
    /// An arm running `weights` with `knowledge` frozen.
    #[must_use]
    pub fn new(weights: impl Into<String>, knowledge: Option<String>) -> Self {
        Self {
            weights: weights.into(),
            knowledge,
        }
    }

    /// What differs between `self` and `other`.
    #[must_use]
    pub fn varied_from(&self, other: &Self) -> Varied {
        match (
            self.weights != other.weights,
            self.knowledge != other.knowledge,
        ) {
            (false, false) => Varied::Nothing,
            (true, false) => Varied::Weights,
            (false, true) => Varied::Knowledge,
            (true, true) => Varied::Both,
        }
    }

    /// The arm in words, for a report.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "weights {} with {}",
            self.weights,
            self.knowledge
                .as_deref()
                .map_or_else(|| "no knowledge".to_owned(), |k| format!("knowledge {k}"))
        )
    }
}

/// What an effect between `reference` and `candidate` can be attributed to.
///
/// # Errors
/// [`NotAttributable`] when nothing varied or both did.
pub fn attribute(reference: &Arm, candidate: &Arm) -> Result<Attribution, NotAttributable> {
    match reference.varied_from(candidate) {
        Varied::Nothing => Err(NotAttributable::Identical),
        Varied::Both => Err(NotAttributable::Confounded),
        Varied::Weights => Ok(Attribution::Weights),
        Varied::Knowledge => Ok(Attribution::Knowledge),
    }
}
