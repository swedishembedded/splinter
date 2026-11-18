// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Why one attempt failed, with the evidence that decided it.
//!
//! A [`DiagnosticResult`] is what a deterministic diagnosis says about a
//! failed attempt: one [`FailureClass`], the rule that assigned it, the step
//! of the trajectory where the decisive error was made (when one was), and
//! the evidence spans the rule read: each names the step, or the record field,
//! and the excerpt that decided it. The same evidence always gives the same
//! result.
//!
//! A model may read the attempt too, but only to add [`Hypothesis`]es: each is
//! labelled with who proposed it and never changes the class, the decisive
//! step or the evidence ([`DiagnosticResult::add_hypothesis`]).

use serde::{Deserialize, Serialize};

use crate::outcome::{Attribution, FailureClass};

/// The longest excerpt an evidence span keeps, in characters.
pub const EXCERPT_CHARS: usize = 240;

/// One piece of evidence a diagnosis read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceSpan {
    /// The ATIF `step_id` of the trajectory step it was read from; `None` for
    /// evidence from a record (the score record, the verifier's output, the
    /// gateway's statuses).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u64>,
    /// What was read: a record field (`score.status`, `verification.reason`,
    /// `gateway.status`) or a part of the step (`tool_call`, `observation`,
    /// `message`).
    pub field: String,
    /// The text that decided it, cut to [`EXCERPT_CHARS`].
    pub excerpt: String,
}

impl EvidenceSpan {
    /// A span of `field` (of `step`, when it is a step's) holding `text`, cut
    /// to [`EXCERPT_CHARS`] at a character boundary.
    #[must_use]
    pub fn new(step: Option<u64>, field: impl Into<String>, text: &str) -> Self {
        let excerpt = match text.char_indices().nth(EXCERPT_CHARS) {
            Some((end, _)) => format!("{}...", &text[..end]),
            None => text.to_string(),
        };
        Self {
            step,
            field: field.into(),
            excerpt,
        }
    }
}

/// A cause a model proposed. It is never the diagnosis: it is kept beside it,
/// labelled with its source.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hypothesis {
    /// The class proposed.
    pub class: FailureClass,
    /// Who proposed it (`model:<identity>`).
    pub source: String,
    /// Why, in the proposer's words.
    pub rationale: String,
}

/// Why an attempt failed, as a deterministic diagnosis decided it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticResult {
    /// The class.
    pub class: FailureClass,
    /// Who the failure is about; `None` when unclassified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<Attribution>,
    /// The rule that assigned the class, by name.
    pub rule: String,
    /// The step where the decisive error was made, when the evidence places
    /// one: the first step a repair must not keep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decisive_step: Option<u64>,
    /// What the rule read.
    pub evidence: Vec<EvidenceSpan>,
    /// Causes a model proposed; never part of the diagnosis.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hypotheses: Vec<Hypothesis>,
}

impl DiagnosticResult {
    /// `class` assigned by `rule` on `evidence`, the decisive error at
    /// `decisive_step`.
    #[must_use]
    pub fn new(
        class: FailureClass,
        rule: impl Into<String>,
        decisive_step: Option<u64>,
        evidence: Vec<EvidenceSpan>,
    ) -> Self {
        Self {
            class,
            attribution: class.attribution(),
            rule: rule.into(),
            decisive_step,
            evidence,
            hypotheses: Vec::new(),
        }
    }

    /// The last good step: the one before the decisive error, where a repair
    /// forks. `None` when no decisive step is placed, or it is the first.
    #[must_use]
    pub fn last_good_step(&self) -> Option<u64> {
        self.decisive_step
            .and_then(|step| step.checked_sub(1))
            .filter(|step| *step > 0)
    }

    /// Adds a model's hypothesis beside the diagnosis. The class, the rule,
    /// the decisive step and the evidence stay as the deterministic rules
    /// decided them.
    pub fn add_hypothesis(&mut self, hypothesis: Hypothesis) {
        self.hypotheses.push(hypothesis);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hypothesis_never_changes_the_diagnosis() {
        let mut result = DiagnosticResult::new(
            FailureClass::Gateway,
            "gateway_status",
            None,
            vec![EvidenceSpan::new(None, "gateway.status", "504")],
        );
        let before = result.clone();
        result.add_hypothesis(Hypothesis {
            class: FailureClass::WrongContent,
            source: "model:judge".into(),
            rationale: "the answer looks wrong".into(),
        });
        assert_eq!(result.class, before.class);
        assert_eq!(result.attribution, Some(Attribution::Infrastructure));
        assert_eq!(result.evidence, before.evidence);
        assert_eq!(result.hypotheses.len(), 1);
    }

    #[test]
    fn the_last_good_step_is_the_one_before_the_decisive_error() {
        let at = |step| DiagnosticResult::new(FailureClass::Tool, "r", step, Vec::new());
        assert_eq!(at(Some(5)).last_good_step(), Some(4));
        assert_eq!(at(Some(1)).last_good_step(), None);
        assert_eq!(at(None).last_good_step(), None);
    }

    #[test]
    fn an_excerpt_is_cut_at_a_character_boundary() {
        let long = "å".repeat(EXCERPT_CHARS + 5);
        let span = EvidenceSpan::new(Some(2), "observation", &long);
        assert_eq!(span.excerpt.chars().count(), EXCERPT_CHARS + 3);
    }
}
