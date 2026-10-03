// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The verifier view: a task and a candidate answer, classified by the
//! decision its verdicts add up to.
//!
//! Each experience is a candidate. It yields one record when its verdicts
//! decide (pass or fail) at the minimum strength or above and it has a
//! final output: the user turn is the student's turn (see [`Strip`]), the
//! answer under `Candidate answer:`, and - when an executable verdict is
//! among its annotations - a summary of how the checks ran under
//! `Execution evidence:` (the lab's
//! `splinter_lab::verifiers::executable::evidence_summary`: runtime and
//! how each run ended, never digests or privileged checks); the supervised
//! reply is [`VerifierView::PASS`] or [`VerifierView::FAIL`].
//!
//! The objective is [`Objective::Classification`], rendered as SFT for
//! now: the label is the supervised assistant turn, so brain's chat
//! fine-tuning trains it without a classification head, and the writer
//! writes it as `generic-messages-v2`.

use splinter_core::annotation::{AnnotationBody, Strength};
use splinter_lab::verifiers::executable;

use crate::render::{message, student_turn, with_candidate, EXECUTION_HEADING};
use crate::{
    Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, Strip, View, ViewError,
};

/// The name every verifier record carries.
const NAME: &str = "verifier";

/// Pass/fail classification of candidate answers.
#[derive(Clone, Debug)]
pub struct VerifierView {
    min_strength: Strength,
    strip: Strip,
}

impl VerifierView {
    /// The label of an answer decided pass.
    pub const PASS: &'static str = "pass";

    /// The label of an answer decided fail.
    pub const FAIL: &'static str = "fail";

    /// The view, accepting decisions at `min_strength` or stronger,
    /// showing the student only the instruction and the answer.
    #[must_use]
    pub fn new(min_strength: Strength) -> Self {
        Self {
            min_strength,
            strip: Strip::default(),
        }
    }

    /// The same view under `strip`.
    #[must_use]
    pub fn with_strip(self, strip: Strip) -> Self {
        Self { strip, ..self }
    }

    fn record(&self, entry: &Entry) -> Result<(RecordBody, Provenance), Exclusion> {
        let decision = entry.decision().ok_or(Exclusion::Undecided)?;
        if decision.strength < self.min_strength {
            return Err(Exclusion::TooWeak);
        }
        let answer = entry
            .experience
            .final_output
            .as_deref()
            .ok_or(Exclusion::NoFinalOutput)?;
        let mut input = with_candidate(&student_turn(entry, &self.strip)?, answer);
        let execution: Vec<String> = entry
            .notes
            .iter()
            .filter(|note| note.producer.name == executable::PRODUCER)
            .filter_map(|note| match &note.body {
                AnnotationBody::Verdict {
                    strength: Strength::Executable,
                    evidence,
                    ..
                } => executable::evidence_summary(evidence),
                _ => None,
            })
            .collect();
        if !execution.is_empty() {
            input.push_str(&format!(
                "\n\n{EXECUTION_HEADING}\n{}",
                execution.join("\n")
            ));
        }
        let label = if decision.passed {
            Self::PASS
        } else {
            Self::FAIL
        };
        let messages = vec![
            message("user", &input, false),
            message("assistant", label, true),
        ];
        Ok((
            RecordBody::Chat { messages },
            Provenance::of(vec![entry.id.clone()]),
        ))
    }
}

impl View for VerifierView {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Classification
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            NAME,
            Objective::Classification,
            Some(self.strip.clone()),
            Some(self.min_strength),
        );
        for entry in corpus.entries() {
            projection.take(self.record(entry));
        }
        Ok(projection)
    }
}
