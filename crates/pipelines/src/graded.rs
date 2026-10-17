// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements auditable evaluation of language models, for
// its clients. If your team needs expertise in keeping every graded answer of
// a run so that a grader can be changed and the run regraded without asking
// the model again, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The graded items of a run report.
//!
//! A report keeps every item a run graded, whatever its verdict: the task,
//! the model's answer, the verdict and the grader's reason. A run that kept
//! only the failures could not be regraded (a stricter or looser grader, a
//! fixed normalisation) without asking the model again.
//!
//! Size is bounded, never silently: an answer over [`ANSWER_INLINE_BYTES`]
//! is cut at a character boundary, the item says so (`truncated`, with the
//! answer's full size), and the full text is kept in the store under the
//! address the item names ([`stored_answer`] reads it back). A reason over
//! [`REASON_BYTES`] is cut and ends in [`TRUNCATED`].

use serde::{Deserialize, Serialize};
use splinter_core::annotation::{Annotation, AnnotationBody};
use splinter_core::digest::Digest;
use splinter_core::experience::Task;

use crate::release::probe::Probe;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The class of the stored full text of an answer a report cut.
const ANSWER_CLASS: &str = "graded_answer";

/// The most bytes of an answer a report holds inline.
pub const ANSWER_INLINE_BYTES: usize = 8 * 1024;

/// The most bytes of a grader's reason a report holds.
pub const REASON_BYTES: usize = 2 * 1024;

/// What a cut reason ends in.
pub const TRUNCATED: &str = "[truncated]";

/// How a grader decided an item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Decided right.
    Passed,
    /// Decided wrong.
    Failed,
    /// No grader decided it.
    Undecided,
}

impl Verdict {
    /// The verdict of a probe's outcome.
    #[must_use]
    pub fn of(decided: Option<bool>) -> Self {
        match decided {
            Some(true) => Self::Passed,
            Some(false) => Self::Failed,
            None => Self::Undecided,
        }
    }
}

/// A model's answer as a report keeps it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeptAnswer {
    /// The answer, or its first bytes when `truncated`.
    pub text: String,
    /// The answer's full size in bytes.
    pub bytes: usize,
    /// Whether `text` is shorter than the answer.
    pub truncated: bool,
    /// Where the full text is kept ([`stored_answer`]); set when
    /// `truncated`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stored: Option<Digest>,
}

/// One item a run graded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradedItem {
    /// The task's address.
    pub task: Digest,
    /// Its kind.
    pub kind: String,
    /// What was asked.
    pub instruction: String,
    /// What the model answered; `None` when its run ended without an answer.
    pub answer: Option<KeptAnswer>,
    /// How it was decided.
    pub verdict: Verdict,
    /// Why the grader decided so, when it said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct StoredAnswer {
    text: String,
}

impl GradedItem {
    /// The item `probe` is for `task`, its answer kept inline or by address
    /// as the module says.
    pub fn keep(ctx: &Context, task: &Task, probe: &Probe) -> Result<Self, OrchestratorError> {
        let answer = probe
            .answer
            .as_deref()
            .map(|text| keep_answer(ctx, text))
            .transpose()?;
        Ok(Self {
            task: task.task.id.clone(),
            kind: task.task.kind.clone(),
            instruction: task.instruction.clone(),
            answer,
            verdict: Verdict::of(probe.verdict),
            reason: probe.reason.as_deref().map(cut_reason),
        })
    }
}

/// The full text of an answer a report cut, by the address its item names.
pub fn stored_answer(ctx: &Context, at: &Digest) -> Result<String, OrchestratorError> {
    ctx.workspace()
        .get_document::<StoredAnswer>(ANSWER_CLASS, at)?
        .map(|stored| stored.text)
        .ok_or_else(|| OrchestratorError::Refused(format!("no answer is stored at {at}")))
}

fn keep_answer(ctx: &Context, text: &str) -> Result<KeptAnswer, OrchestratorError> {
    if text.len() <= ANSWER_INLINE_BYTES {
        return Ok(KeptAnswer {
            text: text.to_string(),
            bytes: text.len(),
            truncated: false,
            stored: None,
        });
    }
    let stored = ctx.workspace().put_document(
        ANSWER_CLASS,
        &StoredAnswer {
            text: text.to_string(),
        },
    )?;
    Ok(KeptAnswer {
        text: prefix(text, ANSWER_INLINE_BYTES).to_string(),
        bytes: text.len(),
        truncated: true,
        stored: Some(stored),
    })
}

fn cut_reason(reason: &str) -> String {
    if reason.len() <= REASON_BYTES {
        return reason.to_string();
    }
    format!("{}{TRUNCATED}", prefix(reason, REASON_BYTES))
}

/// The longest prefix of `text` of at most `bytes` bytes that ends at a
/// character boundary.
fn prefix(text: &str, bytes: usize) -> &str {
    let mut end = bytes.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// What the verdicts among `notes` say, producer by producer: the reason a
/// verifier gave, else why it abstained.
pub(crate) fn reason_of(notes: &[Annotation]) -> Option<String> {
    let said: Vec<String> = notes
        .iter()
        .filter_map(|note| {
            let AnnotationBody::Verdict { evidence, .. } = &note.body else {
                return None;
            };
            let text = |key: &str| evidence.get(key).and_then(|v| v.as_str());
            let why = text("reason").or_else(|| text("abstained"))?;
            Some(match text("detail") {
                Some(detail) => format!("{}: {why} ({detail})", note.producer.name),
                None => format!("{}: {why}", note.producer.name),
            })
        })
        .collect();
    (!said.is_empty()).then(|| said.join("; "))
}
