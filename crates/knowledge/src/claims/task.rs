// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A live claim as a task.
//!
//! The task asks the claim's question closed-book. Its reference is the
//! claim's statement, graded by [`super::answer`]; its evidence is the spans
//! of the session the person's words are in, each resolved to its bytes and
//! checked against the quote before the task is made; and the statement is
//! also a passage only a teacher sees, so a teacher answers with the claim
//! in front of it and a student is trained on the question alone.
//!
//! Of the gates a generated task passes, the ones that concern a task made
//! from a claim apply: its question stands on its own without the material
//! the student is not shown, and it is no repeat of another question of the
//! batch ([`crate::tasks::dedup`]). That the statement's terms are in the
//! words it rests on is the claim gate's.

use splinter_core::claim::Claim;
use splinter_core::experience::{Environment, Privileged, PrivilegedKind, Span, Task};
use splinter_core::selfcontained::check_self_contained;
use splinter_store::error::StoreError;
use splinter_store::sources::SourceStore;

use super::terms::names_of;
use crate::tasks::grounding::STOPWORDS;
use crate::tasks::TAUGHT;

/// A task made from a claim, with what a variant must keep naming.
#[derive(Clone, Debug, PartialEq)]
pub struct ClaimTask {
    /// The task.
    pub task: Task,
    /// What the question is about, in the question's own words; `None` when
    /// the question names nothing the statement also names, and a variant
    /// would have nothing to hold it to.
    pub subject: Option<String>,
}

/// Why a claim could not be made a task.
#[derive(Debug, thiserror::Error)]
pub enum ClaimTaskError {
    /// A span of the claim does not hold the words it is said to.
    #[error("the span of the words {quote:?} of step {step} holds other bytes")]
    SpanMismatch {
        /// The step.
        step: u64,
        /// The words.
        quote: String,
    },
    /// The question points at material the student is not shown.
    #[error("the question does not stand on its own: {0}")]
    NotSelfContained(String),
    /// The task is malformed.
    #[error("the task is malformed: {0}")]
    Task(#[from] splinter_core::experience::ExperienceError),
    /// A span could not be read.
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// The task `claim` is: see the module documentation.
pub fn claim_task(store: &SourceStore, claim: &Claim) -> Result<ClaimTask, ClaimTaskError> {
    let mut evidence: Vec<Span> = Vec::new();
    for quote in claim.quotes.iter().chain(&claim.observations) {
        let bytes = store.read_span(&quote.span)?;
        if bytes != quote.text.as_bytes() {
            return Err(ClaimTaskError::SpanMismatch {
                step: quote.step,
                quote: quote.text.clone(),
            });
        }
        if !evidence.contains(&quote.span) {
            evidence.push(quote.span.clone());
        }
    }
    let statement = Privileged {
        kind: PrivilegedKind::Passage,
        content: claim.statement.clone(),
        span: None,
    };
    check_self_contained(&claim.question, &[&statement])
        .map_err(|e| ClaimTaskError::NotSelfContained(e.to_string()))?;
    let reference = Privileged {
        kind: PrivilegedKind::Reference,
        content: claim.statement.clone(),
        span: match evidence.as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        },
    };
    let task = Task::new(
        TAUGHT,
        evidence,
        Environment::closed_book(),
        claim.question.clone(),
        vec![reference, statement],
    )?;
    Ok(ClaimTask {
        task,
        subject: subject_of(&claim.question, &claim.statement),
    })
}

/// Words too common to be what a question is about, besides
/// [`STOPWORDS`].
const QUESTION_WORDS: &[&str] = &[
    "which",
    "whose",
    "whom",
    "should",
    "could",
    "tell",
    "name",
    "give",
    "remind",
    "currently",
    "again",
    "still",
    "many",
    "much",
    "your",
    "mine",
    "ours",
    "theirs",
    "using",
    "used",
];

/// What `question` is about: the first name or quoted term of it that the
/// statement also states, else the longest of its words that the statement
/// also has (the first, among equals); `None` when they share none.
#[must_use]
pub fn subject_of(question: &str, statement: &str) -> Option<String> {
    let statement_words: Vec<String> = words(statement);
    let stated = |term: &str| statement_words.contains(&term.to_lowercase());
    if let Some(name) = names_of(question).into_iter().find(|n| stated(n)) {
        return Some(name);
    }
    let mut best: Option<&str> = None;
    for word in question
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 4)
    {
        let lowered = word.to_lowercase();
        if STOPWORDS.contains(&lowered.as_str()) || QUESTION_WORDS.contains(&lowered.as_str()) {
            continue;
        }
        if stated(word) && best.is_none_or(|b| word.chars().count() > b.chars().count()) {
            best = Some(word);
        }
    }
    best.map(str::to_string)
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}
