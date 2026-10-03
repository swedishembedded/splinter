// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any source into verifiable training tasks, for its clients. If your team
// needs expertise in synthetic task generation or verifier design, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Model-driven task generation: a generator model proposes tasks of a
//! [`TaskKind`] from a source's sections, and code decides which are
//! admitted.
//!
//! [`ModelTaskGenerator::generate`] asks the model once per kind,
//! closed-book through the same solve every task runs through, for tasks
//! in one strict JSON shape, which the prompt spells out. Nothing the model says about a
//! task is trusted; each candidate is admitted only when:
//!
//! 1. its reply parses, and it is within the requested count;
//! 2. its evidence cites sections it was shown, each quote is found
//!    verbatim in its section, and the evidence spans the kind's
//!    `min_sections` distinct sections; every span names the source part
//!    ([`Span::in_part`](splinter_record::experience::Span::in_part)) and
//!    resolves through the source store;
//! 3. it carries the material its kind requires, and an open-book kind's
//!    material is in the instruction verbatim;
//! 4. its instruction stands on its own
//!    ([`splinter_views::check_self_contained`]): for a closed-book kind the
//!    evidence counts as material the student does not see;
//! 5. it names its subject, when its kind must
//!    ([`TaskKind::names_subject`]): `subject` is the product, document,
//!    tool, component or version the question is about, it is in the
//!    instruction (case and whitespace aside), and the source's identity
//!    ([`SourceIdentity`]) or a cited section names it. A question that
//!    does not say what it is about has a different answer for every
//!    product, and a model trained on one of them answers it wrongly for
//!    the rest;
//! 6. it is new to the batch: not the same instruction once lower-cased
//!    with whitespace collapsed, and not overlapping an admitted one's word
//!    shingles at [`GenerationPolicy::max_overlap`] or more ([`dedup`], the
//!    one near-duplicate rule, which training-set selection reuses). A
//!    repeat of an admitted question of the same subject whose grounded
//!    reference disagrees with it is a contradiction: both are refused
//!    ([`dedup::contradicts`], for kinds with one exact answer,
//!    [`TaskKind::exact_answer`]);
//! 7. it is grounded: a text answer by the evidence's words
//!    ([`grounding`]); a computed answer by running it - a program
//!    reference must pass the task's executable checks in the sandbox and
//!    every generated test must be admitted by mutation validation, and
//!    shown code must print the reference output.
//!
//! What was admitted and what was rejected, by [`Rejection`] and by kind,
//! is the [`GenerationReport`].

pub mod generator;
pub mod grounding;
pub mod kind;

pub mod dedup;
pub mod variants;

mod admit;
mod reply;

use std::collections::BTreeMap;
use std::time::Duration;

use serde::Serialize;
use splinter_record::digest::Digest;
use splinter_record::experience::Task;

pub use generator::{GenerateError, ModelTaskGenerator, SourceIdentity, SourceText};
pub use kind::{
    AnswerForm, Catalogue, Focus, KindError, Material, SolverEnvironment, TaskKind, VerifierKind,
};
pub use variants::{can_vary, DEFAULT_VARIANTS_PER_TASK};

/// The generator's name in a task's provenance, before the model identity.
pub const GENERATOR: &str = "splinter-knowledge/model-tasks@1";

/// The most tasks one request asks for, by default.
pub const DEFAULT_TASKS_PER_REQUEST: usize = 8;

/// The share of a text reference's content words its evidence must carry,
/// by default.
pub const DEFAULT_MIN_SUPPORT: f64 = 0.8;

/// Words in a shingle, by default.
pub const DEFAULT_SHINGLE_WORDS: usize = 3;

/// The shingle overlap at which an instruction is a near duplicate, by
/// default.
pub const DEFAULT_MAX_OVERLAP: f64 = 0.8;

/// How many times a malformed reply is sent back for correction, by
/// default.
pub const DEFAULT_REPAIRS: u32 = 1;

/// How long one request to the generator model may take, by default.
pub const DEFAULT_REQUEST_DEADLINE: Duration = Duration::from_secs(600);

/// The output tokens one request may generate: room for a whole batch of
/// tasks. A request that names no budget gets its model's own default, which
/// for a local model is a few hundred tokens and cuts a batch off mid-reply.
pub const DEFAULT_REQUEST_MAX_OUTPUT_TOKENS: u64 = 6144;

/// The parameters of generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationPolicy {
    /// The most tasks one request asks for; further tasks in a reply are
    /// rejected.
    pub tasks_per_request: usize,
    /// The share of a text reference's content words its evidence must
    /// carry, in `[0, 1]`.
    pub min_support: f64,
    /// Words in a shingle; at least one.
    pub shingle_words: usize,
    /// The shingle overlap at which an instruction is a near duplicate, in
    /// `(0, 1]`.
    pub max_overlap: f64,
    /// How long one request may take.
    pub deadline: Duration,
    /// Output tokens one request may generate; `None` leaves it to the
    /// model's own default.
    pub max_output_tokens: Option<u64>,
    /// How many times a reply that is not the shape is sent back to the
    /// model for correction, within the request's deadline and budget.
    pub repairs: u32,
}

impl Default for GenerationPolicy {
    /// The `DEFAULT_*` constants.
    fn default() -> Self {
        Self {
            tasks_per_request: DEFAULT_TASKS_PER_REQUEST,
            min_support: DEFAULT_MIN_SUPPORT,
            shingle_words: DEFAULT_SHINGLE_WORDS,
            max_overlap: DEFAULT_MAX_OVERLAP,
            deadline: DEFAULT_REQUEST_DEADLINE,
            max_output_tokens: Some(DEFAULT_REQUEST_MAX_OUTPUT_TOKENS),
            repairs: DEFAULT_REPAIRS,
        }
    }
}

impl GenerationPolicy {
    /// Refuses a parameter out of its range.
    pub fn validate(&self) -> Result<(), GenerateError> {
        let refuse =
            |name: &'static str, reason: String| Err(GenerateError::Parameter { name, reason });
        if self.tasks_per_request == 0 {
            return refuse("tasks_per_request", "at least one task per request".into());
        }
        if !(0.0..=1.0).contains(&self.min_support) {
            return refuse(
                "min_support",
                format!("{} is not in [0, 1]", self.min_support),
            );
        }
        if self.shingle_words == 0 {
            return refuse("shingle_words", "a shingle has at least one word".into());
        }
        if !(self.max_overlap > 0.0 && self.max_overlap <= 1.0) {
            return refuse(
                "max_overlap",
                format!("{} is not in (0, 1]", self.max_overlap),
            );
        }
        Ok(())
    }
}

/// Why a proposed task was not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Rejection {
    /// The reply is not the required shape (the whole reply counts once).
    Malformed,
    /// The model run ended without a reply (a deadline, a budget).
    NoReply,
    /// More tasks than were asked for.
    OverCount,
    /// Evidence cites a section the model was not shown, or none at all.
    UnknownSection,
    /// A quote is not found verbatim in the section it cites.
    QuoteNotFound,
    /// The evidence spans fewer distinct sections than the kind needs.
    TooFewSections,
    /// An evidence span does not resolve through the source store.
    Unresolved,
    /// Material the kind requires is missing, or a check expects nothing.
    MissingMaterial,
    /// An open-book kind's material is not in the instruction, or a
    /// closed-book kind's candidate carries material to show.
    MaterialNotShown,
    /// The instruction does not stand on its own.
    NotSelfContained,
    /// A text reference is not supported by its evidence.
    Ungrounded,
    /// A kind whose reference is the source author's own words has a
    /// reference that is not a passage of its evidence word for word.
    NotQuoted,
    /// A computed reference does not pass its own checks.
    ChecksFailed,
    /// A generated test fails the reference or survives its mutants.
    TestNotAdmitted,
    /// The instruction repeats an admitted one.
    Duplicate,
    /// The instruction nearly repeats an admitted one.
    NearDuplicate,
    /// A variant's reference is not the original's: it asks about another
    /// fact.
    ReferenceChanged,
    /// The instruction does not name what it is about: no subject, a
    /// subject the instruction does not contain, or one neither the source
    /// nor the cited sections name. A variant must keep the original's.
    NoSubject,
    /// It asks an admitted question of the same subject and gives another
    /// answer; both are refused, since at least one of them is wrong.
    Contradiction,
    /// The task fails the store's validation (an empty instruction).
    Invalid,
}

/// A task a generator admitted.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedTask {
    /// The task.
    pub task: Task,
    /// What it is about, as its instruction names it; `None` for a kind
    /// that names no subject ([`TaskKind::names_subject`]).
    pub subject: Option<String>,
    /// Who generated it: [`GENERATOR`] and the model identity, as an
    /// experience's provenance records a generator.
    pub generator: String,
    /// The digest of the prompt the model was sent.
    pub prompt: Digest,
}

/// A proposed task that was not admitted.
#[derive(Clone, Debug, PartialEq)]
pub struct RejectedTask {
    /// The kind it was proposed as.
    pub kind: String,
    /// Its position in the reply (0 for a whole reply).
    pub index: usize,
    /// Why.
    pub reason: Rejection,
    /// What exactly failed.
    pub detail: String,
}

/// Admitted and rejected tasks of one kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KindCounts {
    /// Tasks admitted.
    pub admitted: usize,
    /// Proposals rejected (a malformed reply counts once).
    pub rejected: usize,
}

/// What one batch of generation produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationReport {
    /// The admitted tasks, in the order they were proposed.
    pub admitted: Vec<GeneratedTask>,
    /// Rejections counted by reason, in reason order.
    pub rejected: Vec<(Rejection, usize)>,
    /// Counts per kind name.
    pub per_kind: BTreeMap<String, KindCounts>,
    /// Every rejection, with what failed.
    pub rejections: Vec<RejectedTask>,
}

impl GenerationReport {
    /// How many proposals were rejected for `reason`.
    #[must_use]
    pub fn count(&self, reason: Rejection) -> usize {
        self.rejected
            .iter()
            .find(|(r, _)| *r == reason)
            .map_or(0, |(_, n)| *n)
    }

    fn admit(&mut self, kind: &str, task: GeneratedTask) {
        self.per_kind.entry(kind.to_string()).or_default().admitted += 1;
        self.admitted.push(task);
    }

    /// Takes back the admitted task `id`, proposed at `index` of a reply
    /// for `kind`, now refused for `reason`.
    fn withdraw(
        &mut self,
        kind: &str,
        index: usize,
        id: &Digest,
        reason: Rejection,
        detail: String,
    ) {
        let Some(at) = self.admitted.iter().position(|g| &g.task.task.id == id) else {
            return;
        };
        self.admitted.remove(at);
        self.per_kind.entry(kind.to_string()).or_default().admitted -= 1;
        self.reject(kind, index, reason, detail);
    }

    fn reject(&mut self, kind: &str, index: usize, reason: Rejection, detail: String) {
        self.per_kind.entry(kind.to_string()).or_default().rejected += 1;
        match self.rejected.binary_search_by_key(&reason, |(r, _)| *r) {
            Ok(at) => self.rejected[at].1 += 1,
            Err(at) => self.rejected.insert(at, (reason, 1)),
        }
        self.rejections.push(RejectedTask {
            kind: kind.to_string(),
            index,
            reason,
            detail,
        });
    }
}
