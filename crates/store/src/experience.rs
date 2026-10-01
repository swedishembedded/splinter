// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experience records that every training set
// is projected from, for its clients. If your team needs expertise in
// training-data provenance or agent learning pipelines, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The experience: one task, what the solver did with it, and everything
//! needed to judge and learn from it later.
//!
//! An [`Experience`] is immutable and content-addressed: its
//! [`ExperienceId`] is the [`Digest`] of the canonical JSON of the whole
//! record (see [`crate::digest`] for the canonical form). Everything in the
//! record is part of the address, the provenance timestamp included, so two
//! solves of the same task at different times are two experiences. What a
//! grader later says about an experience is an annotation
//! ([`crate::annotation`]), appended beside it, never written into it.
//!
//! The record separates what the student sees from what only the teacher
//! saw: [`Experience::instruction`] is the user turn a training record may
//! show; [`Experience::privileged`] (a reference answer, a hint, the source
//! passage) is for generators and verifiers, and a view must never put it
//! into a record. A teacher's solve ([`Provenance::teacher`]) was prompted
//! with the task's grounding material; its experience still records the
//! task's own instruction, so the record a view makes of it is the
//! student's: closed-book.

use serde::{Deserialize, Serialize};

pub use crate::digest::{Digest, DigestError};
pub use crate::source::PartRef;
use crate::{clock::Clock, digest::canonical_json};

/// The content address of an [`Experience`]: the [`Digest`] of its
/// canonical form.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExperienceId(pub Digest);

impl ExperienceId {
    /// `sha256:<hex>`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// The hex part alone, which names the experience's files.
    #[must_use]
    pub fn hex(&self) -> &str {
        self.0.hex()
    }
}

impl std::fmt::Display for ExperienceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Which task an experience attempted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskRef {
    /// The content address of the task: the [`Digest`] of the canonical
    /// form of its kind, evidence, environment, instruction and privileged
    /// information (see [`Task::new`]).
    pub id: Digest,
    /// What kind of task it is, free-form: `denoise`, `recall`, ...
    pub kind: String,
}

/// A byte range `[start, end)` of the bytes a task is grounded in.
///
/// The offsets index the content whose digest is [`Span::source`]; with a
/// source store, that is a source part's content (see
/// [`crate::source::Part::content`]), and the store resolves the span to
/// its bytes. [`Span::part`] names the source and part those bytes are, so a
/// span traces back to where they were captured and not only to what they
/// are: one content can belong to many sources. It is optional, and left
/// out of the canonical form when unset, so a span that names no part has
/// the same address whether or not the reader knows the field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    /// The digest of the content the offsets index into.
    pub source: Digest,
    /// First byte of the span.
    pub start: u64,
    /// One past the last byte of the span; `start <= end`.
    pub end: u64,
    /// The source part whose content [`Span::source`] is, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<PartRef>,
}

impl Span {
    /// A span of the content `source`, refused when it ends before it
    /// starts.
    pub fn new(source: Digest, start: u64, end: u64) -> Result<Self, ExperienceError> {
        let span = Self {
            source,
            start,
            end,
            part: None,
        };
        span.validate()?;
        Ok(span)
    }

    /// A span of `part`, whose content is `content`, refused when it ends
    /// before it starts.
    pub fn in_part(
        part: PartRef,
        content: Digest,
        start: u64,
        end: u64,
    ) -> Result<Self, ExperienceError> {
        let span = Self {
            source: content,
            start,
            end,
            part: Some(part),
        };
        span.validate()?;
        Ok(span)
    }

    fn validate(&self) -> Result<(), ExperienceError> {
        if self.start > self.end {
            return Err(ExperienceError::Span {
                source_digest: self.source.clone(),
                start: self.start,
                end: self.end,
            });
        }
        Ok(())
    }
}

/// Where the solver worked.
///
/// The spec records everything that determines how the environment
/// behaves, and [`Environment::snapshot`] is its digest: two environments
/// with the same snapshot behave the same, so a solve can be replayed in
/// the environment a snapshot names. [`Environment::new`] computes it; a
/// recorded snapshot that does not match the kind and spec is refused.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Environment {
    /// [`Environment::CLOSED_BOOK`] (no tools), or `runtime:<name>` (one
    /// tool that runs code in the named runtime).
    pub kind: String,
    /// Everything that determines the environment's behaviour, as its
    /// producer defines it.
    pub spec: serde_json::Value,
    /// The [`Digest`] of the canonical form of `{kind, spec}`; absent only
    /// in a record written before environments carried one.
    pub snapshot: Option<Digest>,
}

/// The fields an environment's snapshot is computed over.
#[derive(Serialize)]
struct EnvironmentBody<'a> {
    kind: &'a str,
    spec: &'a serde_json::Value,
}

impl Environment {
    /// The kind of an environment with no tools at all.
    pub const CLOSED_BOOK: &'static str = "closed-book";

    /// An environment of `kind` whose behaviour `spec` determines, with its
    /// snapshot computed.
    #[must_use]
    pub fn new(kind: impl Into<String>, spec: serde_json::Value) -> Self {
        let kind = kind.into();
        let snapshot = Some(Self::snapshot_of(&kind, &spec));
        Self {
            kind,
            spec,
            snapshot,
        }
    }

    /// The closed-book environment: no tools, nothing to configure, so its
    /// spec is empty and its snapshot the digest of that.
    #[must_use]
    pub fn closed_book() -> Self {
        Self::new(Self::CLOSED_BOOK, serde_json::json!({}))
    }

    /// The snapshot an environment of `kind` with `spec` has.
    #[must_use]
    pub fn snapshot_of(kind: &str, spec: &serde_json::Value) -> Digest {
        match canonical_json(&EnvironmentBody { kind, spec }) {
            Ok(bytes) => Digest::of(&bytes),
            // A string and a JSON value always serialize: a
            // `serde_json::Value` cannot hold a non-finite float.
            Err(e) => unreachable!("an environment body serializes: {e}"),
        }
    }
}

/// What kind of privileged information an item is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivilegedKind {
    /// Source text the task was generated from.
    Passage,
    /// A hint the teacher had and the student does not.
    Hint,
    /// A critique of an attempt.
    Critique,
    /// The reference answer.
    Reference,
    /// An oracle's output, such as a tool's ground truth.
    Oracle,
    /// Anything else, named by its producer.
    Other(String),
}

/// Information only the teacher saw: a generator's or verifier's input,
/// never part of what the student is trained to see.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Privileged {
    /// What it is.
    pub kind: PrivilegedKind,
    /// The information itself.
    pub content: String,
    /// Where in a source it came from, when it came from one.
    pub span: Option<Span>,
}

/// Who and what produced an experience, and when.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The model that solved the task, as its identity string.
    pub solver: String,
    /// The generator that produced the task, when one did.
    pub generator: Option<String>,
    /// The policy (adapter or checkpoint identity) the solver served from,
    /// when it differs from the solver's own name.
    pub policy: Option<String>,
    /// Digests of the prompts involved in producing the experience.
    pub prompt_digests: Vec<Digest>,
    /// Which of several solves of one task by one solver in one pass this
    /// is, counted from zero, when the task was solved more than once (a
    /// pass@k measurement); left out of the canonical form when unset, so
    /// a single solve has the same address whether or not the reader knows
    /// the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt: Option<u32>,
    /// Whether the solver was the teacher: prompted open-book, with the
    /// task's grounding material (its evidence passages and hints) beside
    /// the instruction, where a student sees the instruction alone. A
    /// teacher's verified answer is training data for the student; it says
    /// nothing about what the student knows closed-book. Left out of the
    /// canonical form when false, so a student's solve has the same address
    /// whether or not the reader knows the field.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub teacher: bool,
    /// When the experience was recorded, from the injected [`Clock`].
    pub created_at: String,
}

impl Provenance {
    /// Provenance for `solver`, stamped by `clock`, with nothing else known.
    #[must_use]
    pub fn new(solver: impl Into<String>, clock: &dyn Clock) -> Self {
        Self {
            solver: solver.into(),
            generator: None,
            policy: None,
            prompt_digests: Vec::new(),
            attempt: None,
            teacher: false,
            created_at: clock.utc_now(),
        }
    }
}

/// A task as a generator emits it, before anyone solves it: everything an
/// [`Experience`] records about the task itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Task {
    /// The task's kind and content address.
    pub task: TaskRef,
    /// The spans the task is grounded in.
    pub evidence: Vec<Span>,
    /// Where the solver works.
    pub environment: Environment,
    /// The user turn the student sees.
    pub instruction: String,
    /// What only the teacher sees.
    pub privileged: Vec<Privileged>,
    /// The concepts the task declares it exercises, by name; empty when it
    /// declares none and its concepts are derived from its kind and
    /// evidence. Left out of the canonical form when empty, so a task that
    /// declares none has the same address whether or not the reader knows
    /// the field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concepts: Vec<String>,
}

/// The fields a task's address is computed over, in one place so building
/// a task and validating an experience cannot disagree about them.
#[derive(Serialize)]
struct TaskBody<'a> {
    kind: &'a str,
    evidence: &'a [Span],
    environment: &'a Environment,
    instruction: &'a str,
    privileged: &'a [Privileged],
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    concepts: &'a [String],
}

impl TaskBody<'_> {
    fn address(&self) -> Result<Digest, ExperienceError> {
        Ok(Digest::of(&canonical_json(self)?))
    }
}

impl Task {
    /// A validated task whose [`TaskRef::id`] addresses its content.
    pub fn new(
        kind: impl Into<String>,
        evidence: Vec<Span>,
        environment: Environment,
        instruction: impl Into<String>,
        privileged: Vec<Privileged>,
    ) -> Result<Self, ExperienceError> {
        Self::build(
            kind.into(),
            evidence,
            environment,
            instruction.into(),
            privileged,
            Vec::new(),
        )
    }

    fn build(
        kind: String,
        evidence: Vec<Span>,
        environment: Environment,
        instruction: String,
        privileged: Vec<Privileged>,
        concepts: Vec<String>,
    ) -> Result<Self, ExperienceError> {
        let id = TaskBody {
            kind: &kind,
            evidence: &evidence,
            environment: &environment,
            instruction: &instruction,
            privileged: &privileged,
            concepts: &concepts,
        }
        .address()?;
        let task = Self {
            task: TaskRef { id, kind },
            evidence,
            environment,
            instruction,
            privileged,
            concepts,
        };
        task.validate()?;
        Ok(task)
    }

    /// This task declaring that it exercises `concepts` (each named once,
    /// in order, blank names refused): a new task, with its own address.
    pub fn with_concepts(
        &self,
        concepts: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, ExperienceError> {
        let mut declared: Vec<String> = Vec::new();
        for concept in concepts {
            let concept = concept.into();
            if concept.trim().is_empty() {
                return Err(ExperienceError::Missing("concepts[]"));
            }
            if !declared.contains(&concept) {
                declared.push(concept);
            }
        }
        Self::build(
            self.task.kind.clone(),
            self.evidence.clone(),
            self.environment.clone(),
            self.instruction.clone(),
            self.privileged.clone(),
            declared,
        )
    }

    /// Checks what [`Task::new`] checks on a task that was not built by it
    /// (one read back from storage): the required text fields, spans that
    /// end after they start, an environment snapshot that matches, and a
    /// task reference that addresses this task.
    pub fn validate(&self) -> Result<(), ExperienceError> {
        validate_task(
            &self.task,
            &self.evidence,
            &self.environment,
            &self.instruction,
            &self.privileged,
            &self.concepts,
        )
    }

    /// This task asking `instruction` instead, everything else kept - the
    /// kind, evidence, environment, reference and other privileged items,
    /// declared concepts - so the answer that grades one grades the other:
    /// a new task, with its own address.
    pub fn with_instruction(
        &self,
        instruction: impl Into<String>,
    ) -> Result<Self, ExperienceError> {
        Self::build(
            self.task.kind.clone(),
            self.evidence.clone(),
            self.environment.clone(),
            instruction.into(),
            self.privileged.clone(),
            self.concepts.clone(),
        )
    }

    /// This task without the critiques an earlier attempt's retry was
    /// handed: the task as it was generated, whose address a retry's task
    /// leaves. The same task when it carries none.
    pub fn without_critiques(&self) -> Result<Self, ExperienceError> {
        Self::build(
            self.task.kind.clone(),
            self.evidence.clone(),
            self.environment.clone(),
            self.instruction.clone(),
            self.privileged
                .iter()
                .filter(|p| p.kind != PrivilegedKind::Critique)
                .cloned()
                .collect(),
            self.concepts.clone(),
        )
    }

    /// This task with `item` added to what only the teacher sees: a new
    /// task, with its own address.
    pub fn with_privileged(&self, item: Privileged) -> Result<Self, ExperienceError> {
        let mut privileged = self.privileged.clone();
        privileged.push(item);
        Self::build(
            self.task.kind.clone(),
            self.evidence.clone(),
            self.environment.clone(),
            self.instruction.clone(),
            privileged,
            self.concepts.clone(),
        )
    }

    /// Whether `self` and `other` are one task apart from critiques: the
    /// same kind, evidence, environment, instruction and privileged items
    /// once every [`PrivilegedKind::Critique`] is set aside. A critique is
    /// feedback on an earlier attempt, handed to a retry; it changes
    /// neither what the task asks nor what grades an answer to it.
    #[must_use]
    pub fn same_apart_from_critiques(&self, other: &Task) -> bool {
        let graded = |task: &'_ Task| -> Vec<Privileged> {
            task.privileged
                .iter()
                .filter(|p| p.kind != PrivilegedKind::Critique)
                .cloned()
                .collect()
        };
        self.task.kind == other.task.kind
            && self.evidence == other.evidence
            && self.environment == other.environment
            && self.instruction == other.instruction
            && self.concepts == other.concepts
            && graded(self) == graded(other)
    }
}

/// One solved (or attempted) task, immutable once stored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Experience {
    /// The task's kind and content address.
    pub task: TaskRef,
    /// The spans the task is grounded in.
    pub evidence: Vec<Span>,
    /// Where the solver worked.
    pub environment: Environment,
    /// The user turn the student sees.
    pub instruction: String,
    /// What only the teacher saw.
    pub privileged: Vec<Privileged>,
    /// The concepts the task declares it exercises (see [`Task::concepts`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concepts: Vec<String>,
    /// What the solver did, as sven exports it.
    pub trajectory: sven_sdk::atif::Trajectory,
    /// The solver's final answer, when it gave one.
    pub final_output: Option<String>,
    /// Who produced it, and when.
    pub provenance: Provenance,
}

impl Experience {
    /// A validated experience of `task`.
    pub fn new(
        task: Task,
        trajectory: sven_sdk::atif::Trajectory,
        final_output: Option<String>,
        provenance: Provenance,
    ) -> Result<Self, ExperienceError> {
        let experience = Self {
            task: task.task,
            evidence: task.evidence,
            environment: task.environment,
            instruction: task.instruction,
            privileged: task.privileged,
            concepts: task.concepts,
            trajectory,
            final_output,
            provenance,
        };
        experience.validate()?;
        Ok(experience)
    }

    /// The task this experience attempted, as its generator emitted it.
    #[must_use]
    pub fn to_task(&self) -> Task {
        Task {
            task: self.task.clone(),
            evidence: self.evidence.clone(),
            environment: self.environment.clone(),
            instruction: self.instruction.clone(),
            privileged: self.privileged.clone(),
            concepts: self.concepts.clone(),
        }
    }

    /// The canonical form the id is computed over.
    pub fn canonical(&self) -> Result<Vec<u8>, ExperienceError> {
        Ok(canonical_json(self)?)
    }

    /// The content address of this experience.
    pub fn id(&self) -> Result<ExperienceId, ExperienceError> {
        Ok(ExperienceId(Digest::of(&self.canonical()?)))
    }

    /// Checks what the type system cannot: spans that end after they start,
    /// the required text fields present, and a task reference that
    /// addresses the task this record carries.
    pub fn validate(&self) -> Result<(), ExperienceError> {
        validate_task(
            &self.task,
            &self.evidence,
            &self.environment,
            &self.instruction,
            &self.privileged,
            &self.concepts,
        )?;
        if self.provenance.solver.trim().is_empty() {
            return Err(ExperienceError::Missing("provenance.solver"));
        }
        if self.provenance.created_at.trim().is_empty() {
            return Err(ExperienceError::Missing("provenance.created_at"));
        }
        Ok(())
    }
}

fn validate_task(
    task: &TaskRef,
    evidence: &[Span],
    environment: &Environment,
    instruction: &str,
    privileged: &[Privileged],
    concepts: &[String],
) -> Result<(), ExperienceError> {
    for (field, value) in [
        ("task.kind", task.kind.as_str()),
        ("environment.kind", environment.kind.as_str()),
        ("instruction", instruction),
    ] {
        if value.trim().is_empty() {
            return Err(ExperienceError::Missing(field));
        }
    }
    if concepts.iter().any(|c| c.trim().is_empty()) {
        return Err(ExperienceError::Missing("concepts[]"));
    }
    if let Some(recorded) = &environment.snapshot {
        let expected = Environment::snapshot_of(&environment.kind, &environment.spec);
        if *recorded != expected {
            return Err(ExperienceError::EnvironmentSnapshot {
                recorded: recorded.clone(),
                expected,
            });
        }
    }
    for span in evidence
        .iter()
        .chain(privileged.iter().filter_map(|p| p.span.as_ref()))
    {
        span.validate()?;
    }
    let expected = TaskBody {
        kind: &task.kind,
        evidence,
        environment,
        instruction,
        privileged,
        concepts,
    }
    .address()?;
    if expected != task.id {
        return Err(ExperienceError::TaskAddress {
            recorded: task.id.clone(),
            expected,
        });
    }
    Ok(())
}

/// Why a task or an experience is not valid.
#[derive(Debug, thiserror::Error)]
pub enum ExperienceError {
    /// A span ends before it starts.
    #[error("span [{start}, {end}) of {source_digest} ends before it starts")]
    Span {
        /// The source the span indexes.
        source_digest: Digest,
        /// Its start.
        start: u64,
        /// Its end.
        end: u64,
    },
    /// A required text field is empty.
    #[error("{0} is empty")]
    Missing(&'static str),
    /// The task reference does not address the task the record carries.
    #[error("task id {recorded} does not address this task (its content hashes to {expected})")]
    TaskAddress {
        /// The id the record carries.
        recorded: Digest,
        /// The id its content hashes to.
        expected: Digest,
    },
    /// The environment's snapshot is not the digest of its kind and spec.
    #[error("environment snapshot {recorded} does not match its kind and spec (they hash to {expected})")]
    EnvironmentSnapshot {
        /// The snapshot the record carries.
        recorded: Digest,
        /// The digest its kind and spec hash to.
        expected: Digest,
    },
    /// The record cannot be serialized (a float that JSON cannot hold).
    #[error("cannot serialize the record: {0}")]
    Serialize(#[from] serde_json::Error),
}
