// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade agent work by the
// strongest check available, for its clients. If your team needs expertise
// in verifier design or reward modelling, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Verifiers by strength: the strongest check available for a task, its
//! strength recorded on every verdict.
//!
//! A [`Verifier`] grades one experience of one task. It sees the solver's
//! output ([`Experience::final_output`]) and the task's privileged material
//! (references, oracles, checks), never the solver's trajectory, so nothing
//! the solver claims about its own success reaches the grade. It abstains
//! when it does not apply or cannot decide; it never passes by default.
//!
//! | Strength | Verifier |
//! |---|---|
//! | [`Strength::Executable`] | [`executable::ExecutableVerifier`] (authored checks), [`mutation::MutationValidatedVerifier`] (generated tests admitted by mutation) |
//! | [`Strength::Formal`] | [`formal::ExactMatchVerifier`] against the task's reference |
//! | [`Strength::Consistency`] | [`consistency::AgreementVerifier`] over independent answers |
//! | [`Strength::Judged`] | a judge model, gated by [`calibration::CalibratedJudge`] |
//!
//! [`Strongest`] runs several and returns every verdict; the store's
//! [`decide`](splinter_store::annotation::decide) lets the strongest
//! agreeing ones decide, so no verdict is dropped here to reach that.
//!
//! Verification blocks: an executable check runs a process, a judge waits
//! for a model. Call it from a thread that may block (a plain thread, or
//! the runtime's blocking pool), never from inside an async task.

pub mod calibration;
pub mod consistency;
pub mod executable;
pub mod formal;
pub mod mutation;
pub mod normalise;

use serde_json::json;
use splinter_sandbox::SandboxError;
use splinter_store::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_store::digest::Digest;
use splinter_store::experience::{Experience, ExperienceError, PrivilegedKind, Task};
use splinter_store::experiences::{ExperienceStore, StoreError};

/// Grades experiences of tasks at one [`Strength`].
pub trait Verifier: Send + Sync {
    /// Who the verdicts come from; a changed grader carries a new version.
    fn producer(&self) -> Producer;

    /// How its verdicts are reached.
    fn strength(&self) -> Strength;

    /// The verdict on `exp`, an experience of `task`. Abstains when the
    /// verifier does not apply to the task or cannot decide; an error means
    /// it could not carry out the check at all. Blocks until it has run.
    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError>;
}

/// What a verifier found: the body of a verdict, before it is attached to
/// an experience with the verifier's producer and strength.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    /// Pass, fail, or no judgement.
    pub outcome: Outcome,
    /// What the judgement rests on: digests and counts, never the privileged
    /// material itself.
    pub evidence: serde_json::Value,
}

impl Finding {
    /// An abstention for `reason`, with `detail` beside it.
    #[must_use]
    pub fn abstain(reason: &str, detail: serde_json::Value) -> Self {
        Self {
            outcome: Outcome::Abstain,
            evidence: json!({ "abstained": reason, "detail": detail }),
        }
    }

    /// Pass when `passed`, fail otherwise.
    #[must_use]
    pub fn decided(passed: bool, evidence: serde_json::Value) -> Self {
        Self {
            outcome: if passed { Outcome::Pass } else { Outcome::Fail },
            evidence,
        }
    }
}

/// Why a verifier could not carry out its check.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    /// The experience is not an experience of the task it was verified
    /// against.
    #[error("experience of task {experience} verified against task {task}")]
    TaskMismatch {
        /// The task given.
        task: Digest,
        /// The task the experience attempted.
        experience: Digest,
    },
    /// A privileged item the verifier reads is malformed.
    #[error("the task's {kind:?} item {index} is malformed: {reason}")]
    Malformed {
        /// The item's kind.
        kind: PrivilegedKind,
        /// Its index among the task's privileged items.
        index: usize,
        /// What is wrong with it.
        reason: String,
    },
    /// A verifier parameter is out of range.
    #[error("invalid verifier parameter {name}: {reason}")]
    Parameter {
        /// The parameter.
        name: &'static str,
        /// Why it is refused.
        reason: String,
    },
    /// A judge was asked to grade an experience its own model produced.
    #[error("judge {identity} refuses to grade an experience its own model produced")]
    SelfJudging {
        /// The judge's (and the solver's) identity.
        identity: String,
    },
    /// A verifier that blocks on a model run was called from inside an async
    /// task, where blocking would stall (or panic) the runtime. Call it from
    /// a blocking thread, such as `tokio::task::spawn_blocking`.
    #[error("{verifier} blocks on a model run and was called from inside an async task")]
    InsideAsync {
        /// The verifier's producer name.
        verifier: String,
    },
    /// A calibration does not belong to the verifier it was given with.
    #[error("calibration measured {measured} cannot gate {verifier}")]
    CalibrationMismatch {
        /// The producer the calibration measured.
        measured: String,
        /// The producer it was given to.
        verifier: String,
    },
    /// A calibration label is not pass or fail.
    #[error("labelled example {index} is labelled abstain; a label is pass or fail")]
    Label {
        /// The example's index.
        index: usize,
    },
    /// The sandbox could not resolve or run a check.
    #[error(transparent)]
    Sandbox(#[from] SandboxError),
    /// An experience's id or a record's digest could not be computed.
    #[error(transparent)]
    Experience(#[from] ExperienceError),
    /// The experience store refused an annotation.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A verifier's own machinery failed (a judge model's run, for one).
    #[error("{producer} failed: {source}")]
    Failed {
        /// The verifier's producer name.
        producer: String,
        /// What failed.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// `verifier`'s verdict on `exp` as an annotation: the one route from a
/// verifier to the store, refusing an experience of another task.
pub fn annotation(
    verifier: &dyn Verifier,
    task: &Task,
    exp: &Experience,
) -> Result<Annotation, VerifyError> {
    if task.task != exp.task {
        return Err(VerifyError::TaskMismatch {
            task: task.task.id.clone(),
            experience: exp.task.id.clone(),
        });
    }
    let finding = verifier.verify(task, exp)?;
    Ok(Annotation {
        experience: exp.id()?,
        producer: verifier.producer(),
        body: AnnotationBody::Verdict {
            outcome: finding.outcome,
            strength: verifier.strength(),
            evidence: finding.evidence,
        },
    })
}

/// Several verifiers run together, every verdict kept.
pub struct Strongest {
    verifiers: Vec<Box<dyn Verifier>>,
}

/// What [`Strongest::run`] produced.
#[derive(Debug)]
pub struct Verification {
    /// One verdict per verifier that could carry out its check, in the
    /// order the verifiers were given.
    pub annotations: Vec<Annotation>,
    /// The verifiers that could not, with why.
    pub errors: Vec<(Producer, VerifyError)>,
}

impl Strongest {
    /// Runs `verifiers`, in order.
    #[must_use]
    pub fn new(verifiers: Vec<Box<dyn Verifier>>) -> Self {
        Self { verifiers }
    }

    /// Every verifier's verdict on `exp`. One verifier's failure does not
    /// cost the others' verdicts: it is reported beside them. An experience
    /// of another task is refused outright.
    pub fn run(&self, task: &Task, exp: &Experience) -> Result<Verification, VerifyError> {
        if task.task != exp.task {
            return Err(VerifyError::TaskMismatch {
                task: task.task.id.clone(),
                experience: exp.task.id.clone(),
            });
        }
        let mut report = Verification {
            annotations: Vec::new(),
            errors: Vec::new(),
        };
        for verifier in &self.verifiers {
            match annotation(verifier.as_ref(), task, exp) {
                Ok(note) => report.annotations.push(note),
                Err(e) => report.errors.push((verifier.producer(), e)),
            }
        }
        Ok(report)
    }
}

/// Runs `verifiers` on `exp` and appends every verdict to its annotation
/// log in `store`, which must already hold the experience. Returns what
/// was appended and which verifiers failed.
pub fn verify_and_annotate(
    store: &ExperienceStore,
    verifiers: &Strongest,
    task: &Task,
    exp: &Experience,
) -> Result<Verification, VerifyError> {
    let report = verifiers.run(task, exp)?;
    for note in &report.annotations {
        store.annotate(note)?;
    }
    Ok(report)
}

/// The privileged items of `kind`, with their indices.
fn privileged_of<'a>(
    task: &'a Task,
    kind: &'a PrivilegedKind,
) -> impl Iterator<Item = (usize, &'a str)> + 'a {
    task.privileged
        .iter()
        .enumerate()
        .filter(move |(_, p)| p.kind == *kind)
        .map(|(i, p)| (i, p.content.as_str()))
}

/// The task's one reference, or why there is not exactly one.
fn single_reference(task: &Task) -> Result<&str, Finding> {
    let references: Vec<&str> = privileged_of(task, &PrivilegedKind::Reference)
        .map(|(_, c)| c)
        .collect();
    match references[..] {
        [reference] => Ok(reference),
        _ => Err(Finding::abstain(
            "the task does not carry exactly one reference",
            json!({ "references": references.len() }),
        )),
    }
}
