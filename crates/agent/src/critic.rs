// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model critics that explain why an agent's
// attempt failed without handing it the answer, for its clients. If your
// team needs expertise in agent self-correction or learning from feedback,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The critic: a model saying what is wrong with a failed attempt and why,
//! closed-book through the same sven solve every task runs through.
//!
//! The critic is shown the task's instruction, the failed answer and a
//! summary of each verdict on it: its strength and outcome, and for an
//! executable verdict how each check's run ended and which checks failed
//! (`splinter_lab::verifiers::executable::evidence_summary` and
//! `failed_checks`). It is never shown the task's privileged material (the
//! reference, the checks' code, generated tests, oracles, passages or
//! hints), nor a judge's reasons, which may quote the reference: [`critic_prompt`]
//! takes the instruction as text and the verdicts, not the task, so nothing
//! else can reach the prompt. It is told not to write the solution.
//!
//! A critique is stored as an experience of its own: the task is the
//! closed-book question of kind [`CRITIQUE_TASK_KIND`] whose instruction is
//! the critic's prompt, the final output is the critique, the provenance
//! names the critic's model, and a `CritiqueOf` relation names the failed
//! experience. Only an experience decided fail is critiqued.

use splinter_lab::verifiers::executable::{evidence_summary, failed_checks};
use splinter_sandbox::ResolvedEnvironment;
use splinter_store::annotation::{
    decide, Annotation, AnnotationBody, Outcome, Producer, RelationKind, Strength,
};
use splinter_store::clock::Clock;
use splinter_store::experience::{Environment, ExperienceId, Provenance, Task};
use splinter_store::experiences::ExperienceStore;
use sven_sdk::{RunConclusion, Usage};
use tokio::runtime::Handle;

use crate::repair::RepairError;
use crate::solve::{solve, Model, SolveOptions};

/// The producer name of the relations a critic records.
pub const PRODUCER: &str = "splinter-agent/critic";

/// The critic's version: bumped whenever its prompt changes.
pub const VERSION: &str = "1";

/// The task kind of the closed-book question a critic is asked.
pub const CRITIQUE_TASK_KIND: &str = "critique";

/// What the critic is told to do, before the task, answer and findings.
const INSTRUCTIONS: &str = "You are reviewing an attempt at a task that failed its checks. Say \
what is wrong with the answer and why, specifically enough that its author can fix it. Do not \
write a corrected answer, or any part of one.";

/// Stands in for an answer the attempt did not give.
const NO_ANSWER: &str = "(no answer)";

/// Stands in for findings when no verdict decided anything.
const NO_FINDINGS: &str = "(none recorded)";

/// A critique of a failed experience, stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Critique {
    /// The critique's own experience.
    pub id: ExperienceId,
    /// The experience it critiques.
    pub critiqued: ExperienceId,
    /// What the critic said.
    pub text: String,
}

/// What one critic run produced.
#[derive(Clone, Debug)]
pub struct CriticRun {
    /// The critique, when the run concluded with one; a run stopped by a
    /// bound stores nothing.
    pub critique: Option<Critique>,
    /// How the critic's run ended.
    pub conclusion: RunConclusion,
    /// The tokens its provider reported.
    pub usage: Usage,
}

/// A model critiquing failed attempts. See the module documentation.
#[derive(Clone, Debug)]
pub struct Critic {
    model: Model,
}

impl Critic {
    /// A critic on `model`.
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self { model }
    }

    /// The critic's model identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.model.identity
    }

    /// Critiques `failed`, an experience in `store` decided fail, within
    /// `options`, running the critic's solve on `runtime`, and stores the
    /// critique with its `CritiqueOf` relation, stamped by `clock`. Blocks
    /// on `runtime`: call it outside that runtime's async tasks. Refused
    /// for an experience not decided fail.
    pub fn critique(
        &self,
        store: &ExperienceStore,
        clock: &dyn Clock,
        runtime: &Handle,
        failed: &ExperienceId,
        options: SolveOptions,
    ) -> Result<CriticRun, RepairError> {
        refuse_inside_async()?;
        let experience = store.get(failed)?;
        let verdicts = store.annotations(failed)?.annotations;
        let decision = decide(&verdicts);
        if decision.is_none_or(|d| d.passed) {
            return Err(RepairError::NotFailed {
                experience: failed.clone(),
                decision,
            });
        }
        let prompt = critic_prompt(
            &experience.instruction,
            experience.final_output.as_deref(),
            &verdicts,
        );
        let question = Task::new(
            CRITIQUE_TASK_KIND,
            vec![],
            Environment::closed_book(),
            prompt,
            vec![],
        )?;
        let options = SolveOptions {
            stream_idle: self.model.stream_idle,
            ..options
        };
        let solution = runtime.block_on(solve(
            &question,
            &ResolvedEnvironment::ClosedBook,
            self.model.provider.clone(),
            options,
        ))?;
        let (conclusion, usage) = (solution.conclusion, solution.usage);
        let Some(text) = solution.final_output.clone() else {
            return Ok(CriticRun {
                critique: None,
                conclusion,
                usage,
            });
        };
        let critique = solution.into_experience(
            question,
            Provenance::new(self.model.identity.clone(), clock),
        )?;
        let id = store.put(&critique)?;
        store.annotate(&Annotation {
            experience: id.clone(),
            producer: Producer {
                name: PRODUCER.into(),
                version: VERSION.into(),
            },
            body: AnnotationBody::Relation {
                kind: RelationKind::CritiqueOf,
                other: failed.clone(),
            },
        })?;
        Ok(CriticRun {
            critique: Some(Critique {
                id,
                critiqued: failed.clone(),
                text,
            }),
            conclusion,
            usage,
        })
    }
}

/// The prompt a critic is asked about an attempt at `instruction` that
/// answered `answer`, with `verdicts` on it: the critic's instructions, the
/// task, the answer, and one finding per pass or fail verdict - its
/// strength and outcome, and for an executable verdict the summary of how
/// each check ended and which failed. Annotations other than verdicts, and
/// abstentions, are left out; so is everything else a verdict's evidence
/// holds.
#[must_use]
pub fn critic_prompt(instruction: &str, answer: Option<&str>, verdicts: &[Annotation]) -> String {
    let findings: Vec<String> = verdicts.iter().filter_map(finding).collect();
    let findings = if findings.is_empty() {
        NO_FINDINGS.to_string()
    } else {
        findings.join("\n")
    };
    format!(
        "{INSTRUCTIONS}\n\nTASK:\n{instruction}\n\nANSWER:\n{}\n\nFINDINGS:\n{findings}\n",
        answer.unwrap_or(NO_ANSWER)
    )
}

/// One verdict as the critic may read it; `None` for anything else.
fn finding(note: &Annotation) -> Option<String> {
    let AnnotationBody::Verdict {
        outcome,
        strength,
        evidence,
    } = &note.body
    else {
        return None;
    };
    let outcome = match outcome {
        Outcome::Pass => "pass",
        Outcome::Fail => "fail",
        Outcome::Abstain => return None,
    };
    let mut lines = vec![format!("- {} verdict: {outcome}", strength_name(*strength))];
    if let Some(summary) = evidence_summary(evidence) {
        lines.extend(summary.lines().map(|line| format!("  {line}")));
    }
    if let Some(failed) = failed_checks(evidence).filter(|f| !f.is_empty()) {
        let names: Vec<String> = failed.iter().map(|n| format!("check {n}")).collect();
        lines.push(format!("  failed checks: {}", names.join(", ")));
    }
    Some(lines.join("\n"))
}

fn strength_name(strength: Strength) -> &'static str {
    match strength {
        Strength::Judged => "judged",
        Strength::Consistency => "consistency",
        Strength::Formal => "formal",
        Strength::Executable => "executable",
    }
}

/// Refuses to run inside an async task, where blocking on a model run would
/// stall the runtime (and `block_on` would panic).
pub(crate) fn refuse_inside_async() -> Result<(), RepairError> {
    if Handle::try_current().is_ok() {
        return Err(RepairError::InsideAsync);
    }
    Ok(())
}
