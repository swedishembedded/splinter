// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data in a writer's own words, for
// its clients. If your team needs expertise in training models on a person's
// voice, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The writer's own words as the answer: instruction backtranslation.
//!
//! A task of a kind whose reference is a passage the writer wrote (`advise`,
//! `converse`) was written *for* that passage: its instruction is the message
//! the passage could answer. The best answer to teach is then the passage
//! itself, word for word - the writer's own diction, syntax and reasoning,
//! which no teacher's paraphrase has. The answer is recorded as the source's
//! ([`AUTHOR_SOURCE`]), never as a model's, and a judge of fit
//! ([`Judging::Fit`]) keeps only the pairs where the passage is a natural reply
//! to its message: a message written for another passage, or one that needs
//! context it does not give, is dropped. Its verdict is the only one these
//! experiences carry; comparing a passage with itself would say nothing.

use serde::Serialize;
use splinter_agent::CancelToken;
use splinter_core::annotation::Outcome;
use splinter_core::experience::{Experience, Provenance, Task};
use splinter_core::model_ref::ModelRef;
use splinter_eval::verifiers::{verify_and_annotate, Strongest};
use splinter_knowledge::tasks::Catalogue;
use splinter_store::experiences::{ExperienceSet, SetId};
use splinter_store::tasks::TaskSetId;

use crate::judging::reference;
use crate::verify::{Judge, Judging};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What the provenance of an experience answered by the writer's own words
/// names as its solver: the source, not a model.
pub const AUTHOR_SOURCE: &str = "splinter/author";

/// Whether tasks of `kind` have the writer's own passage as their reference,
/// which [`author`] puts forward as the answer.
#[must_use]
pub fn kind_authors(kind: &str) -> bool {
    Catalogue::builtin()
        .get(kind)
        .is_some_and(|spec| spec.reference_verbatim)
}

/// What to author from.
pub struct AuthorRequest<'a> {
    /// The tasks.
    pub task_set: &'a TaskSetId,
    /// The model that judges whether a passage fits its message: another
    /// model than the one that wrote the messages.
    pub judge: &'a ModelRef,
}

/// What the author stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct Authored {
    /// The tasks whose reference is the writer's own passage.
    pub tasks: usize,
    /// The experiences of the passages that fit their messages.
    pub kept: usize,
    /// The passages the judge refused as no reply to their message.
    pub refused: usize,
    /// The passages it could not decide on.
    pub undecided: usize,
    /// The experiences, one per task, graded; `None` when no task had a
    /// passage to put forward, or the stage was skipped.
    pub experience_set: Option<SetId>,
    /// Why the stage did nothing, when it had no judge of fit to keep only
    /// the passages that fit their messages.
    pub skipped: Option<String>,
}

/// Puts the writer's own passage forward as the answer to each task of
/// `request.task_set` whose reference is one, and grades each by fit.
pub fn author(
    ctx: &Context,
    request: &AuthorRequest<'_>,
    cancel: &CancelToken,
) -> Result<Authored, OrchestratorError> {
    let store = ctx.tasks();
    let tasks: Vec<Task> = store
        .get_set(request.task_set)?
        .members
        .iter()
        .map(|entry| store.get(&entry.task))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|task| kind_authors(&task.task.kind) && reference(task).is_some())
        .collect();
    let mut report = Authored {
        tasks: tasks.len(),
        kept: 0,
        refused: 0,
        undecided: 0,
        experience_set: None,
        skipped: None,
    };
    if tasks.is_empty() {
        return Ok(report);
    }
    let judge = Judge::calibrated_for(ctx, request.judge, &tasks, Judging::Fit)?;
    let verifiers = Strongest::new(vec![judge.verifier(ctx)?]);
    let experiences = ctx.experiences();
    let mut members = Vec::with_capacity(tasks.len());
    for task in &tasks {
        if cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        let Some(passage) = reference(task) else {
            continue;
        };
        let experience = Experience::answered_without_a_run(
            task.clone(),
            passage,
            Provenance::new(AUTHOR_SOURCE, ctx.clock()),
        )?;
        members.push(experiences.put(&experience)?);
        let verification = verify_and_annotate(&experiences, &verifiers, task, &experience)?;
        match verification
            .annotations
            .iter()
            .find_map(|note| match note.body {
                splinter_core::annotation::AnnotationBody::Verdict { outcome, .. } => Some(outcome),
                _ => None,
            }) {
            Some(Outcome::Pass) => report.kept += 1,
            Some(Outcome::Fail) => report.refused += 1,
            _ => report.undecided += 1,
        }
    }
    report.experience_set = Some(experiences.put_set(&ExperienceSet {
        name: format!("the writer's own words for {}", request.task_set),
        members,
    })?);
    Ok(report)
}
