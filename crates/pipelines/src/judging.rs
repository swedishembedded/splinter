// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements judges that are measured before their
// verdicts count, for its clients. If your team needs expertise in
// calibrating an LLM judge, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Measuring a judge before its verdicts count: controls made from tasks'
//! own references, by no model.
//!
//! Each task's reference is the right answer to it. The wrong answer to it is
//! the reference of a task from another family of sources
//! ([`crate::grouping::task_clusters`]): another task about the same letter
//! may say the same thing, and calling it wrong would measure the judge on a
//! label that is itself wrong. A task with no other family to borrow from
//! gets no wrong control.

use splinter_core::annotation::Outcome;
use splinter_core::experience::{Experience, PrivilegedKind, Provenance, Task};

use crate::grouping::task_clusters;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What a control's provenance names as its solver: not a model.
pub const CONTROLS_SOURCE: &str = "splinter/exam-controls";

/// An experience of `task` answered with `answer`, as a control is: by no
/// model.
pub fn answered(ctx: &Context, task: &Task, answer: &str) -> Result<Experience, OrchestratorError> {
    Ok(Experience::answered_without_a_run(
        task.clone(),
        answer,
        Provenance::new(CONTROLS_SOURCE, ctx.clock()),
    )?)
}

/// The reference of `task`, the answer it is controlled with.
#[must_use]
pub fn reference(task: &Task) -> Option<&str> {
    task.privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .map(|p| p.content.as_str())
}

/// The judge's controls over `tasks`: each task's reference, labelled right
/// for it, and the reference of a task of another family, labelled wrong for
/// it. A task with no reference makes no control, and a task that has no
/// other family's reference to borrow gets only the right one.
pub fn controls(
    ctx: &Context,
    tasks: &[Task],
) -> Result<Vec<(Task, Experience, Outcome)>, OrchestratorError> {
    let clusters = task_clusters(ctx, tasks)?;
    let mut labelled = Vec::with_capacity(tasks.len() * 2);
    for (n, task) in tasks.iter().enumerate() {
        let Some(right) = reference(task) else {
            continue;
        };
        labelled.push((task.clone(), answered(ctx, task, right)?, Outcome::Pass));
        // The next task, going round, of another family with another reference.
        let wrong = (1..tasks.len())
            .map(|step| (n + step) % tasks.len())
            .find(|&m| {
                let other = reference(&tasks[m]);
                other.is_some_and(|other| other != right)
                    && (clusters[m].is_none() || clusters[m] != clusters[n])
            })
            .and_then(|m| reference(&tasks[m]));
        if let Some(wrong) = wrong {
            labelled.push((task.clone(), answered(ctx, task, wrong)?, Outcome::Fail));
        }
    }
    Ok(labelled)
}

/// `items` thinned to at most `most`, evenly from first to last.
pub(crate) fn spaced<T: Clone>(items: &[T], most: usize) -> Vec<T> {
    if items.len() <= most || most < 2 {
        return items.iter().take(most).cloned().collect();
    }
    (0..most)
        .map(|i| items[i * (items.len() - 1) / (most - 1)].clone())
        .collect()
}
