// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent solvers whose every run is replayable
// evidence, for its clients. If your team needs expertise in agent
// environments or learning from agent experience, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The solve stage: a task set becomes an experience set.
//!
//! Each task is solved in exactly the environment it records - closed-book,
//! or its runtime in the process sandbox - and refused (and reported) when
//! that environment is not what it was. Every solve is stored as an
//! experience, answered or not: a run stopped by its deadline is evidence
//! too.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::Serialize;
use splinter_agent::solve::{solve, SolveError, SolveOptions};
use splinter_store::digest::Digest;
use splinter_store::experience::Provenance;
use splinter_store::experiences::{ExperienceSet, SetId};
use splinter_store::tasks::TaskSetId;
use sven_sdk::{CancelToken, RunConclusion};

use crate::context::Context;
use crate::error::CampaignError;
use crate::model_ref::ModelRef;
use crate::tasks::remaining;

/// How long one task's solve may take.
pub const DEFAULT_SOLVE_DEADLINE: Duration = Duration::from_secs(300);

/// A task left unsolved, and why.
#[derive(Clone, Debug, Serialize)]
pub struct Skipped {
    /// The task.
    pub task: Digest,
    /// Why it was not solved.
    pub reason: String,
}

/// What the solve stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct Solved {
    /// The experience set (`verify <experience_set>`).
    pub experience_set: SetId,
    /// The solver's identity.
    pub solver: String,
    /// Experiences recorded.
    pub solved: usize,
    /// Of those, how many gave a final answer.
    pub answered: usize,
    /// How the runs ended, by conclusion.
    pub conclusions: BTreeMap<String, usize>,
    /// Tasks not solved, with why.
    pub skipped: Vec<Skipped>,
    /// Why solving stopped before every task was tried, if it did.
    pub stopped: Option<String>,
}

/// Solves every task of `task_set` with `solver`; no solve starts after
/// `deadline` or runs past it.
pub fn solve_set(
    ctx: &Context,
    task_set: &TaskSetId,
    solver: &ModelRef,
    deadline: Option<Instant>,
    cancel: &CancelToken,
) -> Result<Solved, CampaignError> {
    let set = ctx.tasks().get_set(task_set)?;
    let model = ctx.model(solver)?;
    let store = ctx.experiences();
    let mut members = Vec::new();
    let mut report = Solved {
        experience_set: SetId(Digest::of(b"")),
        solver: model.identity.clone(),
        solved: 0,
        answered: 0,
        conclusions: BTreeMap::new(),
        skipped: Vec::new(),
        stopped: None,
    };
    for entry in &set.members {
        if cancel.is_cancelled() {
            return Err(CampaignError::Cancelled);
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            report.stopped = Some("the budget was spent before every task was tried".into());
            break;
        }
        let task = ctx.tasks().get(&entry.task)?;
        let environment = match ctx.environments().for_record(&task.environment) {
            Ok(environment) => environment,
            Err(e) => {
                report.skipped.push(skip(&entry.task, &e));
                continue;
            }
        };
        let mut options = SolveOptions::new(remaining(deadline, DEFAULT_SOLVE_DEADLINE));
        options.cancel = Some(cancel.clone());
        let solution =
            match ctx.block_on(solve(&task, &environment, model.provider.clone(), options)) {
                Ok(solution) => solution,
                Err(e @ (SolveError::EnvironmentMismatch { .. } | SolveError::Environment(_))) => {
                    report.skipped.push(skip(&entry.task, &e));
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
        *report
            .conclusions
            .entry(conclusion_name(solution.conclusion))
            .or_default() += 1;
        report.answered += usize::from(solution.final_output.is_some());
        let provenance = Provenance {
            generator: entry.generator.clone(),
            prompt_digests: entry.prompt.iter().cloned().collect(),
            ..Provenance::new(model.identity.clone(), ctx.clock())
        };
        let experience = solution.into_experience(task, provenance)?;
        let id = store.put(&experience)?;
        if !members.contains(&id) {
            members.push(id);
        }
    }
    report.solved = members.len();
    report.experience_set = store.put_set(&ExperienceSet {
        name: format!("{} solved by {}", task_set, model.identity),
        members,
    })?;
    Ok(report)
}

fn skip(task: &Digest, why: &dyn std::fmt::Display) -> Skipped {
    Skipped {
        task: task.clone(),
        reason: why.to_string(),
    }
}

/// How a run ended, as reports name it: `success`, `timeout`, ...
pub(crate) fn conclusion_name(conclusion: RunConclusion) -> String {
    let debug = format!("{conclusion:?}");
    let mut name = String::with_capacity(debug.len() + 4);
    for (i, c) in debug.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            name.push('_');
        }
        name.push(c.to_ascii_lowercase());
    }
    name
}
