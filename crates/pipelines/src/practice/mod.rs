// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `practice`: a git revision of a repository -> its tests made into
//! executable tasks -> solved by the policy as an agent in copies of the
//! tree -> kept where hidden checks confirm -> trained on -> released, as
//! one recorded run.
//!
//! The stages, in order, are the engine's to run and record
//! ([`splinter_orchestrator::pipeline`]); each only does its work and says
//! what it did:
//!
//! * **policy** - the release `policy:default` points at, resolved once.
//! * **snapshot** - the revision's tracked files, archived and kept; the
//!   same tree is the same snapshot whichever clone or day it is taken from.
//! * **profile** - how the tests are run (detected by rule or given), and
//!   the baseline run twice: a red or flaky baseline stops the run, since
//!   nothing generated from it could be trusted.
//! * **tasks** - small faults seeded by code into the files under test,
//!   each admitted only when the project's own tests catch it
//!   reproducibly, the change reversed makes its check pass, and the tests
//!   that catch it can be taken out of the solver's tree. What earlier runs
//!   already hold - the same change to the same snapshot, a request that
//!   repeats one made for the same file - is not made again, so a second run
//!   over the same snapshot adds nothing.
//! * **reserve** - the exam's families (a family is the file that was
//!   changed) are decided here, by a stable hash of their names, **before
//!   any rollout**. Their tasks are kept apart and never solved here, so no
//!   attempt at them can reach training data, and a family reserved by an
//!   earlier run stays reserved: nothing trained on can be examined later.
//! * **solve** - each remaining task is attempted k times by the policy as
//!   an agent in its own copy of the snapshot, every attempt drawing from a
//!   seed that is a function of the task and the attempt alone.
//! * **verify** - the attempt's patch is applied to a clean copy and the
//!   project's checks run, with the hidden tests laid over it; an exposure
//!   audit reads the trajectory for anything only the grader holds, and an
//!   experience it fails is decided fail and counted.
//! * **frontier** - pass@k keeps the tasks the policy solves sometimes: it
//!   has an attempt that passed and one that did not. No teacher is asked: a
//!   change to a repository has no passage to show one.
//! * **dataset** - the passing attempts, action by action (`sft-step`),
//!   with the tools the agent was offered; records longer than the block the
//!   trainer fits are left out, counted, and the dataset is checked to hold
//!   no family of the exam.
//! * **train** - from the champion, with earlier releases' records
//!   replayed.
//! * **release** - the gate, as for any candidate, over the exam this run
//!   reserved: the champion and the candidate attempt the exam's tasks as
//!   agents under the same bounds and seeds the solve stage used, graded by
//!   the same verifiers; every earlier release is held to the exam of the
//!   run that made it; the served candidate is asked the same way through
//!   `brain serve`. A candidate the gate blocks, or cannot measure, is not
//!   released, and the run stops with the reason of each check that failed.

mod exam;
mod held;
mod report;
mod stages;

use std::time::{Duration, Instant};

use serde::Serialize;
use splinter_core::rng::Seed;
use splinter_eval::gate::{AgenticConfig, GatePlan, DEFAULT_PROBE_ATTEMPTS};

use crate::experiment::ExperimentPlan;

use crate::curriculum::frontier::PassAtK;
use crate::train::{Trainer, Tuning};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::runs::{record, Recorded};

pub use exam::ExamReserved;
pub use report::{PracticeReport, Profiled, Snapshotted};

/// The stages, in order, as runs and reports name them.
pub const STAGES: [&str; 10] = [
    "policy", "snapshot", "profile", "tasks", "reserve", "solve", "verify", "frontier", "dataset",
    "train",
];

/// The release stage, which follows training unless the request stops at
/// the candidate.
pub const RELEASE_STAGE: &str = "release";

pub use splinter_eval::gate::{
    DEFAULT_FOLLOW_UPS, DEFAULT_MAX_REPEATED_RESULTS, DEFAULT_MAX_TOOL_CALLS,
};

/// One `practice`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PracticeRequest {
    /// The git repository: a work tree or a bare repository.
    pub repository: std::path::PathBuf,
    /// The revision to learn from: anything git resolves to a commit.
    pub revision: String,
    /// The program and arguments that run the project's whole test suite in
    /// the root of its tree; empty is detected by rule (a Python project run
    /// by `unittest`, a Rust project by `cargo test --offline`).
    pub test_command: Vec<String>,
    /// The files, relative to the tree, that tasks may change; empty is
    /// every source file of the project the rules recognise.
    pub files: Vec<String>,
    /// Snapshot the committed files of a revision that is checked out with
    /// uncommitted changes, instead of refusing.
    pub allow_dirty: bool,
    /// The most tasks one run admits; the generator's default when `None`.
    pub max_tasks: Option<usize>,
    /// The most faults seeded in one file; the generator's default when
    /// `None`.
    pub mutants_per_file: Option<usize>,
    /// How many families are reserved for the exam: `None` is a quarter of
    /// the families of the run's tasks, rounded down; `Some(0)` none.
    pub exam_families: Option<usize>,
    /// The seed the exam families are chosen by, when this run reserves the
    /// first: drawn when `None`. Once a run has reserved, the seed it chose
    /// by is remembered with the families and every later run reads it back;
    /// naming another is refused, since the exam would then not be the one
    /// earlier runs were kept apart from.
    pub exam_seed: Option<Seed>,
    /// k and the sampling of the pass@k that keeps the tasks worth training
    /// on.
    pub pass_at_k: PassAtK,
    /// Follow-up turns after red checks; [`DEFAULT_FOLLOW_UPS`] when `None`.
    pub follow_ups: Option<u32>,
    /// Tool calls in one attempt; [`DEFAULT_MAX_TOOL_CALLS`] when `None`.
    pub max_tool_calls: Option<u32>,
    /// Repeated identical answers before an attempt is stopped;
    /// [`DEFAULT_MAX_REPEATED_RESULTS`] when `None`.
    pub max_repeated_results: Option<u32>,
    /// Wall-clock time the whole pipeline may take.
    pub budget: Option<Duration>,
    /// Stop at the candidate: do not run the release gate.
    pub no_release: bool,
    /// The release gate's statistical settings.
    pub gate: GatePlan,
    /// The step budget of the training; sized from the examples when `None`.
    pub steps: Option<u32>,
    /// LoRA rank of the adapter; the default rank when `None`.
    pub rank: Option<u32>,
    /// The base's precision, the learning rate, how the training is watched
    /// and the longest record it fits.
    pub tuning: Tuning,
    /// The experiment plan this run is a generation of; the repository must be
    /// a family the plan trains in that generation, and never a sealed one.
    pub experiment: Option<PlanSlot>,
}

/// Where in an [`ExperimentPlan`] a run is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PlanSlot {
    /// The plan.
    pub plan: ExperimentPlan,
    /// The generation, counted from 1.
    pub generation: usize,
}

impl PracticeRequest {
    /// How an agent works on a task under this request: the bounds the solve
    /// stage attempts the tasks under and the release gate probes the exam
    /// under, so a difference between the champion and the candidate is the
    /// weights'.
    #[must_use]
    pub fn agentic(&self) -> AgenticConfig {
        AgenticConfig {
            attempts: DEFAULT_PROBE_ATTEMPTS,
            follow_ups: self.follow_ups.unwrap_or(DEFAULT_FOLLOW_UPS),
            max_tool_calls: Some(self.max_tool_calls.unwrap_or(DEFAULT_MAX_TOOL_CALLS)),
            max_repeated_results: Some(
                self.max_repeated_results
                    .unwrap_or(DEFAULT_MAX_REPEATED_RESULTS),
            ),
        }
    }
}

/// Runs `request`, training with `trainer`.
pub fn practice(
    ctx: &Context,
    request: &PracticeRequest,
    trainer: &dyn Trainer,
) -> Result<Recorded<PracticeReport>, OrchestratorError> {
    stages::validate(request)?;
    let practice = stages::Practice {
        request,
        trainer,
        deadline: request.budget.map(|b| Instant::now() + b),
    };
    record(ctx, "practice", request, |run| {
        let mut state = stages::PracticeState::new(practice);
        let deadline = state.practice.deadline;
        let stopped = stages::pipeline().run(ctx, run, &mut state, deadline)?;
        let mut report = state.report;
        report.stopped = stopped;
        Ok(report)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The list a report shows is the pipeline that runs.
    #[test]
    fn the_stage_list_is_the_pipeline() {
        let mut stages = STAGES.to_vec();
        stages.push(RELEASE_STAGE);
        assert_eq!(stages::pipeline().names(), stages);
    }
}
