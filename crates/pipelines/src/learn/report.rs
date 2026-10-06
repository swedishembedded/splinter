// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that acquire a capability
// from a document or a tool and prove it with evidence. If your team needs
// expertise in agent infrastructure or small-model training loops, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a `learn` reports: the plan of a dry run, and the recorded run stage
//! by stage.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_core::release::ReleaseId;
use splinter_core::role::Role;
use splinter_knowledge::survey::Survey;

use crate::author::Authored;
use crate::critique::Critiqued;
use crate::curriculum::frontier::Frontier;
use crate::curriculum::quota::Selected;
use crate::curriculum::teacher::Taught;
use crate::datasets::Built;
use crate::exam::Exam;
use crate::exam_set::ExamSet;
use crate::plan::Plan;
use crate::powered::PoweredExam;
use crate::rehearsal::Rehearsed;
use crate::release::Released;
use crate::reserve::Reservation;
use crate::solving::Solved;
use crate::sources::{SourceSummary, SourceTarget};
use crate::tasks::TasksGenerated;
use crate::train::Candidate;
use crate::variants::VariantsGenerated;
use crate::verify::Verified;
use splinter_orchestrator::runs::Recorded;

/// What a dry run reports: the plan, and nothing written.
#[derive(Clone, Debug, Serialize)]
pub struct LearnPlan {
    /// The state root the run would write under.
    pub state: std::path::PathBuf,
    /// The sources, as they would be captured.
    pub sources: Vec<SourceTarget>,
    /// The task kinds.
    pub kinds: Vec<String>,
    /// The goal.
    pub goal: Option<String>,
    /// The budget, in seconds.
    pub budget_secs: Option<u64>,
    /// The model that solves, critiques and retries.
    pub policy: String,
    /// The model that writes the tasks.
    pub generator: String,
    /// The model that teaches what the policy never solves.
    pub teacher: String,
    /// The model that surveys the sources and plans, when a plan is asked
    /// for.
    pub planner: Option<String>,
    /// The stages, in order.
    pub stages: Vec<&'static str>,
    /// Always `true`: nothing was written.
    pub dry_run: bool,
}

/// What the plan stage found and chose.
#[derive(Clone, Debug, Serialize)]
pub struct Planned {
    /// What the sources hold.
    pub survey: Survey,
    /// How the planner chose to learn from them.
    pub plan: Plan,
}

/// The policy a run works with, as resolved when it started.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PolicyUsed {
    /// The alias.
    pub alias: String,
    /// The release it pointed at; `None` when it pointed at none, and the
    /// run works from the base.
    pub release: Option<ReleaseId>,
}

/// What the policy stage records: the policy the run works with, and who
/// plays the other roles.
#[derive(Serialize)]
pub(super) struct PolicyStage<'a> {
    #[serde(flatten)]
    pub(super) policy: &'a PolicyUsed,
    pub(super) roles: &'a BTreeMap<Role, String>,
}

/// What a `learn` run reports, stage by stage.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LearnReport {
    /// The policy the whole run used.
    pub policy: PolicyUsed,
    /// The model each role the run played was given, by the identity its
    /// records carry.
    pub roles: BTreeMap<Role, String>,
    /// The sources learned from.
    pub sources: Vec<SourceSummary>,
    /// The plan stage: the survey and the planner's choice; `None` when no
    /// plan was asked for.
    pub plan: Option<Planned>,
    /// The reserve stage: the exam's families, taken out of the sources
    /// before anything was generated; `None` when none were reserved.
    pub reserve: Option<Reservation>,
    /// The exam-set stage: the frozen exam written from them.
    pub exam_set: Option<ExamSet>,
    /// The frozen dev suite a checkpoint is to be chosen on, written from its
    /// own reserved families; `None` when none were reserved.
    pub dev_set: Option<ExamSet>,
    /// The tasks stage.
    pub tasks: Option<TasksGenerated>,
    /// The solve stage.
    pub solve: Option<Solved>,
    /// The verify stage.
    pub verify: Option<Verified>,
    /// The teach stage: the teacher's graded solves of the tasks never
    /// solved.
    pub teach: Option<Taught>,
    /// The author stage: the writer's own passages put forward as answers.
    pub authored: Option<Authored>,
    /// The frontier stage: pass@k, and the tasks kept.
    pub frontier: Option<Frontier>,
    /// The variants stage: the tasks kept, asked in other words.
    pub variants: Option<VariantsGenerated>,
    /// The critique stage.
    pub critique: Option<Critiqued>,
    /// The select stage: the training set under the quotas.
    pub select: Option<Selected>,
    /// The dataset stage: the dialogue dataset.
    pub dataset: Option<Built>,
    /// The writer's own text trained beside it (the `voice` view); `None`
    /// when the run has no persona or asked for none.
    pub voice: Option<Built>,
    /// The rehearse stage: the base's own answers to general tasks, mixed
    /// into training; `None` when the run has no persona or asked for none.
    pub rehearsal: Option<Rehearsed>,
    /// The candidate trained.
    pub candidate: Option<Candidate>,
    /// The checkpoint stage: each kept evaluation scored on the dev suite and
    /// the step chosen; the candidate is then the one that carries it.
    pub checkpoint: Option<crate::checkpoints::Selected>,
    /// The release gate on it, and the release when it passed.
    pub release: Option<Released>,
    /// The exam stage: base against candidate on held-out tasks, graded by a
    /// calibrated judge and by the grounding check.
    pub exam: Option<Exam>,
    /// The exam stage of a run that reserved its exam: the powered exam, with
    /// every verdict of every arm.
    pub powered: Option<PoweredExam>,
    /// Why the pipeline stopped before training or releasing, if it did.
    pub stopped: Option<String>,
}

impl LearnReport {
    /// Whether the run got as far as it was asked: a candidate trained,
    /// and released unless the release was not asked for.
    #[must_use]
    pub fn finished(&self, release_asked: bool) -> bool {
        self.candidate.is_some()
            && (!release_asked || self.release.as_ref().is_some_and(|r| r.release.is_some()))
    }
}

/// What `learn` did: the plan of a dry run, or the recorded run.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Learned {
    /// A dry run's plan.
    Planned(Box<LearnPlan>),
    /// A run and its report.
    Ran(Box<Recorded<LearnReport>>),
}
