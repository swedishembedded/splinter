// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The stages of `practice`, and the pipeline that orders them.
//!
//! Each stage is a function over [`PracticeState`]: what the run was asked,
//! and what the stages so far left behind. The engine
//! ([`splinter_orchestrator::pipeline`]) runs them in order and records
//! each; a stage here only does its work and says what it did, and whether
//! the run can go on.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

use splinter_agent::work::WorkProtocol;
use splinter_core::model_ref::{ModelRef, POLICY_DEFAULT};
use splinter_knowledge::capture::{capture_tree, DirtyPolicy, TreeRequest};
use splinter_knowledge::tasks::workspace::{
    RepositoryProfile, WorkspaceError, WorkspaceGenerator, WorkspacePlan,
};
use splinter_orchestrator::pipeline::{FnStage, Pipeline, StageEnd};
use splinter_orchestrator::runs::{to_json, Recorder};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_sandbox::TreeSnapshot;
use splinter_store::experiences::SetId;
use splinter_store::tasks::{TaskSet, TaskSetId};

use super::exam::{assert_no_exam_family, remember_trained, reserve};
use super::held::{held_tasks, keep_tasks};
use super::report::{PracticeReport, Profiled, Snapshotted, TasksMade};
use super::PracticeRequest;
use crate::curriculum::frontier::select_frontier;
use crate::curriculum::teacher::untaught;
use crate::datasets::{build, BuildRequest, ViewName, VoiceBuild};
use crate::learn::PolicyUsed;
use crate::release::exams::{self, ExamRecord};
use crate::release::{release, ReleaseRequest};
use crate::solving::{solve_tasks, SolveRequest};
use crate::train::{train, Trainer, DEFAULT_LORA_RANK, DEFAULT_REPLAY_FRACTION};
use crate::verify::{verify_set, Grading};
use splinter_core::annotation::Strength;
use splinter_eval::gate::GateConfig;

/// What one run was asked, resolved.
pub(super) struct Practice<'a> {
    pub(super) request: &'a PracticeRequest,
    pub(super) trainer: &'a dyn Trainer,
    pub(super) deadline: Option<Instant>,
}

/// The run so far: what it was asked, what its stages reported, and what
/// they hand the stages after them.
pub(super) struct PracticeState<'a> {
    pub(super) practice: Practice<'a>,
    pub(super) report: PracticeReport,
    snapshot: Option<TreeSnapshot>,
    generator: Option<WorkspaceGenerator>,
    qualified: Option<splinter_knowledge::tasks::workspace::Qualified>,
    commit: String,
    /// The tasks of the families not reserved for the exam.
    training: Option<TaskSetId>,
    /// Every family the exam is made of, this run's and earlier runs'.
    exam_families: BTreeSet<String>,
    attempts: Option<SetId>,
    candidate: Option<String>,
}

impl<'a> PracticeState<'a> {
    /// The state of a run that has not begun.
    pub(super) fn new(practice: Practice<'a>) -> Self {
        Self {
            practice,
            report: PracticeReport::default(),
            snapshot: None,
            generator: None,
            qualified: None,
            commit: String::new(),
            training: None,
            exam_families: BTreeSet::new(),
            attempts: None,
            candidate: None,
        }
    }
}

/// Refuses a request that cannot be honoured, before anything is recorded.
pub(super) fn validate(request: &PracticeRequest) -> Result<(), OrchestratorError> {
    if request.revision.is_empty() {
        return Err(OrchestratorError::Refused(
            "name the revision of the repository to learn from".into(),
        ));
    }
    request.pass_at_k.validate()
}

/// The stages of `practice`, in order.
pub(super) fn pipeline<'a>() -> Pipeline<'a, PracticeState<'a>> {
    Pipeline::new()
        .then(FnStage::new("policy", policy_stage).ignoring_budget())
        .then(FnStage::new("snapshot", snapshot_stage).ignoring_budget())
        .then(FnStage::new("profile", profile_stage).ignoring_budget())
        .then(FnStage::new("tasks", tasks_stage).ignoring_budget())
        .then(FnStage::new("reserve", reserve_stage).ignoring_budget())
        .then(FnStage::new("solve", solve_stage))
        .then(FnStage::new("verify", verify_stage).ignoring_budget())
        .then(FnStage::new("frontier", frontier_stage).ignoring_budget())
        .then(FnStage::new("dataset", dataset_stage).ignoring_budget())
        .then(FnStage::new("train", train_stage))
        .then(
            FnStage::new(super::RELEASE_STAGE, release_stage)
                .when(|s| !s.practice.request.no_release)
                .ignoring_budget(),
        )
}

type Done = Result<StageEnd, OrchestratorError>;

fn to_value(summary: &impl serde::Serialize) -> Result<serde_json::Value, OrchestratorError> {
    to_json("the stage summary", summary)
}

/// A failure of the workspace machinery that the person running the command
/// can act on, with what was being done.
fn refused(what: &str, e: &dyn std::fmt::Display) -> OrchestratorError {
    OrchestratorError::Refused(format!("{what}: {e}"))
}

fn policy_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    st.report.policy = PolicyUsed {
        alias: POLICY_DEFAULT.into(),
        release: ctx.policy_pin(POLICY_DEFAULT)?.map(|pin| pin.release),
    };
    Ok(StageEnd::done(to_value(&st.report.policy)?))
}

fn snapshot_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    let request = st.practice.request;
    let backend = ctx.environments().workspace()?;
    let captured = capture_tree(
        &TreeRequest {
            repository: &request.repository,
            revision: &request.revision,
            dirty: if request.allow_dirty {
                DirtyPolicy::Allow
            } else {
                DirtyPolicy::Refuse
            },
            terms: None,
        },
        &backend.host,
        &ctx.artifacts(),
        &backend.scratch,
        ctx.clock(),
    )
    .map_err(|e| refused("capturing the repository", &e))?;
    ctx.sources().put_source(&captured.source)?;
    let snapshotted = Snapshotted {
        commit: captured.commit.clone(),
        snapshot: captured.snapshot.digest.clone(),
        tree_id: captured.snapshot.tree_id.clone(),
        dirty: matches!(
            captured.source.source().origin,
            splinter_core::source::Origin::Repository {
                revision: Some(splinter_core::source::Revision { dirty: true, .. }),
                ..
            }
        ),
    };
    st.commit = captured.commit;
    st.snapshot = Some(captured.snapshot);
    let summary = to_value(&snapshotted)?;
    st.report.snapshot = Some(snapshotted);
    Ok(StageEnd::done(summary))
}

/// The generator that makes this run's tasks, under the request's bounds.
fn generator(
    ctx: &Context,
    request: &PracticeRequest,
) -> Result<WorkspaceGenerator, OrchestratorError> {
    let backend = ctx.environments().workspace()?;
    let defaults = WorkspacePlan::default();
    WorkspaceGenerator::new(
        backend.sandbox,
        backend.host,
        ctx.artifacts(),
        backend.scratch.join("generate"),
    )
    .with_plan(WorkspacePlan {
        max_tasks: request.max_tasks.unwrap_or(defaults.max_tasks),
        mutants_per_file: request
            .mutants_per_file
            .unwrap_or(defaults.mutants_per_file),
        ..defaults
    })
    .map_err(|e| refused("the task generator", &e))
}

/// The profile the request names or the rules give for `snapshot`.
fn profile_of(
    generator: &WorkspaceGenerator,
    scratch: &Path,
    snapshot: &TreeSnapshot,
    request: &PracticeRequest,
) -> Result<RepositoryProfile, OrchestratorError> {
    std::fs::create_dir_all(scratch).map_err(splinter_orchestrator::error::io(scratch))?;
    let laid = tempfile::Builder::new()
        .prefix("splinter-profile-")
        .tempdir_in(scratch)
        .map_err(|e| refused("no scratch directory to read the tree in", &e))?;
    let tree = laid.path().join("tree");
    generator
        .environment(snapshot)
        .materialize(
            &generator
                .archive_of(snapshot)
                .map_err(|e| refused("the snapshot is not kept", &e))?,
            &tree,
        )
        .map_err(|e| refused("laying out the snapshot", &e))?;
    let detected = RepositoryProfile::detect(&tree);
    let mut profile = match (detected, request.test_command.is_empty()) {
        (Ok(profile), _) => profile,
        (Err(WorkspaceError::NoProfile { .. }), false) => RepositoryProfile {
            test_command: Vec::new(),
            files_under_test: Vec::new(),
            baseline_secs: None,
            flaky: Vec::new(),
        },
        (Err(e), _) => {
            return Err(refused(
                "reading how the tests are run (name them with --test-command and --file)",
                &e,
            ))
        }
    };
    if !request.test_command.is_empty() {
        profile.test_command.clone_from(&request.test_command);
    }
    if !request.files.is_empty() {
        profile.files_under_test.clone_from(&request.files);
    }
    if profile.files_under_test.is_empty() {
        return Err(OrchestratorError::Refused(
            "no file under test is known for this tree: name them with --file".into(),
        ));
    }
    Ok(profile)
}

fn profile_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    let Some(snapshot) = st.snapshot.clone() else {
        unreachable!("the profile stage follows the snapshot stage")
    };
    let generator = generator(ctx, st.practice.request)?;
    let scratch = ctx.environments().workspace()?.scratch.join("profile");
    let profile = profile_of(&generator, &scratch, &snapshot, st.practice.request)?;
    let qualified = match generator.qualify(&profile, &snapshot) {
        Ok(qualified) => qualified,
        Err(WorkspaceError::BaselineRefused {
            reason,
            count,
            detail,
        }) => {
            // A baseline that cannot be trusted is a finding about the
            // repository, not a failure of the run.
            return Ok(StageEnd::halt(format!(
                "the baseline is refused ({reason:?}, {count}): {detail}"
            )));
        }
        Err(e) => return Err(refused("qualifying the baseline", &e)),
    };
    let profiled = Profiled {
        test_command: qualified.profile.test_command.clone(),
        files_under_test: qualified.profile.files_under_test.len(),
        tests: qualified.tests.len(),
        baseline_secs: qualified.profile.baseline_secs,
    };
    let summary = to_value(&profiled)?;
    st.generator = Some(generator);
    st.qualified = Some(qualified);
    st.report.profile = Some(profiled);
    Ok(StageEnd::done(summary))
}

fn tasks_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    let (Some(generator), Some(snapshot), Some(qualified)) =
        (&st.generator, &st.snapshot, &st.qualified)
    else {
        unreachable!("the tasks stage follows the profile stage")
    };
    let held = held_tasks(ctx)?;
    let generation = generator
        .from_mutations(snapshot, qualified, &held, ctx.clock())
        .map_err(|e| refused("making tasks", &e))?;
    let task_set = if generation.admitted.is_empty() {
        None
    } else {
        Some(keep_tasks(ctx, &st.commit, &generation.admitted)?)
    };
    let made = TasksMade {
        admitted: generation.admitted.len(),
        refused: generation.refused.clone(),
        held_before: held.len(),
        task_set: task_set.clone(),
    };
    let summary = to_value(&made)?;
    st.report.tasks = Some(made);
    st.training = task_set;
    Ok(match st.training {
        Some(_) => StageEnd::done(summary),
        None => StageEnd::stop(
            summary,
            "no task was admitted that earlier runs do not already hold, so there is nothing to \
             solve",
        ),
    })
}

fn reserve_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    let Some(all) = st.training.take() else {
        unreachable!("the reserve stage follows the tasks stage")
    };
    let members = ctx.tasks().get_set(&all)?.members;
    let divided = reserve(ctx, members, st.practice.request.exam_families)?;
    let summary = to_value(&divided.reserved)?;
    st.exam_families = divided.exam_families;
    let none_left = divided.training.is_empty();
    if !none_left {
        st.training = Some(ctx.tasks().put_set(&TaskSet {
            name: format!("{all} without the exam's families"),
            members: divided.training,
        })?);
    }
    st.report.reserve = Some(divided.reserved);
    Ok(if none_left {
        StageEnd::stop(
            summary,
            "every task is of a family reserved for the exam, so there is nothing to train on",
        )
    } else {
        StageEnd::done(summary)
    })
}

fn solve_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    let Some(tasks) = st.training.as_ref() else {
        unreachable!("the solve stage follows the reserve stage")
    };
    let request = st.practice.request;
    let agentic = request.agentic();
    let policy = ModelRef::policy_default();
    let solved = solve_tasks(
        ctx,
        &SolveRequest {
            task_set: tasks,
            solver: &policy,
            attempts: request.pass_at_k.k,
            sampling: request.pass_at_k.sampling_choice(),
            teacher: false,
            system: None,
            work: WorkProtocol {
                follow_ups: agentic.follow_ups,
                max_tool_calls: agentic.max_tool_calls,
                max_repeated_results: agentic.max_repeated_results,
            },
            deadline: st.practice.deadline,
            cancel: run.cancel_token(),
        },
    )?;
    let summary = to_value(&solved)?;
    st.attempts = Some(solved.experience_set.clone());
    st.report.solve = Some(solved);
    Ok(StageEnd::done(summary))
}

fn verify_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    let Some(attempts) = st.attempts.as_ref() else {
        unreachable!("the verify stage follows the solve stage")
    };
    let verified = verify_set(ctx, attempts, Grading::CodeOnly, &run.cancel_token())?;
    let summary = to_value(&verified)?;
    st.report.verify = Some(verified);
    Ok(StageEnd::done(summary))
}

/// Keeps the tasks the policy solves sometimes: no teacher is asked, since a
/// change to a repository has no passage to show one.
fn frontier_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    let (Some(tasks), Some(solved)) = (st.training.as_ref(), st.report.solve.as_ref()) else {
        unreachable!("the frontier stage follows the solve stage")
    };
    let kept = select_frontier(ctx, tasks, solved, &untaught(ctx, tasks)?)?;
    let summary = to_value(&kept)?;
    st.attempts = Some(kept.frontier_experience_set.clone());
    let distribution = kept.distribution;
    st.report.frontier = Some(kept);
    if distribution.kept() == 0 {
        return Ok(StageEnd::stop(
            summary,
            format!(
                "no task is worth training on: {} always solved, {} never solved, {} unmeasured \
                 of {}",
                distribution.always,
                distribution.never,
                distribution.unmeasured,
                distribution.tasks()
            ),
        ));
    }
    Ok(StageEnd::done(summary))
}

fn dataset_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    let Some(attempts) = st.attempts.clone() else {
        unreachable!("the dataset stage follows the frontier stage")
    };
    let built = match build(
        ctx,
        &BuildRequest {
            sets: vec![attempts],
            view: ViewName::SftStep,
            strip: None,
            // The workspace verifier's executable verdict, which no weaker
            // check can stand in for.
            min_strength: Some(Strength::Executable),
            system_prompt: None,
            export_only: false,
            limit: None,
            max_block: st.practice.request.tuning.max_block.map(|b| b as usize),
            voice: VoiceBuild::default(),
        },
    ) {
        Ok(built) => built,
        Err(OrchestratorError::View(splinter_data::ViewError::Empty)) => {
            return Ok(StageEnd::halt(
                "no attempt passed verification untouched by the exposure audit, so there is \
                 nothing to train on",
            ));
        }
        Err(e) => return Err(e),
    };
    remember_trained(ctx, &assert_no_exam_family(&built.path, &st.exam_families)?)?;
    // The gate measures what this dataset trains on the exam reserved
    // against it, and every later release is held to it.
    exams::record(
        ctx,
        &ExamRecord {
            dataset: built.dataset.clone(),
            families: st.exam_families.clone(),
            tasks: st
                .report
                .reserve
                .as_ref()
                .and_then(|reserved| reserved.exam_task_set.clone()),
        },
    )?;
    let summary = to_value(&built)?;
    let records = built.records;
    st.report.dataset = Some(built);
    Ok(if records < splinter_data::holdout::MIN_SAMPLES {
        StageEnd::stop(
            summary,
            format!(
                "{records} record(s) passed; training holds records out for scoring and needs \
                 at least {}",
                splinter_data::holdout::MIN_SAMPLES
            ),
        )
    } else {
        StageEnd::done(summary)
    })
}

fn train_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    let Some(dataset) = st.report.dataset.as_ref() else {
        unreachable!("the train stage follows the dataset stage")
    };
    let request = st.practice.request;
    let candidate = train(
        ctx,
        &crate::train::TrainRequest {
            datasets: vec![dataset.dataset.to_string()],
            rehearsal: None,
            from: ModelRef::policy_default(),
            replay_fraction: DEFAULT_REPLAY_FRACTION,
            steps: request.steps,
            rank: request.rank.unwrap_or(DEFAULT_LORA_RANK),
            beta: None,
            tuning: request.tuning,
        },
        st.practice.trainer,
        &run.cancel_token(),
    )?;
    let summary = to_value(&candidate)?;
    st.candidate = Some(candidate.candidate.clone());
    st.report.candidate = Some(candidate);
    Ok(StageEnd::done(summary))
}

fn release_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut PracticeState<'_>) -> Done {
    let Some(id) = st.candidate.clone() else {
        unreachable!("the release stage follows the train stage")
    };
    let asked = ReleaseRequest {
        gate: GateConfig {
            agentic: st.practice.request.agentic(),
            ..GateConfig::default()
        },
        ..ReleaseRequest::new(id)
    };
    match release(ctx, &asked, &run.cancel_token()) {
        Ok(released) => {
            let summary = to_value(&released)?;
            let blocked = released.gate.blocked_by();
            let made = released.release.is_some();
            st.report.release = Some(released);
            Ok(if made {
                StageEnd::done(summary)
            } else {
                StageEnd::stop(
                    summary,
                    format!("the candidate was not released: {}", blocked.join("; ")),
                )
            })
        }
        // A candidate the gate may not judge (its champion moved on) is still
        // trained; the run says why it was not released.
        Err(e) if e.is_refusal() => Ok(StageEnd::halt(e.to_string())),
        Err(e) => Err(e),
    }
}
