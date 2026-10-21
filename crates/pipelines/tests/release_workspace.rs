// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in releasing a
// code-editing agent only when it beats the one in use on families of a
// repository it never trained on, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: a candidate trained by `practice` is released only on the evidence
//! of the exam its run reserved.
//!
//! * The champion and the candidate attempt the exam's tasks as agents, under
//!   the same bounds and seeds, and are graded by the workspace verifier and
//!   the exposure audit; the paired outcomes are clustered by family, so
//!   a family counts once in the sign test however many of its tasks there
//!   are. A candidate that beats the champion on the exam is released, with
//!   the anchor suite held (a frozen suite with no coding-agent task is
//!   decided on what it holds, and the report says it holds none), and the
//!   served agent graded as it was in-process; the manifest a server follows
//!   names the release.
//! * A candidate that loses, or ties, is not released, and the run says why;
//!   the next release is held to the exam of the run that made the first.
//! * An attempt whose trajectory shows the hidden tests, and an attempt the
//!   harness could not carry out, are set aside and counted; neither is in a
//!   denominator.
//! * The served candidate is held to the verdicts of the in-process one.
//!
//! The agents are scripted, training is a test double, and `brain` is a
//! stand-in script that serves the adapter as such an agent (see the
//! fixtures); no weights and no container are needed.

#![allow(clippy::unwrap_used)]

mod common;

use std::path::Path;

use common::fake_brain::fake_brain;
use common::gate::{anchor_file, put_base, Brain, FakeTrainer, NOW};
use common::repo::{project_of, Agent, Mender, Wording};
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::build::{BuildInfo, Revisions};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Task};
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_core::rng::Seed;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::frontier::PassAtK;
use splinter_pipelines::practice::{practice, PracticeReport, PracticeRequest};
use splinter_pipelines::release::exams::{self, ExamRecord};
use splinter_pipelines::release::{anchor, arm, release, ReleaseRequest, Released};
use splinter_store::tasks::{TaskEntry, TaskSet};

/// Modules of the project, each a family of tasks.
const MODULES: usize = 9;
/// Families the exam reserves: five discordant families are the fewest at
/// which a unanimous sign test is significant at the default level.
const EXAM_FAMILIES: usize = 5;

/// The build the specs run as: three distinct commits the manifest and the
/// run must each name.
fn build() -> BuildInfo {
    BuildInfo {
        revisions: Revisions {
            splinter: Some("5".repeat(40)),
            sven: Some("7".repeat(40)),
            brain: Some("b".repeat(40)),
        },
        dirty: Some(false),
        ..BuildInfo::default()
    }
}

/// A context over a scratch state root whose `policy:default` is the
/// practising agent, whose base arm - the champion before any release - does
/// nothing, and whose `brain` is `brain`.
fn context(test: &str, brain: Brain) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let mut config = config(&scratch);
    config.brain_binary = match brain {
        Brain::Missing => None,
        other => Some(fake_brain(&scratch.0, other)),
    };
    config.build = build();
    let base = arm(&config, None);
    let ctx = Context::new(config, false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new(NOW)))
        .with_model(
            ModelRef::policy_default(),
            Model::new(Agent::new(), "scripted/agent-1"),
        )
        .with_model(base, Model::new(Mender::new(false), "scripted/idle"));
    put_base(&ctx);
    // The anchor suite is general closed-book tasks; both arms fail them alike.
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    (scratch, ctx)
}

fn request(repo: &Path, release: bool) -> PracticeRequest {
    PracticeRequest {
        repository: repo.to_path_buf(),
        revision: "HEAD".into(),
        max_tasks: Some(64),
        mutants_per_file: Some(1),
        exam_families: Some(EXAM_FAMILIES),
        // seed: which modules form the exam decides how many training records
        // the fixture leaves; the spec needs one split that leaves enough.
        exam_seed: Some(Seed::new(1)),
        pass_at_k: PassAtK::new(2, None, None),
        no_release: !release,
        ..PracticeRequest::default()
    }
}

/// `practice` over the project at `repo`, its candidate a `trainer`'s.
fn practise(ctx: &Context, repo: &Path, trainer: &FakeTrainer, release: bool) -> PracticeReport {
    practice(ctx, &request(repo, release), trainer)
        .unwrap()
        .report
}

/// The released alias's manifest, as a server reads it: the adapter file and
/// its digest.
fn followed(ctx: &Context) -> Option<(String, String)> {
    let path = ctx.releases().manifest_path("default").unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).unwrap();
    Some((
        manifest["adapter"].as_str().unwrap().into(),
        manifest["digest"].as_str().unwrap().into(),
    ))
}

/// The first release: a candidate that mends every module, practised on a
/// project of [`MODULES`] modules, against the base that does nothing.
fn first_release(ctx: &Context, scratch: &Scratch) -> (ReleaseId, PracticeReport) {
    let repo = scratch.0.join("first");
    project_of(&repo, &[], Wording::Short, MODULES);
    let report = practise(ctx, &repo, &FakeTrainer::mending(true), true);
    let released = report
        .release
        .as_ref()
        .unwrap_or_else(|| panic!("no release was decided: {report:#?}"));
    assert!(released.gate.passed, "{:#?}", released.gate);
    (released.release.clone().unwrap(), report)
}

#[test]
fn a_candidate_that_mends_the_exam_the_champion_fails_is_released_and_a_server_follows_it() {
    let (scratch, ctx) = context("workspace-release", Brain::Honest);
    let (id, report) = first_release(&ctx, &scratch);
    assert_eq!(report.stopped, None, "{report:#?}");
    let released = report.release.as_ref().unwrap();
    let gate = &released.gate;
    let reserve = report.reserve.as_ref().unwrap();
    assert_eq!(reserve.families.len(), EXAM_FAMILIES);

    // The improvement is decided over the exam's families: each counts once
    // in the sign test however many tasks it has.
    let improvement = gate.improvement.measured.as_ref().unwrap();
    assert_eq!(improvement.suite.name, "exam");
    assert_eq!(improvement.suite.tasks, reserve.exam_tasks);
    // Faults of one module with different symptoms are different tasks, so a
    // family holds several; the sign test below still counts it once.
    assert!(reserve.exam_tasks >= EXAM_FAMILIES, "{reserve:#?}");
    assert_eq!(improvement.comparison.candidate_wins, reserve.exam_tasks);
    assert_eq!(improvement.comparison.baseline_wins, 0);
    assert_eq!(improvement.sign_test.discordant, EXAM_FAMILIES);
    assert!((improvement.sign_test.p_value - 1.0 / 32.0).abs() < 1e-12);
    // The report states what the exam's hidden suites are worth beside the
    // pass rates; it does not weigh them.
    let strength = improvement.hidden_strength.unwrap();
    assert_eq!(strength.tasks, reserve.exam_tasks);
    assert!(strength.measured <= strength.tasks);
    let probed = improvement.agentic.unwrap();
    assert_eq!(probed.attempts, 1);
    assert_eq!(
        (
            probed.candidate.exposed,
            probed.candidate.environment_failures
        ),
        (0, 0)
    );
    assert_eq!(
        (
            probed.champion.exposed,
            probed.champion.environment_failures
        ),
        (0, 0)
    );

    // Nothing to retain on the first release. The anchor suite holds no
    // coding-agent task, and says so rather than scoring one.
    assert!(gate.retention.passed && gate.retention.measured.as_ref().unwrap().suites.is_empty());
    let anchor = gate.anchor.measured.as_ref().unwrap();
    assert_eq!(anchor.workspace_tasks, Some(0));
    assert_eq!(anchor.agentic, None);
    assert!(gate.anchor.passed);

    // The served agent is held to the in-process verdicts.
    let serve = gate.serve.measured.as_ref().unwrap();
    assert_eq!(serve.agreement, splinter_eval::gate::Agreement::Verdicts);
    assert_eq!(
        serve.served_digest,
        report.candidate.as_ref().unwrap().adapter_digest
    );
    assert!(
        serve.sampled > 0 && serve.agreed == serve.sampled,
        "{serve:#?}"
    );

    // The alias moved, and the manifest a server follows names the release.
    let store = ctx.releases();
    assert_eq!(store.alias("default").unwrap(), Some(id.clone()));
    let stored = store.get(&id).unwrap();
    assert_eq!(
        followed(&ctx),
        Some((
            stored.artifact.display().to_string(),
            report.candidate.as_ref().unwrap().adapter_digest.clone()
        ))
    );
    // The manifest and the run name the three commits the release was made
    // by, so it can be rebuilt.
    let provenance = &stored.manifest.provenance;
    assert_eq!(
        (
            provenance.splinter_commit.as_deref(),
            provenance.sven_commit.as_deref(),
            provenance.brain_commit.as_deref()
        ),
        (
            build().revisions.splinter.as_deref(),
            build().revisions.sven.as_deref(),
            build().revisions.brain.as_deref()
        )
    );
    let run = splinter_store::runs::read_run(ctx.workspace(), &run_of(&ctx)).unwrap();
    assert_eq!(run.build, build());
    let stages: Vec<String> = run.stages.iter().map(|s| s.stage.clone()).collect();
    assert_eq!(stages.last().map(String::as_str), Some("release"));
}

/// The id of the one run recorded.
fn run_of(ctx: &Context) -> String {
    splinter_store::runs::list_runs(ctx.workspace())
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .id
}

/// A second project whose files have the first's names and one more, worded
/// so that none of its requests repeats one the first made.
fn second_repo(scratch: &Scratch) -> std::path::PathBuf {
    let repo = scratch.0.join("second");
    project_of(&repo, &[], Wording::Long, MODULES + 1);
    repo
}

#[test]
fn a_candidate_that_loses_or_ties_is_not_released_and_the_run_says_why() {
    let (scratch, ctx) = context("workspace-blocked", Brain::Honest);
    let (first, _) = first_release(&ctx, &scratch);
    let before = followed(&ctx);

    // The next candidate does nothing: it loses the first release's exam and
    // is no better than the champion on its own.
    let report = practise(
        &ctx,
        &second_repo(&scratch),
        &FakeTrainer::mending(false),
        true,
    );
    let blocked = report.release.as_ref().unwrap();
    assert_eq!(blocked.release, None, "{blocked:#?}");
    assert_eq!(blocked.champion, Some(first.clone()));
    let why = report.stopped.as_deref().unwrap();
    assert!(
        why.contains("was not released") && why.contains("improvement:"),
        "{why}"
    );
    assert!(why.contains("retention:"), "{why}");
    let gate = &blocked.gate;
    assert!(!gate.improvement.passed && !gate.retention.passed && !gate.passed);
    // Held to the exam of the run that made the champion, whose families
    // the candidate never trained on either.
    let retained = &gate.retention.measured.as_ref().unwrap().suites;
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].release, first);
    assert!(retained[0].drop.unwrap() > 0.5, "{retained:#?}");
    assert_eq!(retained[0].agentic.unwrap().attempts, 1);
    assert_eq!(
        ctx.releases().alias("default").unwrap(),
        Some(first.clone())
    );
    assert_eq!(
        followed(&ctx),
        before,
        "a release that was refused moves nothing"
    );

    // Made as good as the champion, it ties on everything: no evidence of an
    // improvement, which is not the same as evidence of none.
    let candidate = report.candidate.unwrap();
    ctx.add_model(
        arm(ctx.config(), Some(&candidate.adapter)),
        Model::new(Mender::new(true), "scripted/mender"),
    );
    let tied = release(
        &ctx,
        &ReleaseRequest::new(candidate.candidate.clone()),
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(tied.release, None);
    // An improvement check that measured nothing says why in its reason (no
    // task graded for both models, usually an attempt the environment could not
    // carry out): the failure must name it.
    let improvement = tied.gate.improvement.measured.as_ref().unwrap_or_else(|| {
        panic!(
            "the tied candidate's improvement was not measured: {:#?}",
            tied.gate.improvement
        )
    });
    assert_eq!(improvement.sign_test.discordant, 0);
    assert!(
        tied.gate
            .improvement
            .reason
            .as_deref()
            .unwrap()
            .contains("no significant improvement"),
        "{:#?}",
        tied.gate.improvement
    );
    assert!(tied.gate.retention.passed, "{:#?}", tied.gate.retention);
    assert_eq!(ctx.releases().alias("default").unwrap(), Some(first));
}

/// The exam tasks `release` for the candidate of `report` is held to, with
/// the module numbers of their files.
fn exam_modules(report: &PracticeReport) -> Vec<usize> {
    report
        .reserve
        .as_ref()
        .unwrap()
        .families
        .iter()
        .map(|file| {
            file.trim_start_matches("shop/tier")
                .trim_end_matches(".py")
                .parse()
                .unwrap()
        })
        .collect()
}

/// Gates the candidate of `report` with its arm served as `candidate`.
fn gated(ctx: &Context, report: &PracticeReport, candidate: Model) -> Released {
    let candidate_id = &report.candidate.as_ref().unwrap();
    ctx.add_model(arm(ctx.config(), Some(&candidate_id.adapter)), candidate);
    release(
        ctx,
        &ReleaseRequest::new(candidate_id.candidate.clone()),
        &CancelToken::new(),
    )
    .unwrap()
}

#[test]
fn an_attempt_that_saw_the_hidden_tests_or_could_not_be_run_is_set_aside_and_counted() {
    let (scratch, ctx) = context("workspace-withheld", Brain::Honest);
    let repo = scratch.0.join("repo");
    project_of(&repo, &[], Wording::Short, MODULES);
    let report = practise(&ctx, &repo, &FakeTrainer::mending(true), false);
    let reserve = report.reserve.as_ref().unwrap();
    let modules = exam_modules(&report);
    assert_eq!(modules.len(), EXAM_FAMILIES);

    // The exam has two tasks of each family - the same task worded again -
    // and one in a tree nobody kept, which the harness cannot lay out for
    // whoever attempts it.
    let candidate = report.candidate.as_ref().unwrap();
    let kept = ctx
        .tasks()
        .get_set(reserve.exam_task_set.as_ref().unwrap())
        .unwrap();
    let put = |task: &Task| TaskEntry {
        task: ctx.tasks().put(task).unwrap(),
        generator: None,
        prompt: None,
        variant_of: None,
        subject: None,
    };
    let mut members = kept.members.clone();
    let mut real = Vec::new();
    for entry in &kept.members {
        let task = ctx.tasks().get(&entry.task).unwrap();
        let again = task
            .with_instruction(format!("{} Please be careful.", task.instruction))
            .unwrap();
        members.push(put(&again));
        real.push(task);
        real.push(again);
    }
    let mut spec = real[0].environment.spec.clone();
    spec["snapshot"] = serde_json::json!(splinter_core::digest::Digest::of(b"a tree nobody kept"));
    let broken = Task::new(
        real[0].task.kind.clone(),
        Vec::new(),
        Environment::new(Environment::WORKSPACE, spec),
        real[0].instruction.clone(),
        real[0].privileged.clone(),
    )
    .unwrap()
    .with_concepts(["shop/tier99.py"])
    .unwrap();
    members.push(put(&broken));
    let tasks = members.len() - 1;
    exams::record(
        &ctx,
        &ExamRecord {
            dataset: candidate.datasets[0].clone(),
            families: reserve.families.iter().cloned().collect(),
            tasks: Some(
                ctx.tasks()
                    .put_set(&TaskSet {
                        name: "the exam, worded twice, and a task nobody can lay out".into(),
                        members,
                    })
                    .unwrap(),
            ),
        },
    )
    .unwrap();

    // A coding-agent task frozen into the anchor suite is attempted by the
    // worker beside the closed-book tasks, and reported apart.
    let mut frozen = anchor::read_tasks(&anchor_file(&scratch.0, 4)).unwrap();
    frozen.push(real[0].clone());
    anchor::freeze_tasks(&ctx, frozen).unwrap();

    // The candidate mends every module but reads the hidden test of two of
    // the exam's first.
    let peeked = &modules[..2];
    let released = gated(
        &ctx,
        &report,
        Model::new(Mender::peeking_at(peeked), "scripted/peeking-mender"),
    );
    let improvement = released.gate.improvement.measured.as_ref().unwrap();
    let probed = improvement.agentic.unwrap();
    let in_peeked = |task: &Task| {
        peeked.iter().any(|n| {
            task.concepts
                .first()
                .is_some_and(|f| f == &format!("shop/tier{n}.py"))
        })
    };
    let exposed = real.iter().filter(|task| in_peeked(task)).count();
    assert!(
        exposed >= peeked.len(),
        "each of the two families holds at least one task"
    );
    assert_eq!(probed.candidate.exposed, exposed);
    assert_eq!(probed.champion.exposed, 0);
    assert_eq!(probed.candidate.environment_failures, 1);
    assert_eq!(probed.champion.environment_failures, 1);

    // Neither is in a denominator: they are unpaired, and the rest decide.
    // The families that decide are counted once each however many of their
    // tasks there are.
    let c = &improvement.comparison;
    assert_eq!(c.items, tasks + 1);
    assert_eq!(c.unpaired, exposed + 1);
    assert_eq!(c.paired, tasks - exposed);
    assert_eq!(c.candidate_wins, c.paired);
    // Each task is attempted twice over (itself and its rewording), and a
    // family holds as many tasks as the generator admitted for it.
    let per_family = kept.members.len() / EXAM_FAMILIES;
    assert_eq!(c.paired, 2 * per_family * (EXAM_FAMILIES - 2));
    assert_eq!(improvement.sign_test.discordant, EXAM_FAMILIES - 2);

    let anchor = released.gate.anchor.measured.as_ref().unwrap();
    assert_eq!((anchor.version, anchor.workspace_tasks), (2, Some(1)));
    assert!(anchor.agentic.is_some(), "{anchor:#?}");
    assert_eq!(anchor.comparison.items, 5);
}

#[test]
fn the_served_candidate_is_held_to_the_verdicts_of_the_in_process_one() {
    let (scratch, ctx) = context("workspace-served", Brain::Different);
    let repo = scratch.0.join("repo");
    project_of(&repo, &[], Wording::Short, MODULES);
    // A server that answers something else than the candidate does in-process
    // leaves every module as it was: graded differently on all of them.
    let report = practise(&ctx, &repo, &FakeTrainer::mending(true), true);
    let blocked = report.release.as_ref().unwrap();
    assert_eq!(blocked.release, None);
    let serve = blocked.gate.serve.measured.as_ref().unwrap();
    assert_eq!(serve.agreed, 0, "{serve:#?}");
    assert_eq!(serve.disagreed.len(), serve.sampled);
    let first = &serve.disagreed[0];
    assert_eq!(
        (first.in_process_verdict, first.served_verdict),
        (Some(true), Some(false))
    );
    let why = blocked.gate.serve.reason.as_deref().unwrap();
    assert!(why.contains("graded differently"), "{why}");
    // Everything else held: it was the serve check alone.
    assert!(blocked.gate.improvement.passed && blocked.gate.retention.passed);
    assert!(blocked.gate.anchor.passed);
    assert_eq!(
        report
            .stopped
            .as_deref()
            .map(|s| s.matches("serve:").count()),
        Some(1)
    );
}

#[test]
fn a_project_with_no_brain_is_not_measured_rather_than_released() {
    let (scratch, ctx) = context("workspace-no-brain", Brain::Missing);
    let repo = scratch.0.join("repo");
    project_of(&repo, &[], Wording::Short, MODULES);
    let report = practise(&ctx, &repo, &FakeTrainer::mending(true), true);
    let blocked = report.release.as_ref().unwrap();
    assert_eq!(blocked.release, None);
    let serve = &blocked.gate.serve;
    assert!(!serve.passed && serve.measured.is_none());
    assert!(serve.reason.as_deref().unwrap().contains("not measured"));
}
