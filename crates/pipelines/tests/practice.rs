// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: `practice` turns a git revision of a repository into a recorded
//! run, stage by stage, with a scripted agent.
//!
//! * every stage is recorded, in order, by the engine;
//! * the exam's families are decided before any attempt, are never solved
//!   and never appear in the training set (the leakage gate), and stay the
//!   exam in a later run;
//! * an experience whose trajectory shows the solver reached for the hidden
//!   tests is decided fail by the exposure audit, counted by reason, and
//!   never trained on, whatever its patch did;
//! * a second run over the same snapshot generates no task that is already
//!   held, and so runs no attempt;
//! * a red baseline stops the run before any task is made.
//!
//! The policy is a scripted agent that edits a tiny Python project through
//! sven's own tools; the tests run in the process sandbox; training is a
//! test double, since training needs weights.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use common::gate::{put_base, FakeTrainer};
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_core::clock::FixedClock;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::frontier::PassAtK;
use splinter_pipelines::practice::{practice, PracticeReport, PracticeRequest};
use splinter_store::runs::{read_run, RunStatus};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream, Role,
};

/// How many modules the project has; each is a family of tasks.
const MODULES: usize = 6;

/// The modules whose solver reaches for the hidden tests before it fixes
/// the module.
const PEEKING: [usize; 3] = [0, 1, 2];

fn module(n: usize) -> String {
    format!(
        "def tier{n}(units):\n    \"\"\"The price tier of an order.\"\"\"\n    if units > 10:\n        \
         return 20 + {n}\n    return 5\n"
    )
}

/// The tests of module `n`. A request lists the symptoms the tests that
/// catch a fault name, so another wording of the names is another request
/// for the same file; one that shares most of its words with an earlier one
/// would be refused as a repeat of it. A name of eight words or more would
/// quote the tests in the request, which is refused too.
fn test_of(n: usize, wording: Wording) -> String {
    let (large, small) = match wording {
        Wording::Short => (
            "large_orders_pay_the_large_tier",
            "small_orders_pay_the_small_tier",
        ),
        Wording::Long => (
            "bulk_buyers_pay_the_wholesale_rate",
            "casual_buyers_pay_the_retail_rate",
        ),
    };
    format!(
        "import unittest\n\nfrom shop.tier{n} import tier{n}\n\n\nclass Tier{n}Test(unittest.TestCase):\n    \
         def test_tier{n}_{large}(self):\n        self.assertEqual(tier{n}(11), 20 + {n})\n\n    \
         def test_tier{n}_{small}(self):\n        self.assertEqual(tier{n}(10), 5)\n"
    )
}

/// How the tests of a project name what they check.
#[derive(Clone, Copy)]
enum Wording {
    Short,
    Long,
}

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
        .args(["-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"])
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// The project in a git repository under `dir`, with `extra` files beside.
fn project(dir: &Path, extra: &[(&str, &str)]) {
    project_worded(dir, extra, Wording::Short);
}

/// [`project`] whose tests are worded as `wording` says.
fn project_worded(dir: &Path, extra: &[(&str, &str)], wording: Wording) {
    std::fs::create_dir_all(dir).unwrap();
    git(dir, &["init", "-q", "-b", "main"]);
    write(
        &dir.join("pyproject.toml"),
        "[project]\nname = \"shop\"\nversion = \"1\"\n",
    );
    write(&dir.join("shop/__init__.py"), "");
    write(&dir.join("tests/__init__.py"), "");
    for n in 0..MODULES {
        write(&dir.join(format!("shop/tier{n}.py")), &module(n));
        write(
            &dir.join(format!("tests/test_tier{n}.py")),
            &test_of(n, wording),
        );
    }
    for (path, text) in extra {
        write(&dir.join(path), text);
    }
    git(dir, &["add", "-A", "-f"]);
    git(dir, &["commit", "-q", "-m", "shop"]);
}

/// A scripted agent. Asked to change a module it cannot see, it restores the
/// module it infers from the request (a request lists the symptoms of the
/// failing tests, and each is named for its module). Of the two attempts at a task
/// it makes, the first fixes the module and the second looks and does
/// nothing, so each task is solved about half the time. For the
/// [`PEEKING`] modules the fixing attempt first reads the test file the
/// request is about, which is withheld from the tree.
struct Agent {
    /// Attempts begun, by request.
    begun: Mutex<BTreeMap<String, u32>>,
}

impl Agent {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            begun: Mutex::new(BTreeMap::new()),
        })
    }
}

fn call(id: &str, name: &str, arguments: serde_json::Value) -> Vec<ResponseEvent> {
    vec![
        ResponseEvent::ToolCall {
            index: 0,
            id: id.into(),
            name: name.into(),
            arguments: arguments.to_string(),
        },
        ResponseEvent::Done,
    ]
}

fn shell(command: &str) -> Vec<ResponseEvent> {
    call(
        "c-shell",
        "shell",
        serde_json::json!({"description": "look", "shell_command": command}),
    )
}

/// The first user turn of a conversation: the request.
fn request_of(req: &CompletionRequest) -> String {
    req.messages
        .iter()
        .find(|m| m.role == Role::User)
        .and_then(|m| m.as_text().map(str::to_string))
        .unwrap_or_default()
}

/// The module a request is about: the number in the symptoms it lists, which
/// are named for the tests that catch them.
fn module_of(request: &str) -> Option<usize> {
    let at = request.find("- tier")? + "- tier".len();
    let digits: String = request[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn tool_calls(req: &CompletionRequest) -> Vec<String> {
    req.messages
        .iter()
        .filter_map(|m| match &m.content {
            MessageContent::ToolCall { function, .. } => Some(function.arguments.clone()),
            _ => None,
        })
        .collect()
}

#[async_trait::async_trait]
impl ModelProvider for Agent {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "agent-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let request = request_of(&req);
        let calls = tool_calls(&req);
        let asked_about = module_of(&request);
        let events = match (calls.len(), asked_about) {
            (0, Some(n)) => {
                let attempt = {
                    let mut begun = self.begun.lock().unwrap();
                    let seen = begun.entry(request.clone()).or_insert(0);
                    *seen += 1;
                    *seen
                };
                if attempt % 2 == 1 && PEEKING.contains(&n) {
                    shell(&format!("cat tests/test_tier{n}.py # fix"))
                } else if attempt % 2 == 1 {
                    shell("true # fix")
                } else {
                    shell("true # look")
                }
            }
            (1, Some(n)) if calls[0].contains("# fix") => call(
                "c-fix",
                "write_file",
                serde_json::json!({"path": format!("shop/tier{n}.py"), "text": module(n), "append": false}),
            ),
            _ => vec![ResponseEvent::TextDelta("done".into()), ResponseEvent::Done],
        };
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

/// A context over a scratch state root whose `policy:default` is the
/// scripted agent.
fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-13T08:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(Agent::new(), "scripted/agent-1"),
        );
    put_base(&ctx);
    (scratch, ctx)
}

fn request(repo: &Path) -> PracticeRequest {
    PracticeRequest {
        repository: repo.to_path_buf(),
        revision: "HEAD".into(),
        max_tasks: Some(32),
        mutants_per_file: Some(1),
        exam_families: Some(1),
        pass_at_k: PassAtK::new(2, None, None),
        no_release: true,
        ..PracticeRequest::default()
    }
}

fn run(ctx: &Context, request: &PracticeRequest) -> (String, PracticeReport) {
    let recorded = practice(ctx, request, &FakeTrainer::knowing(&[])).unwrap();
    (recorded.run, recorded.report)
}

/// The family of each record of a dataset file, and how many records.
fn groups_of(dataset: &PathBuf) -> Vec<String> {
    std::fs::read_to_string(dataset)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            record["metadata"]["group"].as_str().unwrap().to_string()
        })
        .collect()
}

#[test]
fn a_repository_is_practised_as_one_recorded_run_that_never_trains_on_its_exam() {
    let (scratch, ctx) = context("practice");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);

    let (run_id, report) = run(&ctx, &request(&repo));
    assert_eq!(report.stopped, None, "{report:#?}");

    // Every stage is recorded, in order, by the engine.
    let recorded = read_run(ctx.workspace(), &run_id).unwrap();
    assert_eq!(recorded.status, RunStatus::Completed);
    let stages: Vec<&str> = recorded.stages.iter().map(|s| s.stage.as_str()).collect();
    assert_eq!(
        stages,
        [
            "policy", "snapshot", "profile", "tasks", "reserve", "solve", "verify", "frontier",
            "dataset", "train"
        ]
    );

    let profile = report.profile.as_ref().unwrap();
    assert_eq!(profile.test_command, ["python3", "-m", "unittest"]);
    assert_eq!(profile.tests, 2 * MODULES);
    let tasks = report.tasks.as_ref().unwrap();
    assert!(tasks.admitted >= MODULES / 2, "{tasks:#?}");

    // The exam is decided before any attempt: its tasks are set apart and
    // are not among those solved.
    let reserve = report.reserve.as_ref().unwrap();
    assert_eq!(reserve.families.len(), 1, "{reserve:#?}");
    assert!(reserve.exam_tasks > 0 && reserve.training_tasks > 0);
    let solved = report.solve.as_ref().unwrap();
    assert_eq!(solved.solved, 2 * reserve.training_tasks, "{solved:#?}");
    let exam_tasks = ctx
        .tasks()
        .get_set(reserve.exam_task_set.as_ref().unwrap())
        .unwrap()
        .members;
    let exam_ids: BTreeSet<_> = exam_tasks.iter().map(|e| e.task.clone()).collect();
    for id in ctx.experiences().list().unwrap() {
        let attempted = ctx.experiences().get(&id).unwrap().task.id;
        assert!(!exam_ids.contains(&attempted), "an exam task was attempted");
    }

    // The hidden tests are in no tree an agent works in: an agent that asks
    // for one is told there is none, and nothing of them is in what it saw.
    let mut peeked = 0;
    for id in ctx.experiences().list().unwrap() {
        let trajectory =
            serde_json::to_string(&ctx.experiences().get(&id).unwrap().trajectory).unwrap();
        if trajectory.contains("cat tests/test_tier") {
            peeked += 1;
            assert!(
                !trajectory.contains("assertEqual"),
                "a hidden test was in the tree"
            );
        }
    }
    assert!(
        peeked > 0,
        "the scripted agent never reached for a hidden test"
    );

    // The exposure audit failed the experiences that read the hidden tests,
    // by reason, and they were not trained on.
    let verified = report.verify.as_ref().unwrap();
    assert!(verified.exposed > 0, "{verified:#?}");
    assert_eq!(
        verified.exposed_by.get("hidden_path").copied(),
        Some(verified.exposed),
        "{verified:#?}"
    );
    let dataset = report.dataset.as_ref().unwrap();
    assert!(dataset.records > 0, "{dataset:#?}");
    // The trainer's own parser accepts every record, tools and all.
    splinter_model::train::validate_dataset(&dataset.path).unwrap();
    let groups = groups_of(&dataset.path);
    let peeking: BTreeSet<String> = PEEKING.iter().map(|n| format!("shop/tier{n}.py")).collect();
    assert!(
        groups.iter().all(|g| !peeking.contains(g)),
        "an exposed experience was trained on: {groups:?}"
    );

    // The leakage gate: no record of the training set is of an exam family.
    assert!(!groups.is_empty());
    for group in &groups {
        assert!(
            !reserve.families.contains(group),
            "{group} is an exam family"
        );
    }
    // The trajectories are trained action by action, with the tools the
    // agent was offered.
    let first: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(&dataset.path)
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert!(
        first["tools"].as_array().is_some_and(|t| !t.is_empty()),
        "{first}"
    );
    assert!(report.candidate.is_some());
}

#[test]
fn a_second_run_over_the_same_snapshot_makes_no_task_it_already_holds() {
    let (scratch, ctx) = context("practice-again");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let asked = request(&repo);

    let (_, first) = run(&ctx, &asked);
    let made = first.tasks.as_ref().unwrap().admitted;
    assert!(made > 0);
    let held_tasks = ctx.tasks().list().unwrap().len();

    let (second_run, second) = run(&ctx, &asked);
    let again = second.tasks.as_ref().unwrap();
    assert_eq!(again.admitted, 0, "{again:#?}");
    assert!(again.held_before >= made);
    assert!(
        again.refused.values().sum::<usize>() > 0,
        "the candidates it considered were turned away: {again:#?}"
    );
    assert_eq!(
        ctx.tasks().list().unwrap().len(),
        held_tasks,
        "no task was stored twice"
    );
    assert!(second.solve.is_none(), "nothing new, so nothing attempted");
    assert!(second
        .stopped
        .as_deref()
        .is_some_and(|why| why.contains("already hold")));
    let recorded = read_run(ctx.workspace(), &second_run).unwrap();
    assert_eq!(recorded.stages.last().unwrap().stage, "tasks");
}

#[test]
fn the_families_one_run_reserved_stay_the_exam_of_the_next() {
    let (scratch, ctx) = context("practice-exam");
    let first_repo = scratch.0.join("first");
    project(&first_repo, &[]);
    let (_, first) = run(&ctx, &request(&first_repo));
    let reserved: BTreeSet<String> = first
        .reserve
        .as_ref()
        .unwrap()
        .families
        .iter()
        .cloned()
        .collect();
    assert_eq!(reserved.len(), 1);

    // Another repository with files of the same names, and one more: its
    // requests are its own words, so none is a repeat of the first's. The
    // family the first run reserved is the exam still.
    let second_repo = scratch.0.join("second");
    project_worded(&second_repo, &[], Wording::Long);
    write(
        &second_repo.join(format!("shop/tier{MODULES}.py")),
        &module(MODULES),
    );
    write(
        &second_repo.join(format!("tests/test_tier{MODULES}.py")),
        &test_of(MODULES, Wording::Long),
    );
    git(&second_repo, &["add", "-A", "-f"]);
    git(&second_repo, &["commit", "-q", "-m", "more"]);
    let mut asked = request(&second_repo);
    asked.exam_families = Some(1);
    let (_, second) = run(&ctx, &asked);
    let reserve = second
        .reserve
        .as_ref()
        .unwrap_or_else(|| panic!("no exam was reserved: {second:#?}"));
    assert_eq!(reserve.reserved_before, 1, "{reserve:#?}");
    for family in &reserved {
        assert!(
            reserve.families.contains(family),
            "{family} was an exam family and must stay one: {reserve:#?}"
        );
    }
    let dataset = second
        .dataset
        .as_ref()
        .unwrap_or_else(|| panic!("no dataset: {second:#?}"));
    let groups = groups_of(&dataset.path);
    assert!(!groups.is_empty());
    for group in groups {
        assert!(!reserved.contains(&group), "{group} was trained on");
    }
}

#[test]
fn a_red_baseline_stops_the_run_before_any_task_is_made() {
    let (scratch, ctx) = context("practice-red");
    let repo = scratch.0.join("repo");
    let red = "import unittest\n\n\nclass Red(unittest.TestCase):\n    def test_broken(self):\n        self.fail('no')\n";
    project(&repo, &[("tests/test_red.py", red)]);

    let (_, report) = run(&ctx, &request(&repo));
    assert!(
        report.profile.is_none() && report.tasks.is_none(),
        "{report:#?}"
    );
    let why = report.stopped.unwrap();
    assert!(
        why.contains("baseline") && why.contains("RedBaseline"),
        "{why}"
    );
}

#[test]
fn a_revision_that_is_not_one_is_refused_by_name() {
    let (scratch, ctx) = context("practice-revision");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let mut asked = request(&repo);
    asked.revision = "no-such-revision".into();
    let error = practice(&ctx, &asked, &FakeTrainer::knowing(&[])).unwrap_err();
    assert!(error.to_string().contains("no-such-revision"), "{error}");
}
