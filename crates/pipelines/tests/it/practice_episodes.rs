// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn practice task folders
// into graded, replayable agent experience for its clients. If your team
// needs expertise in on-policy experience collection for agent training you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a directory of Harbor-format practice task folders is run as
//! episodes. Every attempt works in a fresh copy of its task's starting
//! files through sven's agent, is graded afterwards by the task's own tests
//! (which the agent never had in reach), and is kept as an experience with a
//! record that says how it ended: graded pass or fail, exposed (it reached
//! for the hidden tests), the verifier failing, or the infrastructure
//! failing. Only a graded attempt has a reward; the verifier or the
//! infrastructure failing is never a reward of zero. Each record names the
//! task by its content address and its folder's digest, the policy, the
//! configuration digest and the attempt's seed, and the trajectory keeps the
//! token ids and log-probabilities the policy sampled.
//!
//! A run is resumable: an attempt already recorded is not run again, and one
//! the infrastructure stopped is. Exam material and a malformed folder are
//! refused before anything runs.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use common::repo::{call, request_of, shell, tool_calls, write};
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::clock::FixedClock;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::Context;
use splinter_pipelines::campaign::endpoint::EpisodeCollector;
use splinter_pipelines::campaign::{CollectRequest, PolicyRef, RolloutCollector};
use splinter_pipelines::episodes::{
    load_collection, run_episodes, EpisodeError, EpisodeOutcome, EpisodeReport, EpisodeRequest,
    EpisodeSettings, PracticeTask, SvenEpisodes,
};
use sven_sdk::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream, SamplingParams,
};

/// A line every task's verifier carries and no agent may ever see.
const MARKER: &str = "hidden-verifier-marker-5c1e";

/// What the scripted policy does with a request.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Habit {
    /// Writes the greeting, and does nothing for any other request.
    Greets,
    /// Reaches for the hidden tests first, then writes the greeting.
    Peeks,
}

/// A policy that writes `hello world` into `greeting.txt` when asked about
/// it and answers anything else without touching a file; every reply
/// carries the token ids and log-probabilities when they are asked for.
/// It fails, as a server that does not serve it does, while `failures` is
/// above zero.
struct Policy {
    habit: Habit,
    /// The task folders' directory, which a peeking policy reads from.
    tasks: PathBuf,
    failures: AtomicUsize,
    asked: AtomicUsize,
    /// The sampling every request was sent with, in arrival order.
    sampled: std::sync::Mutex<Vec<SamplingParams>>,
}

impl Policy {
    fn new(habit: Habit, tasks: &Path) -> Arc<Self> {
        Arc::new(Self {
            habit,
            tasks: tasks.to_path_buf(),
            failures: AtomicUsize::new(0),
            asked: AtomicUsize::new(0),
            sampled: std::sync::Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Policy {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "policy-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        self.sampled.lock().unwrap().push(req.sampling);
        if self
            .failures
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
        {
            anyhow::bail!("the policy is not served (http 404)");
        }
        let calls = tool_calls(&req);
        let greeting = request_of(&req).contains("greeting.txt");
        let mut events = match (greeting, self.habit, calls.len()) {
            (true, Habit::Peeks, 0) => shell(&format!(
                "cat .harbor/tests/test.sh {}/*/tests/test.sh",
                self.tasks.display()
            )),
            (true, Habit::Greets, 0) | (true, Habit::Peeks, 1) => call(
                "c-greet",
                "shell",
                serde_json::json!({"description": "greet", "shell_command": "echo hello world > greeting.txt"}),
            ),
            _ => vec![
                ResponseEvent::TextDelta("Done.".into()),
                ResponseEvent::Done,
            ],
        };
        if req.want_logprobs {
            events.splice(
                0..0,
                [
                    ResponseEvent::PromptTokens {
                        ids: vec![11, 12, 13],
                    },
                    ResponseEvent::Tokens {
                        ids: vec![21, 22],
                        logprobs: vec![-0.5, -0.25],
                    },
                ],
            );
        }
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

/// A practice task folder that ships its starting files: `file` holds
/// `start`, and the verifier rewards `check` (a shell condition) on the tree.
fn task(root: &Path, name: &str, instruction: &str, file: &str, check: &str) -> PathBuf {
    let dir = root.join(name);
    write(&dir.join(format!("workspace/{file}")), "start\n");
    write(&dir.join("instruction.md"), &format!("{instruction}\n"));
    write(
        &dir.join("task.toml"),
        &format!(
            "schema_version = \"1.4\"\n[task]\nname = \"practice/{name}\"\n[metadata]\ncategory = \"{name}\"\nbench_family = \"{name}\"\n"
        ),
    );
    write(
        &dir.join("tests/test.sh"),
        &format!(
            "#!/bin/bash\n# {MARKER}\ncd \"$WORKSPACE_DIR\"\nif {check}; then echo 1 > \"$LOGS_DIR/reward.txt\"; else echo 0 > \"$LOGS_DIR/reward.txt\"; fi\n"
        ),
    );
    dir
}

fn greet(root: &Path) -> PathBuf {
    task(
        root,
        "greet",
        "Write hello world into greeting.txt.",
        "greeting.txt",
        "grep -q 'hello world' greeting.txt",
    )
}

fn count(root: &Path) -> PathBuf {
    task(
        root,
        "count",
        "Make count.txt hold the number three.",
        "count.txt",
        "grep -qx 3 count.txt",
    )
}

fn context(test: &str, policy: Arc<Policy>) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-11-17T09:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(policy, "scripted/policy-1"),
        );
    (scratch, ctx)
}

fn settings(attempts: u32, jobs: usize) -> EpisodeSettings {
    // seed: the records must name the seed the run was given.
    EpisodeSettings::new(attempts, 0x5eed).with_jobs(jobs)
}

fn run(ctx: &Context, tasks: &[PracticeTask], settings: EpisodeSettings) -> EpisodeReport {
    run_episodes(
        ctx,
        &EpisodeRequest {
            tasks,
            policy: &ModelRef::policy_default(),
            settings,
            cancel: CancelToken::new(),
        },
    )
    .unwrap()
}

#[test]
fn practice_attempts_are_graded_outside_the_agent_and_kept_with_their_policy_trace() {
    let tasks_dir = Scratch::new("episodes-tasks");
    let root = tasks_dir.0.join("practice");
    let (greet_dir, _) = (greet(&root), count(&root));
    let policy = Policy::new(Habit::Greets, &root);
    let (_scratch, ctx) = context("episodes", policy.clone());

    let tasks = load_collection(&ctx, &root).unwrap();
    assert_eq!(
        tasks.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["practice/count", "practice/greet"]
    );
    let report = run(&ctx, &tasks, settings(2, 2));
    assert_eq!(report.records.len(), 4);
    assert_eq!((report.ran, report.reused), (4, 0));

    let greeted = tasks.iter().find(|t| t.name == "practice/greet").unwrap();
    for record in &report.records {
        let greets = record.task == greeted.task;
        assert_eq!(
            record.outcome,
            if greets {
                EpisodeOutcome::Passed
            } else {
                EpisodeOutcome::Failed
            },
            "{record:#?}"
        );
        assert_eq!(record.reward, Some(if greets { 1.0 } else { 0.0 }));
        assert_eq!(record.conclusion.as_deref(), Some("success"));
        assert_eq!(record.config, report.config);
        assert_eq!(record.policy.model, "scripted/policy-1");
        assert!(record.trace.provable, "{record:#?}");
        assert!(record.trace.calls > 0);
        assert!(record.usage.turns > 0);
        let experience = ctx
            .experiences()
            .get(record.experience.as_ref().unwrap())
            .unwrap();
        assert_eq!(experience.task.id, record.task);
        let trajectory = serde_json::to_string(&experience.trajectory).unwrap();
        assert!(
            trajectory.contains("\"logprobs\":[-0.5,-0.25]"),
            "{trajectory}"
        );
    }
    let greet_record = report
        .records
        .iter()
        .find(|r| r.task == greeted.task)
        .unwrap();
    assert_eq!(greet_record.source, Some(greeted.source.clone()));
    let seeds: std::collections::BTreeSet<u64> = report.records.iter().map(|r| r.seed).collect();
    assert_eq!(seeds.len(), 4, "every attempt draws from a seed of its own");
    // A policy handed in (as one reached over an API) is seeded too: every
    // request carries the attempt's seed and the exact sampling.
    assert!(report.records.iter().all(|r| r.seeded), "{report:#?}");
    let sampled = policy.sampled.lock().unwrap().clone();
    assert!(!sampled.is_empty());
    for sampling in &sampled {
        assert_eq!(
            (sampling.temperature, sampling.top_p, sampling.top_k),
            (Some(1.0), Some(1.0), Some(0)),
            "temperature one with nothing cut away"
        );
    }
    let sent: std::collections::BTreeSet<u64> = sampled.iter().filter_map(|s| s.seed).collect();
    assert_eq!(sent, seeds, "each request carries its attempt's seed");
    assert!(
        greet_dir.join("tests/test.sh").is_file(),
        "the folder is read, never changed"
    );

    // The same collection run through the campaign's executor seam: groups of
    // graded attempts per task, each with the calls it sampled.
    let executor = SvenEpisodes::new(
        &ctx,
        tasks.clone(),
        ModelRef::policy_default(),
        settings(2, 2),
    );
    let mut collector = EpisodeCollector::new(executor);
    let request = CollectRequest {
        tasks: collector.tasks(),
        attempts: 2,
        policy: PolicyRef::default(),
        seed: 0x5eed, // seed: the same attempts as the run above, so all are reused
    };
    collector.set_policy(&request.policy).unwrap();
    let groups = collector.collect(&request).unwrap();
    assert_eq!(groups.len(), 2);
    for group in &groups {
        assert_eq!(group.rollouts.len(), 2, "{}", group.task);
        let trace = group.rollouts[0].trace.as_ref().unwrap();
        assert!(trace.provable);
        assert_eq!(trace.sequences[0].generation_log_probs, [-0.5, -0.25]);
        assert_eq!(trace.sequences[0].prompt_token_ids, [11, 12, 13]);
    }
}

#[test]
fn a_verifier_that_crashes_is_a_verifier_error_without_a_reward() {
    let tasks_dir = Scratch::new("episodes-crash-tasks");
    let root = tasks_dir.0.join("practice");
    let dir = greet(&root);
    // It judges the starting files (reward 0) and crashes on a greeting.
    write(
        &dir.join("tests/test.sh"),
        &format!(
            "#!/bin/bash\n# {MARKER}\ncd \"$WORKSPACE_DIR\"\ngrep -q 'hello world' greeting.txt && exit 7\necho 0 > \"$LOGS_DIR/reward.txt\"\n"
        ),
    );
    let (_scratch, ctx) = context("episodes-crash", Policy::new(Habit::Greets, &root));

    let tasks = load_collection(&ctx, &root).unwrap();
    let report = run(&ctx, &tasks, settings(1, 1));
    let [record] = &report.records[..] else {
        panic!("one attempt: {report:#?}")
    };
    assert!(
        matches!(record.outcome, EpisodeOutcome::VerifierError { .. }),
        "{record:#?}"
    );
    assert_eq!(
        record.reward, None,
        "a crashed verifier is no reward of zero"
    );
    assert!(record.experience.is_some(), "the episode itself is kept");

    // A verifier that reaches no verdict even on the starting files (a
    // library it sources is not in the folder) can grade nothing: the
    // collection is refused before any attempt, naming the folder.
    let broken_root = tasks_dir.0.join("broken");
    let broken = greet(&broken_root);
    write(
        &broken.join("tests/test.sh"),
        &format!(
            "#!/bin/bash\n# {MARKER}\nsource \"$(dirname \"$0\")/lib/verify.sh\"\ngrade_task\n"
        ),
    );
    let (_scratch, fresh) = context("episodes-broken", Policy::new(Habit::Greets, &broken_root));
    let why = load_collection(&fresh, &broken_root)
        .unwrap_err()
        .to_string();
    assert!(why.contains("greet") && why.contains("NoVerdict"), "{why}");
}

#[test]
fn exam_material_and_malformed_folders_are_refused_before_anything_runs() {
    let tasks_dir = Scratch::new("episodes-refused-tasks");
    let policy = Policy::new(Habit::Greets, &tasks_dir.0);
    let (_scratch, ctx) = context("episodes-refused", policy.clone());

    let sealed = tasks_dir.0.join("sealed");
    let dir = greet(&sealed);
    let toml = std::fs::read_to_string(dir.join("task.toml")).unwrap();
    write(
        &dir.join("task.toml"),
        &format!("{toml}bench_sealed = true\n"),
    );
    count(&sealed);
    let refused = load_collection(&ctx, &sealed).unwrap_err();
    assert!(matches!(refused, EpisodeError::Sealed { .. }), "{refused}");

    let split = tasks_dir.0.join("split");
    greet(&split);
    write(
        &split.join("split.toml"),
        "[sets]\ntrain = [\"count\"]\neval = [\"greet\"]\n",
    );
    let refused = load_collection(&ctx, &split).unwrap_err();
    assert!(matches!(refused, EpisodeError::Sealed { .. }), "{refused}");

    let malformed = tasks_dir.0.join("malformed");
    let dir = greet(&malformed);
    std::fs::remove_file(dir.join("tests/test.sh")).unwrap();
    let refused = load_collection(&ctx, &malformed).unwrap_err();
    assert!(
        matches!(refused, EpisodeError::Malformed { .. }),
        "{refused}"
    );

    assert_eq!(policy.asked.load(Ordering::SeqCst), 0);
    assert!(
        ctx.tasks().list_sets().unwrap().is_empty(),
        "nothing was imported"
    );
}

#[test]
fn a_rerun_skips_finished_attempts_and_redoes_the_ones_that_did_not_finish() {
    let tasks_dir = Scratch::new("episodes-resume-tasks");
    let root = tasks_dir.0.join("practice");
    greet(&root);
    let policy = Policy::new(Habit::Greets, &root);
    let (_scratch, ctx) = context("episodes-resume", policy.clone());
    let tasks = load_collection(&ctx, &root).unwrap();

    // The policy is not served for the first attempt: no experience, no
    // reward.
    policy.failures.store(1_000, Ordering::SeqCst);
    let first = run(&ctx, &tasks, settings(1, 1));
    assert!(
        matches!(
            first.records[0].outcome,
            EpisodeOutcome::Infrastructure { .. }
        ),
        "{first:#?}"
    );
    assert_eq!(first.records[0].reward, None);
    assert_eq!(first.records[0].experience, None);

    // Back up, and asked for more attempts: the failed one is redone, the
    // new one runs; then nothing is left to run.
    policy.failures.store(0, Ordering::SeqCst);
    let second = run(&ctx, &tasks, settings(2, 1));
    assert_eq!((second.ran, second.reused), (2, 0));
    assert!(second
        .records
        .iter()
        .all(|r| r.outcome == EpisodeOutcome::Passed));

    let asked = policy.asked.load(Ordering::SeqCst);
    let third = run(&ctx, &tasks, settings(2, 1));
    assert_eq!((third.ran, third.reused), (0, 2));
    assert_eq!(
        third.records, second.records,
        "the recorded attempts, unchanged"
    );
    assert_eq!(
        policy.asked.load(Ordering::SeqCst),
        asked,
        "the policy is not asked again"
    );

    // Loading the collection again finds the tasks it imported.
    let again = load_collection(&ctx, &root).unwrap();
    assert_eq!(
        again.iter().map(|t| &t.task).collect::<Vec<_>>(),
        tasks.iter().map(|t| &t.task).collect::<Vec<_>>()
    );
}

#[test]
fn the_agent_cannot_read_the_tasks_tests_and_reaching_for_them_fails_the_attempt() {
    let tasks_dir = Scratch::new("episodes-peek-tasks");
    let root = tasks_dir.0.join("practice");
    greet(&root);
    let (_scratch, ctx) = context("episodes-peek", Policy::new(Habit::Peeks, &root));
    let tasks = load_collection(&ctx, &root).unwrap();

    let report = run(&ctx, &tasks, settings(1, 1));
    let [record] = &report.records[..] else {
        panic!("one attempt: {report:#?}")
    };
    let experience = ctx
        .experiences()
        .get(record.experience.as_ref().unwrap())
        .unwrap();
    let trajectory = serde_json::to_string(&experience.trajectory).unwrap();
    assert!(
        trajectory.contains("cat .harbor/tests/test.sh"),
        "it did ask"
    );
    assert!(
        !trajectory.contains(MARKER),
        "the hidden tests never reached the agent: {trajectory}"
    );
    assert_eq!(record.outcome, EpisodeOutcome::Exposed, "{record:#?}");
    assert_eq!(record.reward, Some(0.0));
}
