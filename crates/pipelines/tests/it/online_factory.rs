// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning that trains a
// served policy on its own rollouts while it runs, for its clients. If your
// team needs expertise in asynchronous policy optimisation, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec (GPU-free, end to end): a recipe probes its tasks, keeps the ones
//! the policy solves only sometimes (`dataset.rlvr`), trains a candidate
//! online on them (`train.online`: sven rollouts of a scripted policy
//! against the version a scripted brain streaming job serves, every group
//! streamed to the job) and evaluates it. The controller is killed in the
//! middle of the online stage, once the learner has trained a step, and the
//! run resumed: the stage finishes from
//! its ledger, no group is trained twice, every committed group is in the
//! stage's experience manifest, and the candidate (the newest committed
//! version, load-checked) flows into the evaluation unchanged.

#![allow(clippy::unwrap_used)]

use crate::common;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

#[allow(dead_code)]
#[path = "../../../factory/tests/it/fakes.rs"]
mod fakes;

use common::repo::{call, request_of, tool_calls, write};
use common::{config, Scratch};
use fakes::{FakeBackend, Outcome};
use serde_json::Value;
use splinter_agent::solve::Model;
use splinter_core::clock::FixedClock;
use splinter_core::model_ref::ModelRef;
use splinter_factory::connections::Connections;
use splinter_factory::resolve::resolve;
use splinter_factory::run::{Factory, RunError, Services};
use splinter_factory::training::BrainTrainer;
use splinter_model::endpoint::BrainEndpoints;
use splinter_model::scripted_endpoint::{Card, ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::ArtifactStore;
use splinter_orchestrator::Context;
use splinter_pipelines::episodes::online::{FactoryRollouts, OnlinePolicy};
use splinter_pipelines::episodes::EpisodeSettings;
use sven_sdk::model::{
    CompletionRequest, LogprobKind, ModelProvider, ResponseEvent, ResponseStream, ServedIdentity,
};

const MARKER: &str = "hidden-online-marker-41b2";

/// A policy that solves a task when its request's seed is even: it writes
/// what the task asks for, then says it is done. Every reply carries the
/// policy trace and the identity of the version that decoded it.
struct Policy;

/// The output bound every request that asked for the policy trace (an online
/// rollout's) carried.
static ROLLOUT_BOUNDS: std::sync::Mutex<Vec<Option<u32>>> = std::sync::Mutex::new(Vec::new());

#[async_trait::async_trait]
impl ModelProvider for Policy {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "policy-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        if req.want_logprobs {
            ROLLOUT_BOUNDS
                .lock()
                .unwrap()
                .push(req.max_output_tokens_override);
        }
        let solves = req.sampling.seed.is_some_and(|s| s % 2 == 0);
        let asked = request_of(&req);
        let command = if asked.contains("greeting.txt") {
            "echo hello world > greeting.txt"
        } else {
            "echo 3 > count.txt"
        };
        let answer = if solves && tool_calls(&req).is_empty() {
            call(
                "c-1",
                "shell",
                serde_json::json!({"description": "do it", "shell_command": command}),
            )
        } else {
            vec![
                ResponseEvent::TextDelta("Done.".into()),
                ResponseEvent::Done,
            ]
        };
        let prompt: Vec<i64> = (0..i64::try_from(req.messages.len()).unwrap() + 1).collect();
        let mut events = vec![
            ResponseEvent::PromptTokens { ids: prompt },
            ResponseEvent::Tokens {
                ids: vec![21, 22],
                logprobs: vec![-0.5, -0.25],
            },
            ResponseEvent::Identity(ServedIdentity {
                policy_version: Some("0".into()),
                tokenizer_digest: Some(format!("sha256:{}", "ab".repeat(32))),
                logprob_kind: Some(LogprobKind::Policy),
                logprob_temperature: req.sampling.temperature,
                ..ServedIdentity::default()
            }),
        ];
        events.extend(answer);
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

fn task(root: &Path, id: &str, instruction: &str, file: &str, check: &str) {
    let dir = root.join("tasks/rl").join(id);
    write(&dir.join(format!("workspace/{file}")), "start\n");
    write(&dir.join("instruction.md"), &format!("{instruction}\n"));
    let name = id.replace('/', "-");
    write(
        &dir.join("task.toml"),
        &format!("schema_version = \"1.4\"\n[task]\nname = \"rl/{name}\"\n[metadata]\nbench_family = \"{name}\"\n"),
    );
    write(
        &dir.join("tests/test.sh"),
        &format!(
            "#!/bin/bash\n# {MARKER}\ncd \"$WORKSPACE_DIR\"\nif {check}; then echo 1 > \"$LOGS_DIR/reward.txt\"; else echo 0 > \"$LOGS_DIR/reward.txt\"; fi\n"
        ),
    );
}

// seed: a recipe states every seed; the scripted policy's choices follow the
// rollouts' own seeds, which derive from it.
const RECIPE: &str = r#"
schema: splinter.recipe/v2
name: online
inputs:
  course: { kind: files, path: collections/rl.yaml }
models:
  student: "brain:Qwen/Toy"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: train } }
  - id: probe
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: student, environment: environments/rl.yaml, attempts: 2, seed: 11 }
  - id: rlvr
    uses: dataset.rlvr
    inputs: { tasks: $stages.tasks.tasks, attempts: [$stages.probe.attempts] }
  - { id: learnable, uses: tasks.pin, inputs: { collection: $stages.rlvr.tasks }, with: { split: train } }
  - id: online
    uses: train.online
    inputs: { tasks: $stages.learnable.tasks }
    with:
      base: student
      groups: 6
      group_size: 4
      max_concurrent_attempts: 4
      max_staleness: 100
      rollout_tokens: 4
      step_tokens: 8
      queue_tokens: 64
      checkpoint_every: 1
      max_output_tokens: 48
      seed: 3
  - id: eval
    uses: evaluation
    inputs: { candidate: $stages.online.model }
    with: { plan: eval/dev.yaml, agent: sven, baseline: student }
"#;

fn repo(root: &Path) {
    task(
        root,
        "train/greet",
        "Write hello world into greeting.txt.",
        "greeting.txt",
        "grep -q 'hello world' greeting.txt",
    );
    task(
        root,
        "train/count",
        "Make count.txt hold the number three.",
        "count.txt",
        "grep -qx 3 count.txt",
    );
    task(
        root,
        "eval/held",
        "Write hello world into greeting.txt.",
        "greeting.txt",
        "grep -q 'hello world' greeting.txt",
    );
    write(
        &root.join("collections/rl.yaml"),
        "schema: splinter.collection/v1\nname: rl\nroot: tasks/rl\n",
    );
    write(
        &root.join("environments/rl.yaml"),
        "schema: splinter.environment/v1\nsandbox: { provider: proot, resources: { timeout: 10m } }\nnetwork: { mode: open }\n",
    );
    // seed: the plan must state its seed; the fake backend's outcomes are scripted.
    write(
        &root.join("eval/dev.yaml"),
        "schema: splinter.evalplan/v1\ncollection: collections/rl.yaml\nsplit: eval\n\
         environment: environments/rl.yaml\nattempts: 2\nseed: 5\ndecision: { alpha: 0.05 }\n",
    );
}

fn lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn an_online_candidate_is_trained_killed_resumed_and_evaluated_with_no_group_trained_twice() {
    let scratch = Scratch::new("online-factory");
    let root = scratch.0.join("repo");
    repo(&root);
    let store = scratch.0.join("artifacts");
    let adapters = scratch.0.join("adapters");
    std::fs::create_dir_all(&store).unwrap();
    std::fs::create_dir_all(&adapters).unwrap();
    let service = ScriptedEndpoint::start(vec![Card::base("Qwen/Toy")])
        .with_training(JobScript::over(store.clone()))
        .serving_adapters(adapters.clone());
    let endpoints = BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY)
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&store).unwrap())
        .with_adapter_directory(adapters)
        .with_refused_wait(Duration::from_millis(50));
    let trainer = BrainTrainer::new(endpoints, scratch.0.join("models"));
    let connections = Connections::parse(&format!(
        "connections:\n  brain: {{ protocol: openai, url: \"{}\" }}\n",
        service.url()
    ))
    .unwrap();
    let resolved = resolve(&root, RECIPE, connections).unwrap();

    let policy = Arc::new(Policy);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-11-25T09:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(policy.clone(), "scripted/policy-1"),
        );
    // seed: the probe's outcomes are scripted; episodes draw their own seeds.
    let rollouts = FactoryRollouts::new(&ctx, EpisodeSettings::new(1, 0), |_served| {
        Ok(OnlinePolicy::Reference(ModelRef::policy_default()))
    });
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Fail)
    };
    for id in ["train/greet", "train/count"] {
        backend.script(id, &[Outcome::Pass, Outcome::Fail]);
    }
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: Some(&rollouts),
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let run = scratch.0.join("run");

    // The controller is killed in the middle of the online stage, once the
    // learner has trained something.
    let killed = Factory::open(&resolved, &run, services, "test")
        .unwrap()
        .with_cancel(|| !service.jobs().stream_runs.applied().is_empty())
        .run();
    assert!(matches!(killed, Err(RunError::Cancelled)), "{killed:?}");
    let before = service.jobs().stream_runs.applied();
    assert!(
        !before.is_empty(),
        "some groups were trained before the kill"
    );

    let manifest = Factory::open(&resolved, &run, services, "test")
        .unwrap()
        .run()
        .unwrap();
    let ops: Vec<&str> = manifest
        .stages
        .iter()
        .map(|s| s.operation.as_str())
        .collect();
    assert!(
        ops.contains(&"train.online") && ops.contains(&"evaluation"),
        "{ops:?}"
    );

    // A rollout's reply is bounded as the stage says: a runaway sample ends truncated, not at the context's end.
    let bounds = ROLLOUT_BOUNDS.lock().unwrap().clone();
    assert!(
        !bounds.is_empty() && bounds.iter().all(|b| b.is_some_and(|b| b <= 48)),
        "{bounds:?}"
    );

    let applied = service.jobs().stream_runs.applied();
    let unique: std::collections::BTreeSet<&String> = applied.iter().collect();
    assert_eq!(
        unique.len(),
        applied.len(),
        "no group trained twice: {applied:?}"
    );
    let committed = service.jobs().stream_runs.committed();
    assert_eq!(
        committed.len(),
        applied.len(),
        "every applied group committed"
    );

    let view = run.join("online");
    let experiences = lines(&view.join("experiences.jsonl"));
    let mut manifest_keys: Vec<String> = experiences
        .iter()
        .map(|e| e["group"].as_str().unwrap().to_string())
        .collect();
    manifest_keys.sort();
    assert_eq!(
        manifest_keys, committed,
        "the manifest is the committed groups"
    );
    // The live group log holds every group's commit once, through the kill.
    let mut logged: Vec<String> = lines(&view.join("groups.jsonl"))
        .iter()
        .filter(|l| l["state"] == "committed")
        .map(|l| l["group"].as_str().unwrap().to_string())
        .collect();
    logged.sort();
    assert_eq!(logged, committed, "the group log names each commit once");
    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(view.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["groups"]["started"], 6, "{report:#}");
    assert_eq!(
        report["groups"]["committed"].as_u64().unwrap() as usize,
        committed.len()
    );
    let filtered = &report["groups"]["filtered"];
    assert_eq!(
        6 - committed.len() as u64,
        filtered["zero_variance"].as_u64().unwrap()
            + filtered["not_exact"].as_u64().unwrap()
            + filtered["unscored"].as_u64().unwrap(),
        "every group not trained was filtered: {report:#}"
    );
    assert!(report["frac_zero_std"].is_number() && report["reward_mean"].is_number());
    // Training health: each step the learner reported, and per policy version what its rollouts came to,
    // the same figures `splinter observe` shows of the run.
    let health = &report["health"];
    assert!(
        health["steps"].as_array().is_some_and(|s| !s.is_empty()),
        "{health:#}"
    );
    let versions = health["versions"].as_object().unwrap();
    assert!(!versions.is_empty(), "{health:#}");
    for summary in versions.values() {
        assert!(summary["false_completion_rate"].is_number(), "{summary}");
        assert!(summary["repeated_call_rate"].is_number(), "{summary}");
    }
    let observed = splinter_factory::observe::observe(
        &run,
        // seed: the bootstrap intervals of evaluations, not read here.
        &splinter_factory::observe::Options {
            seed: 1,
            utilisation: None,
        },
    )
    .unwrap();
    assert_eq!(observed.online.len(), 1);
    assert!(observed.online[0].finished);
    assert_eq!(
        serde_json::to_value(&observed.online[0].versions).unwrap(),
        health["versions"]
    );
    assert!(report["experience_utilisation"].as_f64().unwrap() > 0.0);
    // Every published version names the run-level identity that served it, and
    // the report carries the run's revision manifest: the server it reached (its
    // record contract this build's), the base and tokenizer it named.
    let versions = report["versions"]
        .as_array()
        .expect("the published versions");
    assert!(!versions.is_empty(), "{report:#}");
    for version in versions {
        let lineage = &version["lineage"];
        assert_eq!(lineage["update"], version["step"], "{version}");
        assert_eq!(lineage["adapter_digest"], version["digest"], "{version}");
        assert!(lineage["epoch"].as_u64().unwrap() >= 1, "{version}");
        assert!(
            lineage["run"].as_str().unwrap().starts_with("online-"),
            "{version}"
        );
    }
    let revisions = &report["revisions"];
    assert_eq!(
        revisions["training_records"]["digest"],
        splinter_core::revision::RECORD_CONTRACT_DIGEST
    );
    assert_eq!(
        revisions["served"][0]["training_records"]["digest"],
        splinter_core::revision::RECORD_CONTRACT_DIGEST,
        "{revisions:#}"
    );
    assert_eq!(
        revisions["models"],
        serde_json::json!([{"base_digest": splinter_model::scripted_stream::SCRIPTED_BASE_DIGEST,
                "tokenizer_digest": splinter_model::scripted_stream::SCRIPTED_TOKENIZER_DIGEST}])
    );
    assert!(revisions["inputs"]
        .as_array()
        .is_some_and(|i| !i.is_empty()));
    let product: Value = serde_json::from_str(
        &std::fs::read_to_string(view.join(splinter_core::revision::REVISIONS_FILE)).unwrap(),
    )
    .unwrap();
    assert_eq!(&product, revisions, "the product holds the same manifest");
    let candidate: Value =
        serde_json::from_str(&std::fs::read_to_string(view.join("candidate.json")).unwrap())
            .unwrap();
    assert_eq!(candidate["kind"], "adapter");
    assert_eq!(candidate["load_checked"], true);
    assert!(view.join("adapter.safetensors").is_file());
    assert!(run.join("eval").is_dir(), "the candidate was evaluated");
}

// seed: a recipe states every seed; the scripted policy's choices follow the rollouts' own seeds.
const WATCHED: &str = r#"
schema: splinter.recipe/v2
name: watched
inputs:
  course: { kind: files, path: collections/rl.yaml }
models:
  student: "brain:Qwen/Toy"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: train } }
  - { id: monitor, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: monitor } }
  - { id: confirm, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: confirm } }
  - id: online
    uses: train.online
    inputs: { tasks: $stages.tasks.tasks, monitor: $stages.monitor.tasks, confirm: $stages.confirm.tasks }
    with:
      base: student
      groups: 6
      group_size: 4
      max_concurrent_attempts: 4
      max_staleness: 100
      rollout_tokens: 4
      step_tokens: 8
      queue_tokens: 64
      checkpoint_every: 1
      max_output_tokens: 48
      seed: 3
      watchdog:
        bounds: { guard_skip_rate: 1.0, ratio_max: 1000, truncation_rate: 1.0, infra_failure_rate: 1.0,
                  window: 8, min_rollouts: 1000 }
        drift: { slack: 0.05, threshold: 10, reference_window: 3 }
        test: { delta: 0.25, alpha: 0.05, beta: 0.1, max_discordant: 4 }
        evaluate_every: 1
        monitor_size: 2
        max_uses: 50
        max_rollbacks: 1
        cooldown: 0.5
"#;

#[test]
fn a_watched_run_whose_policy_does_not_regress_is_evaluated_and_never_rolled_back() {
    let scratch = Scratch::new("online-watched");
    let root = scratch.0.join("repo");
    repo(&root);
    for (id, file) in [
        ("monitor/one", "greeting.txt"),
        ("monitor/two", "count.txt"),
        ("confirm/sealed", "greeting.txt"),
    ] {
        let (instruction, check) = if file == "count.txt" {
            (
                "Make count.txt hold the number three.",
                "grep -qx 3 count.txt",
            )
        } else {
            (
                "Write hello world into greeting.txt.",
                "grep -q 'hello world' greeting.txt",
            )
        };
        task(&root, id, instruction, file, check);
    }
    let store = scratch.0.join("artifacts");
    let adapters = scratch.0.join("adapters");
    std::fs::create_dir_all(&store).unwrap();
    std::fs::create_dir_all(&adapters).unwrap();
    let service = ScriptedEndpoint::start(vec![Card::base("Qwen/Toy")])
        .with_training(JobScript::over(store.clone()))
        .serving_adapters(adapters.clone());
    let endpoints = BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY)
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&store).unwrap())
        .with_adapter_directory(adapters)
        .with_refused_wait(Duration::from_millis(50));
    let trainer = BrainTrainer::new(endpoints, scratch.0.join("models"));
    let connections = Connections::parse(&format!(
        "connections:\n  brain: {{ protocol: openai, url: \"{}\" }}\n",
        service.url()
    ))
    .unwrap();
    let resolved = resolve(&root, WATCHED, connections).unwrap();
    let policy = Arc::new(Policy);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-11-27T09:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(policy.clone(), "scripted/policy-1"),
        );
    // seed: episodes draw their own seeds from the stage's.
    let rollouts = FactoryRollouts::new(&ctx, EpisodeSettings::new(1, 0), |_served| {
        Ok(OnlinePolicy::Reference(ModelRef::policy_default()))
    });
    let backend = FakeBackend::with(Outcome::Fail);
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: Some(&rollouts),
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let run = scratch.0.join("run");
    Factory::open(&resolved, &run, services, "test")
        .unwrap()
        .run()
        .unwrap();
    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("online/report.json")).unwrap())
            .unwrap();
    let watchdog = &report["watchdog"];
    assert!(
        watchdog["comparisons"].as_u64().unwrap() >= 1,
        "the published versions were evaluated against the anchor: {watchdog:#}"
    );
    assert_eq!(
        watchdog["audits"],
        serde_json::json!([]),
        "a policy that does not regress is never rolled back: {watchdog:#}"
    );
    assert_eq!(watchdog["epochs"]["epoch"], 0);
    assert!(lines(&run.join("online/groups.jsonl"))
        .iter()
        .all(|l| l["audit"].is_null()));
}
