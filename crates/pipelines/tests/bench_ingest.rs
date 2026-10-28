// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from agent benchmark runs to
// fine-tuning data that is verified and free of evaluation leakage, for its
// clients. If your team needs expertise in learning from benchmark
// trajectories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the trials a benchmark exports become experiences, and the split of
//! the benchmark is never crossed.
//!
//! * A trial is admitted only when every file matches the digest its manifest
//!   names and the manifest is the one the index names; anything else is
//!   refused with the reason and the rest of the export is still admitted.
//! * The verdict is the benchmark's grade, the record its ATIF trajectory.
//! * Ingesting an export again changes nothing; a trial name with other
//!   content is refused.
//! * A trial whose export names no split is not admissible unless the operator
//!   says which; a task is never both train and eval.
//! * An experience of an `eval` task cannot enter a dataset by any route: not
//!   by building from its set, not by mixing it with train experience, and
//!   not by a task registered as eval whose experience claims train.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::{config, Scratch};
use serde_json::{json, Value};
use splinter_core::annotation::Strength;
use splinter_core::clock::FixedClock;
use splinter_data::Exclusion;
use splinter_orchestrator::Context;
use splinter_pipelines::bench::artifact::sha256_hex;
use splinter_pipelines::bench::split::Split;
use splinter_pipelines::bench::{ingest_bench_trials, IngestOptions};
use splinter_pipelines::datasets::{
    build, build_dropping_leaks, BuildRequest, ViewName, VoiceBuild,
};
use splinter_store::experiences::{ExperienceSet, SetId};

fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-28T09:00:00.000Z")));
    (scratch, ctx)
}

/// One synthetic trial: a task, the split the export states for it (if any)
/// and whether the benchmark passed it.
struct Trial {
    name: &'static str,
    task: &'static str,
    set: Option<&'static str>,
    pass: bool,
}

impl Trial {
    fn new(name: &'static str, task: &'static str, set: Option<&'static str>, pass: bool) -> Self {
        Self {
            name,
            task,
            set,
            pass,
        }
    }
}

/// An ATIF trajectory of one tool call answered by its result, then a final
/// answer; `note` distinguishes the trials.
fn trajectory(note: &str) -> String {
    json!({
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "sven", "version": "4.0.0", "model_name": "openai/main"},
        "extra": {"sven": {"created_at": "2026-10-27T18:16:21.405627709Z"}},
        "steps": [
            {"step_id": 1, "source": "user", "message": format!("Repair the firmware ({note}).")},
            {"step_id": 2, "source": "agent", "model_name": "m", "message": "\n",
             "reasoning_content": "look first",
             "tool_calls": [{"tool_call_id": "c0", "function_name": "shell",
                             "arguments": {"command": "ls"}}],
             "observation": {"results": [{"source_call_id": "c0", "content": "main.c"}]}},
            {"step_id": 3, "source": "agent", "model_name": "m", "message": "Fixed."}
        ]
    })
    .to_string()
}

/// The requests the agent sent: the first holds the system prompt and the
/// tool schemas the model was given; the commit in the prompt is the trial's.
fn requests(trial: &str) -> String {
    let request = json!({"kind": "request", "seq": 1, "request": {
        "messages": [
            {"role": "system", "content": format!("You are Sven.\nCommit: {trial}")},
            {"role": "user", "content": "Repair the firmware."}],
        "tools": [{"name": "shell", "description": "Run a command.",
                   "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}],
    }});
    format!("{request}\n{{\"kind\": \"response\", \"seq\": 1}}\n")
}

/// Writes the export `dir` holding `trials`; returns it.
fn export(dir: &Path, trials: &[Trial]) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let mut index = String::new();
    for t in trials {
        let trial_dir = dir.join(t.name);
        fs::create_dir_all(&trial_dir).unwrap();
        let files = [
            ("trajectory.atif.json", trajectory(t.name)),
            ("model-requests.jsonl", requests(t.name)),
            ("verification.json", "{\"ok\": true}".to_string()),
            ("submission.patch", "diff --git a/x b/x\n".to_string()),
        ];
        for (name, text) in &files {
            fs::write(trial_dir.join(name), text).unwrap();
        }
        let reward = if t.pass { 1.0 } else { 0.0 };
        let verdict = if t.pass { "pass" } else { "fail" };
        let mut task = json!({"id": t.task, "version": "2", "digest": "d1",
                              "dimension": "firmware.x", "path": "tasks/x"});
        if let Some(set) = t.set {
            task["set"] = json!(set);
        }
        let manifest = json!({
            "schema": "bench.trial-artifact/1",
            "trial": t.name,
            "task": task,
            "snapshot": "abc123",
            "image": {"image": "workstation", "environment": "userns"},
            "agent": {"name": "sven", "version": "4.0.0", "sven_build_sha256": "b1"},
            "models": {"main": {"model": "qwen-27b", "base_url": "http://x"}},
            "verifier": {"digest": "v1", "scoring": "all"},
            "grade": {"reward": reward, "verdict": verdict, "metrics": {"build_success": 1.0}},
            "tokens": {"prompt": 10, "completion": 5},
            "wall_seconds": 1.5,
            "files": files.iter()
                .map(|(n, text)| (n.to_string(), json!(sha256_hex(text.as_bytes()))))
                .collect::<serde_json::Map<String, Value>>(),
        });
        let manifest_text = serde_json::to_string_pretty(&manifest).unwrap();
        fs::write(trial_dir.join("manifest.json"), &manifest_text).unwrap();
        let mut row = json!({"trial": t.name, "task": t.task, "reward": reward,
                             "verdict": verdict, "manifest_sha256": sha256_hex(manifest_text.as_bytes()),
                             "path": t.name});
        if let Some(set) = t.set {
            row["set"] = json!(set);
        }
        index.push_str(&format!("{row}\n"));
    }
    fs::write(dir.join("index.jsonl"), index).unwrap();
    dir.to_path_buf()
}

fn sft_request(set: SetId) -> BuildRequest {
    BuildRequest {
        sets: vec![set],
        view: ViewName::SftStep,
        strip: None,
        min_strength: Some(Strength::Executable),
        system_prompt: None,
        export_only: false,
        limit: None,
        voice: VoiceBuild::default(),
        max_block: None,
        pack: false,
        split_over_block: false,
        control: Default::default(),
        serialise_calls: false,
        only_actions: None,
    }
}

fn mixed(ctx: &Context, a: &SetId, b: &SetId) -> SetId {
    let store = ctx.experiences();
    let mut members = store.get_set(a).unwrap().members;
    members.extend(store.get_set(b).unwrap().members);
    store
        .put_set(&ExperienceSet {
            name: "mixed".into(),
            members,
        })
        .unwrap()
}

const NO_OPTIONS: IngestOptions = IngestOptions { assume_set: None };

#[test]
fn trials_become_experiences_with_the_benchmarks_verdict_in_one_set_per_split() {
    let (scratch, ctx) = context("bench-admit");
    let dir = export(
        &scratch.0.join("export"),
        &[
            Trial::new("t-pass", "org/a", Some("train"), true),
            Trial::new("t-fail", "org/a", Some("train"), false),
            Trial::new("e-pass", "org/b", Some("eval"), true),
        ],
    );
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    assert!(done.refused.is_empty(), "{:?}", done.refused);
    assert_eq!(done.admitted.get("train"), Some(&2));
    assert_eq!(done.admitted.get("eval"), Some(&1));
    assert_eq!(done.verdicts.get("pass"), Some(&2));
    assert_eq!(done.verdicts.get("fail"), Some(&1));
    let store = ctx.experiences();
    let train = store.get_set(&done.train_set.unwrap()).unwrap();
    let eval = store.get_set(&done.eval_set.unwrap()).unwrap();
    assert_eq!((train.members.len(), eval.members.len()), (2, 1));
    let experience = store.get(&eval.members[0]).unwrap();
    assert_eq!(
        experience.trajectory.steps.len(),
        4,
        "the system prompt leads the three exported steps"
    );
    assert_eq!(experience.environment.spec["set"], "eval");
    assert_eq!(experience.environment.spec["task"]["version"], "2");
    assert_eq!(experience.environment.spec["verifier"]["digest"], "v1");
    assert_eq!(experience.provenance.solver, "bench/sven@4.0.0/qwen-27b");
}

#[test]
fn ingesting_an_export_again_changes_nothing_and_other_content_under_a_name_is_refused() {
    let (scratch, ctx) = context("bench-idempotent");
    let trials = [
        Trial::new("t1", "org/a", Some("train"), true),
        Trial::new("t2", "org/a", Some("train"), false),
    ];
    let dir = export(&scratch.0.join("export"), &trials);
    let first = ingest_bench_trials(&ctx, std::slice::from_ref(&dir), NO_OPTIONS).unwrap();
    let again = ingest_bench_trials(&ctx, std::slice::from_ref(&dir), NO_OPTIONS).unwrap();
    assert_eq!(again.admitted.values().sum::<usize>(), 0);
    assert_eq!(again.already_held, 2);
    assert_eq!(first.train_set, again.train_set);
    assert_eq!(ctx.experiences().list().unwrap().len(), 2);

    // The same trial name, graded otherwise, is another artifact: refused.
    let other = export(
        &scratch.0.join("other"),
        &[Trial::new("t1", "org/a", Some("train"), false)],
    );
    let clash = ingest_bench_trials(&ctx, &[other], NO_OPTIONS).unwrap();
    assert_eq!(clash.refused.len(), 1);
    assert!(
        clash.refused[0].reason.contains("already admitted"),
        "{:?}",
        clash.refused
    );
    assert_eq!(ctx.experiences().list().unwrap().len(), 2);
}

#[test]
fn a_tampered_or_partial_trial_is_refused_with_the_reason_and_the_rest_is_admitted() {
    let (scratch, ctx) = context("bench-tamper");
    let names = ["ok", "patched", "manifest", "extra", "missing"];
    let trials: Vec<Trial> = names
        .iter()
        .map(|n| Trial::new(n, "org/a", Some("train"), true))
        .collect();
    let dir = export(&scratch.0.join("export"), &trials);
    let touch =
        |trial: &str, file: &str, text: &str| fs::write(dir.join(trial).join(file), text).unwrap();
    touch("patched", "submission.patch", "diff --git a/y b/y\n");
    let manifest = fs::read_to_string(dir.join("manifest/manifest.json")).unwrap();
    touch(
        "manifest",
        "manifest.json",
        &manifest.replace("\"pass\"", "\"fail\""),
    );
    touch("extra", "notes.txt", "not in the manifest");
    fs::remove_file(dir.join("missing/verification.json")).unwrap();

    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    assert_eq!(done.admitted.get("train"), Some(&1), "{:?}", done.refused);
    let why = |trial: &str| {
        done.refused
            .iter()
            .find(|r| r.trial.as_deref() == Some(trial))
            .unwrap_or_else(|| panic!("{trial} was not refused"))
            .reason
            .clone()
    };
    assert!(why("patched").contains("submission.patch is altered"));
    assert!(why("manifest").contains("manifest.json is altered"));
    assert!(why("extra").contains("notes.txt is in the directory but not in the manifest"));
    assert!(why("missing").contains("verification.json"));
    assert_eq!(ctx.experiences().list().unwrap().len(), 1);
}

#[test]
fn a_trial_with_no_stated_split_is_admissible_only_when_the_operator_says_which() {
    let (scratch, ctx) = context("bench-assume");
    let dir = export(
        &scratch.0.join("export"),
        &[Trial::new("old", "org/old", None, true)],
    );
    let refused = ingest_bench_trials(&ctx, std::slice::from_ref(&dir), NO_OPTIONS).unwrap();
    assert!(
        refused.refused[0].reason.contains("--assume-set"),
        "{refused:?}"
    );
    assert!(ctx.experiences().list().unwrap().is_empty());

    let options = IngestOptions {
        assume_set: Some(Split::Train),
    };
    let done = ingest_bench_trials(&ctx, &[dir], options).unwrap();
    assert_eq!(done.admitted.get("train"), Some(&1));

    // An operator cannot overrule an export that states its split.
    let stated = export(
        &scratch.0.join("stated"),
        &[Trial::new("new", "org/new", Some("eval"), true)],
    );
    let clash = ingest_bench_trials(&ctx, &[stated], options).unwrap();
    assert!(
        clash.refused[0].reason.contains("operator said train"),
        "{clash:?}"
    );
}

#[test]
fn a_task_is_never_train_in_one_export_and_eval_in_another() {
    let (scratch, ctx) = context("bench-conflict");
    let evaluated = export(
        &scratch.0.join("a"),
        &[Trial::new("x1", "org/x", Some("eval"), true)],
    );
    ingest_bench_trials(&ctx, &[evaluated], NO_OPTIONS).unwrap();
    let relabelled = export(
        &scratch.0.join("b"),
        &[Trial::new("x2", "org/x", None, true)],
    );
    let options = IngestOptions {
        assume_set: Some(Split::Train),
    };
    let done = ingest_bench_trials(&ctx, &[relabelled], options).unwrap();
    assert!(
        done.refused[0].reason.contains("cannot also be train"),
        "{done:?}"
    );
}

#[test]
fn only_passing_train_trials_become_tool_use_records_and_eval_never_enters_a_dataset() {
    let (scratch, ctx) = context("bench-dataset");
    let dir = export(
        &scratch.0.join("export"),
        &[
            Trial::new("t-pass", "org/a", Some("train"), true),
            Trial::new("t-fail", "org/a", Some("train"), false),
            Trial::new("e-pass", "org/b", Some("eval"), true),
        ],
    );
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    let (train, eval) = (done.train_set.unwrap(), done.eval_set.unwrap());

    // The view for multi-turn tool use yields one record per action of the
    // passing trial: the tool call (with its result as context) and the answer.
    let built = build(&ctx, &sft_request(train.clone())).unwrap();
    assert_eq!(built.records, 2);
    assert_eq!(built.excluded.get(&Exclusion::Failed), Some(&1));
    let text = fs::read_to_string(&built.path).unwrap();
    assert!(text.contains("\"tool_calls\""), "{text}");
    assert!(
        text.contains("main.c"),
        "the tool result is context: {text}"
    );
    assert!(
        text.contains("\"reasoning_content\":\"look first\""),
        "the step's reasoning rides on its turn: {text}"
    );
    assert!(
        text.contains("You are Sven.\\nCommit: t-pass"),
        "the trial's own system turn: {text}"
    );
    assert!(
        text.contains("\"tools\"") && text.contains("Run a command."),
        "{text}"
    );

    // The eval set cannot be built from, alone or mixed with train.
    let refused = build(&ctx, &sft_request(eval.clone()))
        .unwrap_err()
        .to_string();
    assert!(
        refused.contains("leakage") && refused.contains("org/b"),
        "{refused}"
    );
    let both = mixed(&ctx, &train, &eval);
    assert!(build(&ctx, &sft_request(both.clone())).is_err());

    // Dropping leaks leaves the eval trial out, lists it, and builds the rest.
    let dropped = build_dropping_leaks(&ctx, &sft_request(both)).unwrap();
    assert_eq!(dropped.dropped.len(), 1);
    assert_eq!(dropped.records, built.records);
    assert!(!fs::read_to_string(&dropped.path).unwrap().contains("org/b"));
}

#[test]
fn the_record_opens_with_the_trials_own_system_prompt_and_tools_and_the_derivation_is_recorded() {
    let (scratch, ctx) = context("bench-context");
    let dir = export(
        &scratch.0.join("export"),
        &[
            Trial::new("t-one", "org/a", Some("train"), true),
            Trial::new("t-two", "org/b", Some("train"), true),
        ],
    );
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    let set = done.train_set.unwrap();
    let store = ctx.experiences();

    // The trajectory is the exported one plus the context, and says so.
    let id = store.get_set(&set).unwrap().members[0].clone();
    let trajectory = store.get(&id).unwrap().trajectory;
    assert_eq!(trajectory.steps[0].source, atif::StepOrigin::System);
    assert_eq!(trajectory.steps.len(), 4);
    assert_eq!(trajectory.steps[1].step_id, 2, "steps stay sequential");
    assert_eq!(trajectory.agent.tool_definitions.as_ref().unwrap().len(), 1);
    let enrichment = &trajectory.extra.as_ref().unwrap()["splinter_enrichment"];
    assert_eq!(enrichment["source"], "model-requests.jsonl");
    assert_eq!(enrichment["source_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(enrichment["trajectory_sha256"].as_str().unwrap().len(), 64);

    // Two trials served different prompts (the commit) build one dataset,
    // each record under its own prompt, with the tools the agent offered.
    let built = build(&ctx, &sft_request(set)).unwrap();
    let lines: Vec<Value> = fs::read_to_string(&built.path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let opening: std::collections::BTreeSet<&str> = lines
        .iter()
        .map(|l| l["messages"][0]["content"].as_str().unwrap())
        .collect();
    assert_eq!(opening.len(), 2, "{opening:?}");
    assert!(lines.iter().all(|l| l["messages"][0]["role"] == "system"));
    assert!(lines
        .iter()
        .all(|l| l["tools"][0]["function"]["name"] == "shell"));
}

#[test]
fn a_trial_without_the_model_requests_is_refused_because_its_context_is_unknown() {
    let (scratch, ctx) = context("bench-no-requests");
    let dir = export(
        &scratch.0.join("export"),
        &[Trial::new("t1", "org/a", Some("train"), true)],
    );
    // Rewrite the manifest without the requests file, keeping the index honest.
    let trial = dir.join("t1");
    fs::remove_file(trial.join("model-requests.jsonl")).unwrap();
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(trial.join("manifest.json")).unwrap()).unwrap();
    manifest["files"]
        .as_object_mut()
        .unwrap()
        .remove("model-requests.jsonl");
    let text = serde_json::to_string_pretty(&manifest).unwrap();
    fs::write(trial.join("manifest.json"), &text).unwrap();
    let row = json!({"trial": "t1", "task": "org/a", "reward": 1.0, "verdict": "pass",
                     "set": "train", "manifest_sha256": sha256_hex(text.as_bytes()), "path": "t1"});
    fs::write(dir.join("index.jsonl"), format!("{row}\n")).unwrap();
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    assert!(
        done.refused[0].reason.contains("model-requests.jsonl"),
        "{done:?}"
    );
}

#[test]
fn an_experience_of_a_task_registered_as_eval_is_refused_even_if_it_claims_train() {
    let (scratch, ctx) = context("bench-claims-train");
    let evaluated = export(
        &scratch.0.join("a"),
        &[Trial::new("x1", "org/x", Some("eval"), true)],
    );
    ingest_bench_trials(&ctx, &[evaluated], NO_OPTIONS).unwrap();
    // An experience written by another route, naming the same task as train.
    let sneaked = export(
        &scratch.0.join("b"),
        &[Trial::new("x2", "org/x", Some("train"), true)],
    );
    let other = context("bench-claims-train-other").1;
    let admitted = ingest_bench_trials(&other, &[sneaked], NO_OPTIONS).unwrap();
    let experience = admitted.train_set.unwrap();
    let members = other.experiences().get_set(&experience).unwrap().members;
    // Carry it into the context that knows the task is eval.
    let carried = other.experiences().get(&members[0]).unwrap();
    let id = ctx.experiences().put(&carried).unwrap();
    let set = ctx
        .experiences()
        .put_set(&ExperienceSet {
            name: "sneaked".into(),
            members: vec![id],
        })
        .unwrap();
    let refused = build(&ctx, &sft_request(set)).unwrap_err().to_string();
    assert!(refused.contains("registered as an eval task"), "{refused}");
}
