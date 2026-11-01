// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data for multi-role agent systems,
// where each model role learns only from the conversations it answered, for
// its clients. If your team needs expertise in fine-tuning the roles of an
// agent harness, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a benchmark trial that ran a harness is one experience per
//! conversation, each with the trial's task, split and verdict and the role
//! that answered it; the same checks, register and leakage protection apply
//! to each. A dataset can be built from the conversations of some roles, and
//! a conversation whose model was shown images is left out and counted.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::bench::{context, sft_request, NO_OPTIONS};
use serde_json::{json, Value};
use splinter_data::Exclusion;
use splinter_orchestrator::Context;
use splinter_pipelines::bench::artifact::sha256_hex;
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::datasets::build;

/// One conversation of a harness trial.
struct Conv {
    name: &'static str,
    /// The role the export names (`extra.harness.role`), if it does.
    exported_role: Option<&'static str>,
    /// The model role the request log names for its requests.
    logged_role: &'static str,
    /// Whether its requests carry an image part.
    images: bool,
}

impl Conv {
    fn new(name: &'static str, role: &'static str) -> Self {
        Self {
            name,
            exported_role: Some(role),
            logged_role: role,
            images: false,
        }
    }

    fn shown_images(self) -> Self {
        Self {
            images: true,
            ..self
        }
    }

    fn role_not_exported(self) -> Self {
        Self {
            exported_role: None,
            ..self
        }
    }
}

/// A conversation: system prompt, instruction, one tool call, final answer.
fn conversation(c: &Conv, trial: &str) -> String {
    let mut harness = json!({"conversation": c.name});
    if let Some(role) = c.exported_role {
        harness["role"] = json!(role);
    }
    json!({
        "schema_version": "ATIF-v1.7",
        "trajectory_id": format!("run/{}", c.name),
        "agent": {"name": "sven", "version": "4.0.0", "model_name": "main",
                  "tool_definitions": [{"type": "function", "function": {
                      "name": "shell", "description": "Run a command.",
                      "parameters": {"type": "object"}}}]},
        "extra": {"harness": harness},
        "steps": [
            {"step_id": 1, "source": "system", "message": format!("You are the {} of {trial}.", c.name)},
            {"step_id": 2, "source": "user", "message": format!("Phase {} of {trial}.", c.name)},
            {"step_id": 3, "source": "agent", "model_name": "m", "message": "\n",
             "tool_calls": [{"tool_call_id": "c0", "function_name": "shell",
                             "arguments": {"command": "ls"}}],
             "observation": {"results": [{"source_call_id": "c0", "content": "part.py"}]}},
            {"step_id": 4, "source": "agent", "model_name": "m", "message": "Done."}
        ]
    })
    .to_string()
}

/// The request log: two requests per conversation (one per agent step), each
/// naming the model role it went to.
fn conversation_requests(convs: &[Conv]) -> String {
    let mut log = String::new();
    let mut seq = 0;
    for c in convs {
        for _ in 0..2 {
            seq += 1;
            let user = if c.images {
                json!([{"type": "text", "text": "look"},
                       {"type": "image", "source": {"data": "AAAA"}}])
            } else {
                json!("go")
            };
            let request = json!({"kind": "request", "seq": seq, "model": c.logged_role,
                "request": {"messages": [{"role": "system", "content": "s"},
                                         {"role": "user", "content": user}]}});
            log.push_str(&format!(
                "{request}\n{{\"kind\": \"response\", \"seq\": {seq}}}\n"
            ));
        }
    }
    log
}

/// Writes an export of one harness trial; returns it. Files are listed in the
/// manifest as the bench lists them (`conversations/NNNN-<name>.json`).
fn harness_export(
    dir: &Path,
    name: &str,
    task: &str,
    set: &str,
    pass: bool,
    convs: &[Conv],
) -> PathBuf {
    let trial_dir = dir.join(name);
    fs::create_dir_all(trial_dir.join("conversations")).unwrap();
    let top = json!({
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "sven", "version": "4.0.0", "model_name": "main"},
        "extra": {"harness": {"conversations": convs.iter().map(|c| c.name).collect::<Vec<_>>()}},
        "steps": []
    })
    .to_string();
    let mut files: Vec<(String, String)> = vec![
        ("trajectory.atif.json".into(), top),
        ("model-requests.jsonl".into(), conversation_requests(convs)),
        ("verification.json".into(), "{\"ok\": true}".into()),
        ("submission.patch".into(), String::new()),
    ];
    for (n, c) in convs.iter().enumerate() {
        files.push((
            format!("conversations/{n:04}-{}.json", c.name),
            conversation(c, name),
        ));
    }
    for (file, text) in &files {
        fs::write(trial_dir.join(file), text).unwrap();
    }
    let verdict = if pass { "pass" } else { "fail" };
    let reward = if pass { 1.0 } else { 0.0 };
    let manifest = json!({
        "schema": "bench.trial-artifact/1", "trial": name,
        "task": {"id": task, "version": "1", "digest": "d1", "dimension": "cad.x", "set": set},
        "snapshot": "abc", "image": {"image": "workstation"},
        "agent": {"name": "sven", "version": "4.0.0"},
        "models": {"main": {"model": "base-27b"}, "specifier": {"model": "spec-model"},
                   "designer": {"model": "design-model"}, "visual": {"model": "vis-model"}},
        "verifier": {"digest": "v1", "scoring": "functional"},
        "grade": {"reward": reward, "verdict": verdict, "metrics": {"valid_solid": reward}},
        "files": files.iter().map(|(n, t)| (n.clone(), json!(sha256_hex(t.as_bytes()))))
            .collect::<serde_json::Map<String, Value>>(),
    });
    let text = serde_json::to_string_pretty(&manifest).unwrap();
    fs::write(trial_dir.join("manifest.json"), &text).unwrap();
    let row = json!({"trial": name, "task": task, "reward": reward, "verdict": verdict,
                     "set": set, "manifest_sha256": sha256_hex(text.as_bytes()), "path": name});
    fs::write(dir.join("index.jsonl"), format!("{row}\n")).unwrap();
    dir.to_path_buf()
}

fn trio() -> Vec<Conv> {
    vec![
        Conv::new("survey", "visual").shown_images(),
        Conv::new("specify", "specifier"),
        Conv::new("design", "designer"),
    ]
}

fn harness_of(ctx: &Context, id: &splinter_core::experience::ExperienceId) -> Value {
    let trajectory = ctx.experiences().get(id).unwrap().trajectory;
    trajectory.extra.unwrap()["splinter_harness"].clone()
}

#[test]
fn a_harness_trial_is_one_experience_per_conversation_under_the_trials_task_split_and_verdict() {
    let (scratch, ctx) = context("harness-admit");
    let dir = harness_export(
        &scratch.0.join("export"),
        "h1",
        "org/cad",
        "train",
        true,
        &trio(),
    );
    let done = ingest_bench_trials(&ctx, std::slice::from_ref(&dir), NO_OPTIONS).unwrap();
    assert!(done.refused.is_empty(), "{:?}", done.refused);
    assert_eq!(done.admitted.get("train"), Some(&1), "one trial");
    assert_eq!(done.verdicts.get("pass"), Some(&1));
    assert_eq!(done.roles.get("specifier"), Some(&1));
    assert_eq!(done.roles.get("designer"), Some(&1));
    let members = ctx
        .experiences()
        .get_set(&done.train_set.clone().unwrap())
        .unwrap()
        .members;
    assert_eq!(members.len(), 3);
    for id in &members {
        let experience = ctx.experiences().get(id).unwrap();
        assert_eq!(experience.environment.spec["task"]["id"], "org/cad");
        assert_eq!(experience.environment.spec["set"], "train");
        assert_eq!(
            experience.trajectory.steps[0].source,
            atif::StepOrigin::System
        );
        assert!(experience
            .trajectory
            .extra
            .as_ref()
            .unwrap()
            .get("splinter_enrichment")
            .is_none());
    }
    let marks: Vec<Value> = members.iter().map(|id| harness_of(&ctx, id)).collect();
    let find = |name: &str| marks.iter().find(|m| m["conversation"] == name).unwrap();
    assert_eq!(find("specify")["role"], "specifier");
    assert_eq!(find("specify")["role_source"], "export");
    assert_eq!(find("specify")["shown_images"], false);
    assert_eq!(find("survey")["shown_images"], true);
    let model = ctx
        .experiences()
        .get(&members[0])
        .unwrap()
        .provenance
        .solver;
    assert!(model.starts_with("bench/sven@4.0.0/"), "{model}");

    // Again: nothing changes, nothing is made twice.
    let again = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    assert_eq!(again.admitted.values().sum::<usize>(), 0);
    assert_eq!(again.already_held, 1);
    assert_eq!(again.train_set, done.train_set);
    assert_eq!(ctx.experiences().list().unwrap().len(), 3);
}

#[test]
fn an_export_without_roles_gets_them_from_the_request_log_only_when_it_attributes_them() {
    let (scratch, ctx) = context("harness-roles");
    let convs: Vec<Conv> = trio().into_iter().map(Conv::role_not_exported).collect();
    let dir = harness_export(
        &scratch.0.join("a"),
        "old",
        "org/cad",
        "train",
        true,
        &convs,
    );
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    let members = ctx
        .experiences()
        .get_set(&done.train_set.unwrap())
        .unwrap()
        .members;
    let sources: Vec<(String, String)> = members
        .iter()
        .map(|id| {
            let m = harness_of(&ctx, id);
            (m["conversation"].to_string(), m["role_source"].to_string())
        })
        .collect();
    assert!(
        sources.iter().all(|s| s.1 == "\"request_log\""),
        "{sources:?}"
    );
    assert_eq!(done.roles.get("designer"), Some(&1));

    // A log that does not divide among the conversations (a retry made an
    // extra request) attributes nothing: the roles are unknown, not guessed,
    // and since any conversation may have seen the images, all are marked.
    let mut skewed = trio()
        .into_iter()
        .map(Conv::role_not_exported)
        .collect::<Vec<_>>();
    let dir = harness_export(
        &scratch.0.join("b"),
        "skewed",
        "org/cad2",
        "train",
        true,
        &skewed,
    );
    let log = dir.join("skewed/model-requests.jsonl");
    let extra = format!("{}{{\"kind\": \"request\", \"seq\": 99, \"model\": \"visual\", \"request\": {{\"messages\": []}}}}\n", fs::read_to_string(&log).unwrap());
    fs::write(&log, &extra).unwrap();
    // Re-seal the manifest and index over the edited log.
    let manifest_path = dir.join("skewed/manifest.json");
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["files"]["model-requests.jsonl"] = json!(sha256_hex(extra.as_bytes()));
    let text = serde_json::to_string_pretty(&manifest).unwrap();
    fs::write(&manifest_path, &text).unwrap();
    let row = json!({"trial": "skewed", "task": "org/cad2", "reward": 1.0, "verdict": "pass",
                     "set": "train", "manifest_sha256": sha256_hex(text.as_bytes()), "path": "skewed"});
    fs::write(dir.join("index.jsonl"), format!("{row}\n")).unwrap();
    skewed.clear();
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    assert!(done.refused.is_empty(), "{:?}", done.refused);
    assert_eq!(done.roles.get("unknown"), Some(&3));
    let members = ctx
        .experiences()
        .get_set(&done.train_set.unwrap())
        .unwrap()
        .members;
    for id in &members {
        let m = harness_of(&ctx, id);
        assert_eq!(m["role"], Value::Null);
        assert_eq!(m["role_source"], "unknown");
        assert_eq!(m["shown_images"], true);
    }
}

#[test]
fn a_conversation_file_that_is_altered_unlisted_or_missing_refuses_the_trial() {
    let (scratch, ctx) = context("harness-tamper");
    let dir = harness_export(
        &scratch.0.join("export"),
        "h",
        "org/cad",
        "train",
        true,
        &trio(),
    );
    let conv = |n: &str| dir.join("h/conversations").join(n);
    fs::write(conv("0001-specify.json"), "{}").unwrap();
    let done = ingest_bench_trials(&ctx, std::slice::from_ref(&dir), NO_OPTIONS).unwrap();
    assert!(
        done.refused[0]
            .reason
            .contains("conversations/0001-specify.json is altered"),
        "{done:?}"
    );
    assert!(ctx.experiences().list().unwrap().is_empty());

    let dir = harness_export(
        &scratch.0.join("two"),
        "h",
        "org/cad",
        "train",
        true,
        &trio(),
    );
    fs::write(dir.join("h/conversations/0009-extra.json"), "{}").unwrap();
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    assert!(
        done.refused[0]
            .reason
            .contains("conversations/0009-extra.json is in the directory but not in the manifest"),
        "{done:?}"
    );

    let dir = harness_export(
        &scratch.0.join("three"),
        "h",
        "org/cad",
        "train",
        true,
        &trio(),
    );
    fs::remove_file(dir.join("h/conversations/0002-design.json")).unwrap();
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    assert!(
        done.refused[0].reason.contains("0002-design.json"),
        "{done:?}"
    );
    assert!(ctx.experiences().list().unwrap().is_empty());
}

#[test]
fn the_conversations_of_an_eval_trial_never_enter_a_dataset_and_the_task_stays_on_its_side() {
    let (scratch, ctx) = context("harness-eval");
    let dir = harness_export(&scratch.0.join("e"), "he", "org/cad", "eval", true, &trio());
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    let eval = done.eval_set.unwrap();
    let refused = build(&ctx, &sft_request(eval)).unwrap_err().to_string();
    assert!(
        refused.contains("leakage") && refused.contains("org/cad"),
        "{refused}"
    );
    let train = harness_export(
        &scratch.0.join("t"),
        "ht",
        "org/cad",
        "train",
        true,
        &trio(),
    );
    let clash = ingest_bench_trials(&ctx, &[train], NO_OPTIONS).unwrap();
    assert!(
        clash.refused[0].reason.contains("cannot also be train"),
        "{clash:?}"
    );
}

#[test]
fn a_dataset_can_be_built_from_the_conversations_of_chosen_roles_and_image_conversations_are_counted_out(
) {
    let (scratch, ctx) = context("harness-dataset");
    let dir = harness_export(
        &scratch.0.join("export"),
        "h",
        "org/cad",
        "train",
        true,
        &trio(),
    );
    let set = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS)
        .unwrap()
        .train_set
        .unwrap();

    // Every role: the two text conversations give two records each; the
    // survey, whose model was shown images the trajectory does not carry, is
    // left out and counted.
    let all = build(&ctx, &sft_request(set.clone())).unwrap();
    assert_eq!(all.records, 4);
    assert_eq!(all.excluded.get(&Exclusion::ShownImages), Some(&1));

    let mut request = sft_request(set.clone());
    request.roles = Some(vec!["specifier".into()]);
    let only = build(&ctx, &request).unwrap();
    assert_eq!(only.records, 2);
    assert_eq!(only.excluded.get(&Exclusion::RoleNotWanted), Some(&2));
    let text = fs::read_to_string(&only.path).unwrap();
    assert!(
        text.contains("You are the specify of h.") && !text.contains("the design of h"),
        "{text}"
    );

    // Asking for the image role gets nothing: the reason is counted, never
    // silently dropped.
    request.roles = Some(vec!["visual".into()]);
    let visual = build(&ctx, &request).unwrap_err().to_string();
    assert!(visual.contains("ShownImages: 1"), "{visual}");
}

#[test]
fn a_role_filter_never_selects_a_conversation_whose_role_is_unknown() {
    let (scratch, ctx) = context("harness-unknown-role");
    let convs = vec![
        Conv::new("specify", "specifier").role_not_exported(),
        Conv::new("odd", "designer").role_not_exported(),
    ];
    // One request too few for two conversations: nothing is attributed.
    let dir = harness_export(
        &scratch.0.join("export"),
        "h",
        "org/cad",
        "train",
        true,
        &convs,
    );
    let log = dir.join("h/model-requests.jsonl");
    let kept: String = fs::read_to_string(&log)
        .unwrap()
        .lines()
        .take(6)
        .map(|l| format!("{l}\n"))
        .collect();
    fs::write(&log, &kept).unwrap();
    let manifest_path = dir.join("h/manifest.json");
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["files"]["model-requests.jsonl"] = json!(sha256_hex(kept.as_bytes()));
    let text = serde_json::to_string_pretty(&manifest).unwrap();
    fs::write(&manifest_path, &text).unwrap();
    let row = json!({"trial": "h", "task": "org/cad", "reward": 1.0, "verdict": "pass",
                     "set": "train", "manifest_sha256": sha256_hex(text.as_bytes()), "path": "h"});
    fs::write(dir.join("index.jsonl"), format!("{row}\n")).unwrap();
    let set = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS)
        .unwrap()
        .train_set
        .unwrap();
    let mut request = sft_request(set);
    request.roles = Some(vec!["specifier".into()]);
    let refused = build(&ctx, &request).unwrap_err().to_string();
    assert!(refused.contains("RoleNotWanted: 2"), "{refused}");
}
