// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from agent benchmark runs to
// fine-tuning data that is verified and free of evaluation leakage, for its
// clients. If your team needs expertise in learning from benchmark
// trajectories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What the benchmark-ingestion tests share: a context with a fixed clock, the
//! sft-step request they build with and the default ingestion options.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use splinter_core::annotation::Strength;
use splinter_core::clock::FixedClock;
use splinter_orchestrator::Context;
use splinter_pipelines::bench::artifact::sha256_hex;
use splinter_pipelines::bench::IngestOptions;
use splinter_pipelines::datasets::{BuildRequest, ViewName, VoiceBuild};
use splinter_store::experiences::SetId;

use super::{config, Scratch};

pub fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-28T09:00:00.000Z")));
    (scratch, ctx)
}

pub fn sft_request(set: SetId) -> BuildRequest {
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
        roles: None,
        images: false,
        rl: Default::default(),
    }
}

pub const NO_OPTIONS: IngestOptions = IngestOptions { assume_set: None };

/// One synthetic trial: a task, the split the export states for it (if any)
/// and whether the benchmark passed it.
pub struct Trial {
    pub name: &'static str,
    pub task: &'static str,
    pub set: Option<&'static str>,
    pub pass: bool,
    /// What the verifier measured, in the grade.
    pub metrics: Value,
}

impl Trial {
    pub fn new(
        name: &'static str,
        task: &'static str,
        set: Option<&'static str>,
        pass: bool,
    ) -> Self {
        Self {
            name,
            task,
            set,
            pass,
            metrics: json!({"build_success": 1.0}),
        }
    }

    /// The same trial with `metrics` as the verifier's measurements.
    pub fn metrics(self, metrics: Value) -> Self {
        Self { metrics, ..self }
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
pub fn export(dir: &Path, trials: &[Trial]) -> PathBuf {
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
            "grade": {"reward": reward, "verdict": verdict, "metrics": t.metrics},
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
