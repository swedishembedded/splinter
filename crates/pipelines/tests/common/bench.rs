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
        practice: Vec::new(),
        split_seed: None,
        history: Default::default(),
        base: None,
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
    /// The opening user turn carries a picture beside its text, as an agent
    /// given an attachment records it.
    pub picture: bool,
    /// The verifier's own statement of why it decided as it did.
    pub reason: Option<&'static str>,
    /// The examiner's score record (`bench.score/1`), when the export has
    /// one.
    pub score: Option<Value>,
    /// Whether the trial ran a released harness package.
    pub confidential: bool,
    /// The capability its task exercises, `<domain>.<task>` as the benchmark
    /// declares it.
    pub dimension: &'static str,
    /// The shell command its one tool call runs.
    pub command: &'static str,
    /// The commit its agent's system prompt and its instruction name; `None`
    /// is the trial's own name (each trial checked out a commit of its own).
    pub commit: Option<&'static str>,
    /// The capability ids the export names for its task, as the examiner
    /// writes them into the index row and the manifest; `None` writes none,
    /// as an export made before the examiner labelled tasks.
    pub capabilities: Option<&'static [&'static str]>,
    /// Each agent step records the policy's own trace (prompt ids, sampled
    /// ids, their log-probabilities), and the request log holds every request
    /// the model was sent, the harness's time line in the user turn, with the
    /// response it got; the manifest declares the log's schema.
    pub traced: bool,
    /// A traced trial's model also answered one request with nothing (no
    /// text, no call: the agent nudged it and asked again) and was sent one
    /// more it never answered (the trial ended first); neither is an agent step.
    pub cut_short: bool,
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
            picture: false,
            reason: None,
            score: None,
            confidential: false,
            dimension: "firmware.x",
            command: "ls",
            commit: None,
            capabilities: None,
            traced: false,
            cut_short: false,
        }
    }

    /// The same traced trial with an empty answer and a last request left
    /// unanswered in its request log.
    pub fn cut_short(self) -> Self {
        Self {
            traced: true,
            cut_short: true,
            ..self
        }
    }

    /// The same trial with each agent step's policy trace and the full
    /// request log beside it.
    pub fn traced(self) -> Self {
        Self {
            traced: true,
            ..self
        }
    }

    /// The same trial of a task the examiner labels with `capabilities`.
    pub fn capabilities(self, capabilities: &'static [&'static str]) -> Self {
        Self {
            capabilities: Some(capabilities),
            ..self
        }
    }

    /// The same trial of a task checked out at `commit`, as the other attempts
    /// of the task are: one prompt for all of them.
    pub fn at_commit(self, commit: &'static str) -> Self {
        Self {
            commit: Some(commit),
            ..self
        }
    }

    /// The same trial with the score record the examiner gives a graded
    /// trial of its verdict.
    pub fn scored(self) -> Self {
        let reward = if self.pass { 1.0 } else { 0.0 };
        let score = json!({
            "schema": "bench.score/1", "status": "graded", "counted": true,
            "score": reward, "pass": self.pass, "threshold": 1.0, "cause": null,
            "metrics": self.metrics, "vetoes": [],
            "usage": {"requests": 2, "prompt_tokens": 10, "completion_tokens": 5},
            "identity": {"task": self.task, "version": "2", "digest": "d1", "attempt": 1,
                         "models": {"main": "qwen-27b"}},
        });
        self.score(score)
    }

    /// The same trial with `score` as its score record.
    pub fn score(self, score: Value) -> Self {
        Self {
            score: Some(score),
            ..self
        }
    }

    /// The same trial run with a released harness package.
    pub fn confidential(self) -> Self {
        Self {
            confidential: true,
            ..self
        }
    }

    /// The same trial with the verifier's `reason` in its `verification.json`.
    pub fn reason(self, reason: &'static str) -> Self {
        Self {
            reason: Some(reason),
            ..self
        }
    }

    /// The same trial with a picture attached to its opening turn.
    pub fn with_picture(self) -> Self {
        Self {
            picture: true,
            ..self
        }
    }

    /// The same trial of a task that declares `dimension`.
    pub fn dimension(self, dimension: &'static str) -> Self {
        Self { dimension, ..self }
    }

    /// The same trial with its one tool call running `command`.
    pub fn command(self, command: &'static str) -> Self {
        Self { command, ..self }
    }

    /// The same trial with `metrics` as the verifier's measurements.
    pub fn metrics(self, metrics: Value) -> Self {
        Self { metrics, ..self }
    }
}

/// An ATIF trajectory of one tool call answered by its result, then a final
/// answer; `note` distinguishes the trials.
fn trajectory(note: &str, picture: bool, command: &str, traced: bool) -> String {
    let instruction = format!("Repair the firmware ({note}).");
    let opening = if picture {
        json!([{"type": "text", "text": instruction},
               {"type": "image", "source": {"media_type": "image/png", "path": picture_path()}}])
    } else {
        json!(instruction)
    };
    let mut trajectory = json!({
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "sven", "version": "4.0.0", "model_name": "openai/main"},
        "extra": {"sven": {"created_at": "2026-10-27T18:16:21.405627709Z"}},
        "steps": [
            {"step_id": 1, "source": "user", "message": opening},
            {"step_id": 2, "source": "agent", "model_name": "m", "message": "\n",
             "reasoning_content": "look first",
             "tool_calls": [{"tool_call_id": "c0", "function_name": "shell",
                             "arguments": {"command": command}}],
             "observation": {"results": [{"source_call_id": "c0", "content": "main.c"}]}},
            {"step_id": 3, "source": "agent", "model_name": "m", "message": "Fixed."}
        ]
    });
    if traced {
        for (step, metrics) in [
            (1, trace(&[1, 2, 3], &[10, 11])),
            (2, trace(&[1, 2, 3, 10, 11, 4], &[12, 13])),
        ] {
            trajectory["steps"][step]["metrics"] = metrics;
        }
    }
    trajectory.to_string()
}

/// What sven records on an agent step the policy served: the ids it was
/// given, the ids it sampled and one log-probability of the policy per
/// sampled id.
fn trace(prompt: &[i64], completion: &[i64]) -> Value {
    json!({"prompt_token_ids": prompt, "completion_token_ids": completion,
           "logprobs": completion.iter().map(|_| -0.5).collect::<Vec<_>>(),
           "extra": {"sven": {"logprob_kind": "policy", "tokenizer_digest": "sha256:aa"}}})
}

/// The instruction of a traced trial as the harness sent it: the task's,
/// followed by the time line the trajectory does not record.
pub const TIME_LINE: &str = "Time: 0 of 18 min used, 18 min left.";

/// The request log of a traced trial (sven's `--record-requests`): one
/// request per agent step, each the whole conversation so far, and the
/// response it got.
fn requests_as_sent(note: &str, command: &str, cut_short: bool) -> String {
    let shell = json!({"name": "shell", "description": "Run a command.",
                       "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}});
    let opening = vec![
        json!({"role": "system", "content": format!("You are Sven.\nCommit: {note}")}),
        json!({"role": "user", "content": format!("Repair the firmware ({note}).\n\n{TIME_LINE}")}),
    ];
    let arguments = json!({"command": command}).to_string();
    let mut later = opening.clone();
    later.extend([
        json!({"role": "assistant", "content": "", "reasoning": "look first"}),
        json!({"role": "assistant", "content": {"tool_call_id": "c0", "function": {"name": "shell", "arguments": arguments}}}),
        json!({"role": "tool", "content": {"tool_call_id": "c0", "content": "main.c"}}),
    ]);
    let request = |seq: u64, messages: &[Value]| {
        json!({"kind": "request", "seq": seq, "provider": "brain", "model": "main",
               "request": {"messages": messages, "tools": [shell.clone()]}})
    };
    let mut lines = vec![
        request(1, &opening),
        json!({"kind": "response", "seq": 1, "response": {"text": "", "reasoning": "look first",
               "tool_calls": [{"index": 0, "id": "c0", "name": "shell", "arguments": arguments}]}}),
    ];
    let mut seq = 2;
    if cut_short {
        lines.push(request(seq, &later));
        lines.push(json!({"kind": "response", "seq": seq, "response": {"text": "", "reasoning": "hmm", "tool_calls": []}}));
        later.push(json!({"role": "user", "content": "Continue."}));
        seq += 1;
    }
    lines.push(request(seq, &later));
    lines.push(json!({"kind": "response", "seq": seq, "response": {"text": "Fixed.", "reasoning": "", "tool_calls": []}}));
    if cut_short {
        let mut after = later.clone();
        after.extend([
            json!({"role": "assistant", "content": "Fixed."}),
            json!({"role": "user", "content": "Time is up."}),
        ]);
        lines.push(request(seq + 1, &after));
    }
    lines.iter().map(|line| format!("{line}\n")).collect()
}

/// The bytes of the picture a trial with a picture shows.
const PICTURE: &[u8] = b"\x89PNG drawing";

/// Where the trajectory names its picture: `images/<sha256>.png` beside it,
/// as the benchmark exports it.
fn picture_path() -> String {
    format!("images/{}.png", sha256_hex(PICTURE))
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
        let mut files: Vec<(String, Vec<u8>)> = vec![
            (
                "trajectory.atif.json".into(),
                trajectory(t.commit.unwrap_or(t.name), t.picture, t.command, t.traced).into_bytes(),
            ),
            (
                "model-requests.jsonl".into(),
                if t.traced {
                    requests_as_sent(t.commit.unwrap_or(t.name), t.command, t.cut_short)
                } else {
                    requests(t.commit.unwrap_or(t.name))
                }
                .into_bytes(),
            ),
            (
                "verification.json".into(),
                match t.reason {
                    Some(reason) => json!({"verdict": "fail", "reason": reason})
                        .to_string()
                        .into_bytes(),
                    None => b"{\"ok\": true}".to_vec(),
                },
            ),
            ("submission.patch".into(), b"diff --git a/x b/x\n".to_vec()),
        ];
        if t.picture {
            files.push((picture_path(), PICTURE.to_vec()));
        }
        for (name, bytes) in &files {
            let path = trial_dir.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        let reward = if t.pass { 1.0 } else { 0.0 };
        let verdict = if t.pass { "pass" } else { "fail" };
        let mut task = json!({"id": t.task, "version": "2", "digest": "d1",
                              "dimension": t.dimension, "path": "tasks/x"});
        if let Some(set) = t.set {
            task["set"] = json!(set);
        }
        if let Some(ids) = t.capabilities {
            task["capabilities"] = json!(ids);
        }
        let mut manifest = json!({
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
                .map(|(n, bytes)| (n.clone(), json!(sha256_hex(bytes))))
                .collect::<serde_json::Map<String, Value>>(),
        });
        if let Some(score) = &t.score {
            manifest["score"] = score.clone();
        }
        if t.confidential {
            manifest["confidential"] = json!(true);
        }
        if t.traced {
            let (requests, answered) = if t.cut_short { (4, 3) } else { (2, 2) };
            manifest["requests"] = json!({"file": "model-requests.jsonl",
                "schema": "sven.request-log/1", "requests": requests, "answered": answered});
        }
        let manifest_text = serde_json::to_string_pretty(&manifest).unwrap();
        fs::write(trial_dir.join("manifest.json"), &manifest_text).unwrap();
        let mut row = json!({"trial": t.name, "task": t.task, "reward": reward,
                             "verdict": verdict, "manifest_sha256": sha256_hex(manifest_text.as_bytes()),
                             "path": t.name});
        if let Some(set) = t.set {
            row["set"] = json!(set);
        }
        if let Some(ids) = t.capabilities {
            row["capabilities"] = json!(ids);
        }
        index.push_str(&format!("{row}\n"));
    }
    fs::write(dir.join("index.jsonl"), index).unwrap();
    dir.to_path_buf()
}
