// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Training records from runs the supervisor can vouch for.
//!
//! A run qualifies as training material only when everything that decides
//! its worth was measured, not claimed: the loop's own acceptance checks
//! passed on the candidate, no supervisor hint was given, the model was a
//! local one, and the record of the accepted attempt is complete (no event
//! dropped, no provider failure). Only the accepted attempt is used: an
//! attempt that failed, however instructive, is never a positive example, and
//! neither is anything the accepted attempt did not do. The conversation is
//! rebuilt from the event stream (what the model said and asked for, what the
//! tools answered, what the loop told it after a failed check), with the
//! model's own turns the only ones marked for training.
//!
//! The records are in the chat format brain trains from: messages with an
//! explicit training flag, tool calls with their arguments as a JSON string,
//! and the tool definitions the model was offered. Everything in a record
//! has passed through the same redaction as the trace.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_sdk::vocabulary::digest::Digest;

use crate::outcome::{Outcome, Status};
use crate::run::system_prompt;
use crate::store::{read_json, LoopHome, OUTCOME_FILE};
use crate::trace::{read_events, ARTIFACTS_DIR};

/// The version of the dataset manifest.
pub const DATASET_SCHEMA: u32 = 1;

/// Why a run was left out of a dataset.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Ineligible(pub String);

/// What became of one run offered to the dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Considered {
    /// The run.
    pub run: String,
    /// The accepted attempt used, when the run qualified.
    pub attempt: Option<u32>,
    /// Why it did not qualify.
    pub excluded: Option<String>,
    /// The digest of the record written for it.
    pub record_digest: Option<String>,
}

/// What a dataset is made of, written beside it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    /// The manifest's version.
    pub schema: u32,
    /// Every run offered, with its fate.
    pub runs: Vec<Considered>,
    /// Records written.
    pub records: usize,
    /// SHA-256 of the system prompt every record carries.
    pub system_prompt_digest: String,
    /// SHA-256 of the dataset file's bytes.
    pub dataset_digest: String,
}

/// An event's data with an artifact reference replaced by what it points at.
fn full(dir: &Path, data: &Value) -> Result<Value> {
    let Some(address) = data.get("artifact").and_then(Value::as_str) else {
        return Ok(data.clone());
    };
    let digest =
        Digest::parse(address).with_context(|| format!("bad artifact address {address}"))?;
    let path = dir.join(ARTIFACTS_DIR).join(digest.hex());
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading artifact {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))
}

/// One conversation as brain's chat record.
fn record_of(
    run: &str,
    attempt: u32,
    events: &[Value],
    dir: &Path,
    system: &str,
    tools: &[Value],
) -> Result<Result<Value, Ineligible>> {
    let mut messages = vec![json!({"role": "system", "content": system, "train": false})];
    let (mut text, mut calls): (String, Vec<(String, String, Value)>) = (String::new(), Vec::new());
    let mut in_results = false;
    let mut next_id = 0usize;
    let mut round_ids: Vec<(String, String)> = Vec::new();
    let mut flush = |messages: &mut Vec<Value>,
                     text: &mut String,
                     calls: &mut Vec<(String, String, Value)>,
                     ids: &mut Vec<(String, String)>| {
        if calls.is_empty() && text.trim().is_empty() {
            return;
        }
        let tool_calls: Vec<Value> = calls
            .drain(..)
            .map(|(call_id, name, args)| {
                next_id += 1;
                let unique = format!("call_{next_id}");
                ids.push((call_id, unique.clone()));
                json!({"id": unique, "type": "function",
                       "function": {"name": name, "arguments": args.to_string()}})
            })
            .collect();
        // A turn that only calls tools has no content: the whitespace a
        // model leaves around an empty reasoning block would make the chat
        // template render the message differently alone and in context, and
        // the trainer refuses a record it cannot give a loss mask for.
        let content = std::mem::take(text).trim().to_string();
        messages.push(json!({"role": "assistant", "content": content,
                             "tool_calls": tool_calls, "train": true}));
    };
    for event in events {
        let kind = event["type"].as_str().unwrap_or_default();
        // Only the events that carry the conversation are read back from
        // their artifacts; an audit log is an artifact too, and is not JSON.
        let data = if matches!(
            kind,
            "attempt_started" | "follow_up" | "tool_request" | "assistant_message" | "tool_result"
        ) {
            full(dir, &event["data"])?
        } else {
            Value::Null
        };
        match kind {
            "attempt_started" => {
                let prompt = data["prompt"].as_str().unwrap_or_default();
                messages.push(json!({"role": "user", "content": prompt, "train": false}));
            }
            "follow_up" => {
                flush(&mut messages, &mut text, &mut calls, &mut round_ids);
                in_results = false;
                let said = data["message"].as_str().unwrap_or_default();
                messages.push(json!({"role": "user", "content": said, "train": false}));
            }
            "tool_request" => {
                if in_results {
                    in_results = false;
                    round_ids.clear();
                }
                calls.push((
                    data["call_id"].as_str().unwrap_or_default().to_string(),
                    data["tool"].as_str().unwrap_or_default().to_string(),
                    data["arguments"].clone(),
                ));
            }
            "assistant_message" => {
                if in_results {
                    in_results = false;
                    round_ids.clear();
                }
                text.push_str(data["text"].as_str().unwrap_or_default());
            }
            "tool_result" => {
                if !in_results {
                    flush(&mut messages, &mut text, &mut calls, &mut round_ids);
                    in_results = true;
                }
                let call_id = data["call_id"].as_str().unwrap_or_default();
                let Some((_, unique)) = round_ids.iter().find(|(c, _)| c == call_id) else {
                    return Ok(Err(Ineligible(format!(
                        "a tool result answers no call: {call_id}"
                    ))));
                };
                messages.push(
                    json!({"role": "tool", "content": data["output"].as_str().unwrap_or_default(),
                                     "tool_call_id": unique, "train": false}),
                );
            }
            "events_dropped" => {
                return Ok(Err(Ineligible(
                    "events were dropped during the attempt".into(),
                )));
            }
            "provider_error" | "provider_retry" => {
                return Ok(Err(Ineligible(
                    "the provider failed during the attempt".into(),
                )));
            }
            _ => {}
        }
    }
    // What the model said last, with no call: the closing answer.
    if !text.trim().is_empty() && calls.is_empty() {
        messages.push(
            json!({"role": "assistant", "content": text.trim(), "tool_calls": [], "train": true}),
        );
    } else if !calls.is_empty() {
        return Ok(Err(Ineligible(
            "the attempt ended on an unanswered tool call".into(),
        )));
    }
    let trained = messages.iter().filter(|m| m["train"] == true).count();
    if trained < 2 || messages.last().map(|m| m["role"].as_str()) != Some(Some("assistant")) {
        return Ok(Err(Ineligible(
            "the conversation has no closing answer".into(),
        )));
    }
    Ok(Ok(json!({"messages": messages, "tools": tools,
                 "metadata": {"run": run, "attempt": attempt}})))
}

/// What a training record holds of an accepted run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// The whole conversation of the accepted attempt: the model's tool
    /// calls, their results and the closing answer. A trainer whose chat
    /// template renders an assistant turn differently depending on whether
    /// it is the last one (Qwen3's does) refuses such a record.
    Conversation,
    /// One exchange: the task as the attempt was given it, and the patch
    /// that was accepted as the single assistant turn. What it teaches is
    /// writing the change, not the use of tools.
    Patch,
}

/// The record of the accepted attempt of `run`, or why there is none.
pub fn record_for_run(home: &LoopHome, run: &str) -> Result<Result<(u32, Value), Ineligible>> {
    record_for_run_in(home, run, Form::Conversation)
}

/// [`record_for_run`] in the given [`Form`].
pub fn record_for_run_in(
    home: &LoopHome,
    run: &str,
    form: Form,
) -> Result<Result<(u32, Value), Ineligible>> {
    let dir = home.run_dir(run);
    let outcome: Outcome = match read_json(&dir.join(OUTCOME_FILE)) {
        Ok(o) => o,
        Err(_) => return Ok(Err(Ineligible("the run has no outcome".into()))),
    };
    if outcome.status != Status::Accepted {
        return Ok(Err(Ineligible(format!(
            "the run was {:?}, not accepted",
            outcome.status
        ))));
    }
    if outcome.assistance != "unaided" {
        return Ok(Err(Ineligible("the run was assisted".into())));
    }
    let events = read_events(&dir)?;
    let class = events
        .iter()
        .find(|e| e["type"] == "model_selected")
        .and_then(|e| e["data"]["class"].as_str())
        .unwrap_or("unknown")
        .to_string();
    if class != "local" && class != "local_served" {
        return Ok(Err(Ineligible(format!(
            "the model was {class}: remote output is not training material here"
        ))));
    }
    let Some(attempt) = outcome
        .attempts
        .iter()
        .rev()
        .find(|a| a.failure.is_none())
        .map(|a| a.attempt)
    else {
        return Ok(Err(Ineligible("no attempt passed".into())));
    };
    let (system, _) = system_prompt(home)?;
    let contract: crate::contract::Contract = read_json(&dir.join(crate::store::CONTRACT_FILE))?;
    if Digest::sha256_of(system.as_bytes()).to_string() != contract.system_prompt_digest {
        return Ok(Err(Ineligible(
            "the system prompt in effect is not the one the run used".into(),
        )));
    }
    if form == Form::Patch {
        let patch = std::fs::read_to_string(dir.join("patch.diff")).unwrap_or_default();
        if patch.trim().is_empty() {
            return Ok(Err(Ineligible("the accepted run kept no patch".into())));
        }
        let task = crate::attempt::compose_prompt(&contract, None);
        return Ok(Ok((
            attempt,
            json!({"messages": [
                    {"role": "system", "content": system, "train": false},
                    {"role": "user", "content": task, "train": false},
                    {"role": "assistant", "content": patch.trim(), "tool_calls": [], "train": true}],
                   "tools": [],
                   "metadata": {"run": run, "attempt": attempt, "form": "patch"}}),
        )));
    }
    let tools = tool_definitions(&dir, attempt)?;
    let mine: Vec<Value> = events
        .iter()
        .filter(|e| e["attempt"].as_u64() == Some(u64::from(attempt)))
        .cloned()
        .collect();
    Ok(record_of(run, attempt, &mine, &dir, &system, &tools)?.map(|r| (attempt, r)))
}

/// The tool definitions the model was offered in an attempt, from sven's own
/// trajectory of it.
fn tool_definitions(dir: &Path, attempt: u32) -> Result<Vec<Value>> {
    let path = dir
        .join("trajectories")
        .join(format!("attempt-{attempt}.atif.json"));
    let trajectory: Value = read_json(&path)?;
    Ok(trajectory["agent"]["tool_definitions"]
        .as_array()
        .cloned()
        .unwrap_or_default())
}

/// Writes the dataset of `runs` to `out` and its manifest beside it
/// (`<out>.manifest.json`). The same runs give the same bytes.
pub fn export(home: &LoopHome, runs: &[String], out: &Path) -> Result<Manifest> {
    export_in(home, runs, out, Form::Conversation)
}

/// [`export`] with records in the given [`Form`].
pub fn export_in(home: &LoopHome, runs: &[String], out: &Path, form: Form) -> Result<Manifest> {
    let mut seen = BTreeSet::new();
    let (mut lines, mut considered) = (Vec::new(), Vec::new());
    for run in runs {
        match record_for_run_in(home, run, form)? {
            Ok((attempt, record)) => {
                let line = serde_json::to_string(&record)?;
                let digest = Digest::sha256_of(line.as_bytes()).to_string();
                if !seen.insert(digest.clone()) {
                    considered.push(Considered {
                        run: run.clone(),
                        attempt: Some(attempt),
                        excluded: Some("a duplicate of an earlier record".into()),
                        record_digest: Some(digest),
                    });
                    continue;
                }
                considered.push(Considered {
                    run: run.clone(),
                    attempt: Some(attempt),
                    excluded: None,
                    record_digest: Some(digest),
                });
                lines.push(line);
            }
            Err(Ineligible(why)) => considered.push(Considered {
                run: run.clone(),
                attempt: None,
                excluded: Some(why),
                record_digest: None,
            }),
        }
    }
    let mut text = lines.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    crate::store::write_atomic(out, text.as_bytes())?;
    let (system, _) = system_prompt(home)?;
    let manifest = Manifest {
        schema: DATASET_SCHEMA,
        records: lines.len(),
        runs: considered,
        system_prompt_digest: Digest::sha256_of(system.as_bytes()).to_string(),
        dataset_digest: Digest::sha256_of(text.as_bytes()).to_string(),
    };
    let mut manifest_path = out.as_os_str().to_owned();
    manifest_path.push(".manifest.json");
    crate::store::write_json(Path::new(&manifest_path), &manifest)?;
    Ok(manifest)
}
