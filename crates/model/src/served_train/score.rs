// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that measure a model on
// held-out text through the service that serves it, for its clients. If your
// team needs expertise in teacher-forced evaluation over a model API, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A base, or a base with an adapter, scored teacher-forced on chat records:
//! the `score` action a served chat model takes through `POST /v1/run`; and
//! given token ids scored after given context ids ([`score_tokens`]).
//!
//! Every message marked `"train": true` is an answer, scored under the
//! model's full distribution at temperature 1 as the model is asked for it
//! (its end-of-turn token included); the score is the mean negative
//! log-likelihood per answer token. The model is the one the endpoint serves
//! the weights under ([`BrainEndpoints::locate`]), so an adapter is offered to
//! the server first when it does not list it. The service bounds a call, so
//! the records go in calls of at most [`CALL_RECORDS`] records and
//! [`CALL_ANSWERS`] answers.

use std::path::Path;

use base64::Engine as _;
use serde_json::{json, Value};
use splinter_core::training::HeldOutScore;

use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::local::LocalWeights;

/// The most records brain scores in one call.
pub const CALL_RECORDS: usize = 4096;
/// The most answers brain scores in one call.
pub const CALL_ANSWERS: usize = 8192;

/// `weights` scored on the chat records of `dataset`: the mean negative
/// log-likelihood per supervised token, over every answer. The token
/// accuracy is not measured by the action and stays absent.
///
/// # Errors
/// The file cannot be read or holds a line that is not JSON, the endpoint
/// does not serve the weights, or the service refused a record (it names
/// it) or answered something else than a score.
pub fn score_chat(
    brain: &BrainEndpoints,
    weights: &LocalWeights,
    dataset: &Path,
    thinking: bool,
) -> Result<HeldOutScore, PolicyError> {
    let unreadable = |reason: String| PolicyError::Dataset {
        path: dataset.to_path_buf(),
        reason,
    };
    let text = std::fs::read_to_string(dataset).map_err(|e| unreadable(e.to_string()))?;
    let model = brain
        .locate(weights)?
        .into_iter()
        .next()
        .ok_or_else(|| unreadable("no endpoint to score on".into()))?;
    let mut batches: Vec<(String, usize)> = Vec::new();
    let (mut batch, mut records, mut answers) = (String::new(), 0usize, 0usize);
    for (index, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let record: Value = serde_json::from_str(line)
            .map_err(|e| unreadable(format!("line {}: {e}", index + 1)))?;
        let supervised = record["messages"]
            .as_array()
            .map_or(0, |m| m.iter().filter(|m| m["train"] == true).count());
        if records == CALL_RECORDS || (records > 0 && answers + supervised > CALL_ANSWERS) {
            batches.push((std::mem::take(&mut batch), records));
            (records, answers) = (0, 0);
        }
        batch.push_str(line);
        batch.push('\n');
        records += 1;
        answers += supervised;
    }
    if records > 0 {
        batches.push((batch, records));
    }
    let (mut tokens, mut logprob, mut scored) = (0u64, 0f64, 0usize);
    for (index, (records_text, count)) in batches.iter().enumerate() {
        let body = json!({
            "model": model,
            "action": "score",
            "params": {"enable_thinking": thinking},
            "blobs": {"records": {
                "media": "text",
                "data": base64::engine::general_purpose::STANDARD.encode(records_text),
            }},
        });
        let first = brain.pin(&(index as u64).to_le_bytes());
        let result = super::job::call(
            brain.urls(),
            brain.key(),
            brain.refused_wait(),
            first,
            &body,
        )
        .map_err(|e| match e {
            super::job::JobError::Transport(url, reason) => PolicyError::Endpoint { url, reason },
            super::job::JobError::Failed(reason) => {
                unreadable(format!("scoring under {model}: {reason}"))
            }
            super::job::JobError::Cancelled(_) => {
                unreadable(format!("scoring under {model} was cancelled"))
            }
        })?;
        let outputs = &result["outputs"];
        let (Some(t), Some(l)) = (outputs["tokens"].as_u64(), outputs["logprob"].as_f64()) else {
            return Err(PolicyError::Endpoint {
                url: brain.urls().join(", "),
                reason: format!("the score of {model} names no tokens or logprob: {outputs}"),
            });
        };
        tokens += t;
        logprob += l;
        scored += count;
    }
    Ok(HeldOutScore {
        loss: (tokens > 0).then(|| (-logprob / tokens as f64) as f32),
        token_accuracy: None,
        positions: usize::try_from(tokens).unwrap_or(usize::MAX),
        records: scored,
        skipped: 0,
    })
}

/// The log-probability `model` (an id the endpoints serve) gives each of
/// `token_ids` after `prompt_token_ids`, both as given, teacher-forced at
/// `temperature` over the whole vocabulary: one list per pair of `answers`,
/// in order. The figure a sampled answer's behaviour log-probabilities are
/// checked against.
///
/// # Errors
/// More answers than one call scores, the service refused a pair (it names
/// it), or answered without a log-probability per id.
pub fn score_tokens(
    brain: &BrainEndpoints,
    model: &str,
    answers: &[(Vec<u32>, Vec<u32>)],
    temperature: f64,
) -> Result<Vec<Vec<f64>>, PolicyError> {
    let refused = |reason: String| PolicyError::Endpoint {
        url: brain.urls().join(", "),
        reason,
    };
    if answers.len() > CALL_RECORDS {
        return Err(refused(format!(
            "{} answers to score, more than the {CALL_RECORDS} one call takes",
            answers.len()
        )));
    }
    let mut records = String::new();
    for (prompt, generated) in answers {
        records.push_str(&json!({"prompt_token_ids": prompt, "token_ids": generated}).to_string());
        records.push('\n');
    }
    let body = json!({
        "model": model,
        "action": "score",
        "params": {"per_token": true, "temperature": temperature},
        "blobs": {"records": {
            "media": "text",
            "data": base64::engine::general_purpose::STANDARD.encode(&records),
        }},
    });
    let result = super::job::call(
        brain.urls(),
        brain.key(),
        brain.refused_wait(),
        brain.pin(model.as_bytes()),
        &body,
    )
    .map_err(|e| match e {
        super::job::JobError::Transport(url, reason) => PolicyError::Endpoint { url, reason },
        super::job::JobError::Failed(reason) => refused(format!("scoring under {model}: {reason}")),
        super::job::JobError::Cancelled(_) => {
            refused(format!("scoring under {model} was cancelled"))
        }
    })?;
    let scored = result["outputs"]["records"]
        .as_array()
        .filter(|r| r.len() == answers.len())
        .ok_or_else(|| refused(format!("the score of {model} names no record per answer")))?;
    scored
        .iter()
        .zip(answers)
        .map(|(record, (_, generated))| {
            let values: Option<Vec<f64>> = record["answers"][0]["token_logprobs"]
                .as_array()
                .map(|v| v.iter().filter_map(Value::as_f64).collect());
            values
                .filter(|v| v.len() == generated.len())
                .ok_or_else(|| {
                    refused(format!(
                        "the score of {model} names no log-probability per generated id"
                    ))
                })
        })
        .collect()
}
