// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning from the exact
// tokens an agent's policy sampled, for its clients. If your team needs
// expertise in training agents on their own rollouts while they run, you can
// procure our services by sending an email to info@swedishembedded.com.

//! One training record per model call, from what the agent runtime recorded
//! of each request exactly.
//!
//! A rollout of an online run is trained call by call: each model request is
//! a record whose context is the request as it was sent (its messages and
//! tools, from the session's request log) and whose one supervised turn is
//! the answer, carrying the ids the sampler was given and generated, the
//! policy's log-probability of each generated id, the sampler's top-K when it
//! was asked for, the policy version that decoded it and the sampling
//! controls it was drawn under. Nothing is re-rendered, merged or guessed;
//! a call is never paired with a step or a conversation it did not have.
//!
//! What sven recorded of each request is read from the trajectory: a step
//! that holds several requests lists each under `metrics.extra.sven.requests`
//! (authoritative: the step's own fields then run the requests together),
//! a step of one request carries it in its own metrics and `extra.sven`. The
//! requests are taken in step order and paired, in order, with the request
//! log's calls of the policy; a rollout whose two counts differ pairs none.
//!
//! A call is exact when its request carries prompt and generated ids, one
//! policy log-probability per generated id (`logprob_kind` `policy`, at the
//! sampling temperature), the tokenizer's digest (the same for every call of
//! the rollout), a policy version, a well-formed top-K when one was recorded,
//! and an answer the scrub left as the model wrote it. A rollout is exact when
//! every call is ([`CallProjection::exact`]); any other is counted by reason
//! ([`ProxyReason`]) and its calls are never training material. Records carry
//! no reward or group: the learner's adapter adds the scored group's
//! ([`CallRecord::record`]).

use std::collections::BTreeMap;

use atif::{StepOrigin, Trajectory};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::chat::WireMessage;
use splinter_core::harness::ModelCall;

use crate::markers::chat_tool_list;
use crate::record::{Record, RecordBody, RecordMetadata, RolloutOutcome};
use crate::requests::{scope_calls, sent_and_answered};
use crate::sampled::{
    sampled_record, token_ids, trained_ids, ProxyReason, SampledTurn, SampledTurns, StepIds,
    POLICY_KIND,
};
use crate::scrub::Scrub;
use crate::Objective;

/// How far the recorded log-probability temperature may be from the
/// sampling temperature: the same figure written by two programs.
const TEMPERATURE_TOLERANCE: f64 = 1e-6;

/// The view name call records carry in their metadata.
pub const CALL_VIEW: &str = "online-calls";

/// The sampling controls a record's tokens were drawn under, as brain's
/// training records declare them (`sampler`): a control left out was not
/// applied.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampler {
    /// The sampling temperature, above zero.
    pub temperature: f64,
    /// The top-k cut, when one was applied (at least 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// The nucleus, when one below 1 was applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// The min-p cut, when one above 0 was applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
}

impl Sampler {
    /// Sampling from the whole distribution at `temperature`.
    ///
    /// # Errors
    /// The temperature is not a finite number above zero.
    pub fn untruncated(temperature: f64) -> Result<Self, String> {
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(format!(
                "the sampling temperature {temperature} is not a finite number above zero"
            ));
        }
        Ok(Self {
            temperature,
            top_k: None,
            top_p: None,
            min_p: None,
        })
    }

    /// Whether the tokens were drawn from a cut of the distribution (top-k,
    /// top-p below 1, min-p above 0) rather than the whole row: the policy
    /// log-probability is then not the behaviour probability.
    #[must_use]
    pub fn truncates(&self) -> bool {
        self.top_k.is_some_and(|k| k > 0)
            || self.top_p.is_some_and(|p| p < 1.0)
            || self.min_p.is_some_and(|p| p > 0.0)
    }
}

/// One model call of a rollout, ready to train once its group is scored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallRecord {
    /// The request as sent, then the answer: the one supervised turn.
    pub messages: Vec<WireMessage>,
    /// The tools the request offered, as brain's chat template takes them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    /// The answer's sampled tokens: one turn.
    pub sampled: SampledTurns,
    /// The policy version that decoded the call.
    pub policy_version: u64,
    /// The controls it was drawn under.
    pub sampler: Sampler,
    /// The rollout it belongs to stopped at a length limit.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

impl CallRecord {
    /// The training record of this call once its rollout `rollout` earned
    /// `reward` in the group `group` (keys of 1 to 512 bytes without control
    /// characters, as the controller makes them): brain's training-record
    /// contract (version 2) through [`crate::dataset_line`].
    #[must_use]
    pub fn record(&self, reward: f64, group: &str, rollout: &str) -> Record {
        // The online run's group and rollout keys are what the learner
        // deduplicates by, so they are the record's own, not digests.
        let outcome = RolloutOutcome {
            reward,
            group: group.to_string(),
            rollout: Some(rollout.to_string()),
            baseline_only: false,
            sampled: Some(self.sampled.clone()),
            policy_version: Some(self.policy_version),
            sampler: Some(self.sampler),
            truncated: self.truncated,
        };
        Record {
            body: RecordBody::Chat {
                messages: self.messages.clone(),
                tools: self.tools.clone(),
                outcome: None,
                teacher_context: Vec::new(),
            }
            .with_outcome(outcome),
            metadata: RecordMetadata::of(CALL_VIEW, Objective::PolicyGradient),
        }
    }
}

/// The calls of one rollout, projected.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CallProjection {
    /// The exact calls' records, in request order.
    pub records: Vec<CallRecord>,
    /// The policy's model calls the request log holds.
    pub calls: usize,
    /// Calls that are not exact, by reason.
    pub proxy: BTreeMap<ProxyReason, usize>,
    /// The oldest and newest policy version among the requests, when every
    /// request named a whole-number version.
    pub span: Option<(u64, u64)>,
    /// Some request stopped at its length limit.
    pub truncated: bool,
    /// Ids the policy generated over every request.
    pub tokens: u64,
}

impl CallProjection {
    /// Whether every call is exact: the rollout is training material.
    #[must_use]
    pub fn exact(&self) -> bool {
        self.calls > 0
            && self.proxy.is_empty()
            && self.records.len() == self.calls
            && self.span.is_some()
    }
}

/// What sven recorded of one request.
#[derive(Debug)]
struct Traced {
    /// The prompt and generated ids, or why they cannot be read.
    ids: Result<Option<StepIds>, ProxyReason>,
    logprobs: Option<Vec<f64>>,
    kind: Option<String>,
    temperature: Option<f64>,
    version: Option<String>,
    digest: Option<String>,
    top_ids: Option<Vec<Vec<i64>>>,
    top_logprobs: Option<Vec<Vec<f64>>>,
    truncated: bool,
}

fn field<T: serde::de::DeserializeOwned>(object: &Value, key: &str) -> Option<T> {
    object
        .get(key)
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

impl Traced {
    /// The identity and top lists from `facts` (a request entry, or a single
    /// request step's `extra.sven`), with its ids and log-probabilities.
    fn of(
        facts: &Value,
        sampled: Option<&Value>,
        prompt: Option<Vec<i64>>,
        completion: Option<Vec<i64>>,
        logprobs: Option<Vec<f64>>,
    ) -> Self {
        Self {
            ids: trained_ids(sampled, prompt.as_deref(), completion.as_deref()),
            logprobs,
            kind: field(facts, "logprob_kind"),
            temperature: field(facts, "logprob_temperature"),
            version: field(facts, "policy_version"),
            digest: field(facts, "tokenizer_digest"),
            top_ids: field(facts, "generation_top_ids"),
            top_logprobs: field(facts, "generation_top_log_probs"),
            truncated: field(facts, "truncated").unwrap_or(false),
        }
    }
}

/// Every request the trajectory's agent steps recorded, in order.
fn requests(trajectory: &Trajectory) -> Vec<Traced> {
    let mut out = Vec::new();
    let empty = Value::Null;
    for step in trajectory
        .steps
        .iter()
        .filter(|s| s.source == StepOrigin::Agent)
    {
        let Some(metrics) = &step.metrics else {
            continue;
        };
        let sven = metrics
            .extra
            .as_ref()
            .and_then(|e| e.get("sven"))
            .unwrap_or(&empty);
        let sampled = sampled_record(step);
        match sven.get("requests").and_then(Value::as_array) {
            // The step's sampled record is its last reply's.
            Some(entries) => out.extend(entries.iter().enumerate().map(|(k, entry)| {
                Traced::of(
                    entry,
                    sampled.filter(|_| k + 1 == entries.len()),
                    field(entry, "prompt_token_ids"),
                    field(entry, "completion_token_ids"),
                    field(entry, "logprobs"),
                )
            })),
            None => out.push(Traced::of(
                sven,
                sampled,
                metrics.prompt_token_ids.clone(),
                metrics.completion_token_ids.clone(),
                metrics.logprobs.clone(),
            )),
        }
    }
    out
}

/// The digest, sampled turn and policy version of `traced`, when it is
/// exact under `sampler`.
fn exact_turn(
    traced: &Traced,
    sampler: &Sampler,
) -> Result<(String, SampledTurn, u64), ProxyReason> {
    let (Some((prompt, completion)), Some(logprobs), Some(digest)) =
        (traced.ids.clone()?, &traced.logprobs, &traced.digest)
    else {
        return Err(ProxyReason::NoTrace);
    };
    match traced.kind.as_deref() {
        Some(POLICY_KIND) => {}
        Some(_) => return Err(ProxyReason::SamplerLogprobs),
        None => return Err(ProxyReason::NoTrace),
    }
    if !traced
        .temperature
        .is_some_and(|t| (t - sampler.temperature).abs() <= TEMPERATURE_TOLERANCE)
    {
        return Err(ProxyReason::TemperatureMismatch);
    }
    if prompt.is_empty() {
        return Err(ProxyReason::NoTrace);
    }
    if completion.is_empty() || completion.len() != logprobs.len() {
        return Err(ProxyReason::CountMismatch);
    }
    if logprobs.iter().any(|l| !l.is_finite() || *l > 0.0) {
        return Err(ProxyReason::InvalidLogprobs);
    }
    let version = traced
        .version
        .as_deref()
        .and_then(|v| v.parse::<u64>().ok())
        .ok_or(ProxyReason::NoPolicyVersion)?;
    let (top_ids, top_logprobs) = top_lists(traced, completion.len())?;
    // The trainer reads f32; the policy's own figures are f32 in the server.
    #[allow(clippy::cast_possible_truncation)]
    let narrow = |row: &[f64]| row.iter().map(|l| *l as f32).collect::<Vec<f32>>();
    Ok((
        digest.clone(),
        SampledTurn {
            prompt_token_ids: prompt,
            token_ids: completion,
            behavior_logprobs: narrow(logprobs),
            behavior_top_ids: top_ids,
            behavior_top_logprobs: top_logprobs
                .map(|rows| rows.iter().map(|r| narrow(r)).collect()),
        },
        version,
    ))
}

type TopLists = (Option<Vec<Vec<u32>>>, Option<Vec<Vec<f64>>>);

/// The request's top-K, checked: none, or both lists with one row per
/// generated id, every row the same `K` of distinct valid ids and
/// log-probabilities finite and at most zero.
fn top_lists(traced: &Traced, generated: usize) -> Result<TopLists, ProxyReason> {
    let (ids, logprobs) = match (&traced.top_ids, &traced.top_logprobs) {
        (None, None) => return Ok((None, None)),
        (Some(ids), Some(logprobs)) => (ids, logprobs),
        _ => return Err(ProxyReason::TopMismatch),
    };
    let k = ids.first().map_or(0, Vec::len);
    let rows_fit = ids.len() == generated
        && logprobs.len() == generated
        && k > 0
        && ids
            .iter()
            .zip(logprobs)
            .all(|(i, l)| i.len() == k && l.len() == k);
    if !rows_fit {
        return Err(ProxyReason::TopMismatch);
    }
    let mut parsed = Vec::with_capacity(ids.len());
    for (row, values) in ids.iter().zip(logprobs) {
        let row = token_ids(row)?;
        let distinct: std::collections::HashSet<_> = row.iter().collect();
        if distinct.len() != row.len() || values.iter().any(|l| !l.is_finite() || *l > 0.0) {
            return Err(ProxyReason::TopMismatch);
        }
        parsed.push(row);
    }
    Ok((Some(parsed), Some(logprobs.clone())))
}

/// The request `call` and its answer as one record's messages and tools,
/// the answer supervised, call ids named by the assistant turn that made
/// them; `Err` when it cannot be read or the scrub changed its text.
fn messages_of(
    call: &ModelCall,
    trajectory: &Trajectory,
) -> Result<(Vec<WireMessage>, Vec<Value>), ProxyReason> {
    let scrub = Scrub::of(trajectory);
    let unreadable = |_| ProxyReason::UnreadableRequest;
    let (mut messages, answer) = sent_and_answered(call, &scrub, call.seq).map_err(unreadable)?;
    let answer = answer.ok_or(ProxyReason::EmptyAnswer)?;
    if !answer.parsed {
        return Err(ProxyReason::UnparsedArguments);
    }
    let mut answer = answer.turn;
    if scrub.changed_anything() {
        return Err(ProxyReason::Scrubbed);
    }
    answer.train = true;
    messages.push(answer);
    let turns: std::collections::HashMap<usize, (usize, u64)> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == "assistant")
        .enumerate()
        .map(|(turn, (at, _))| (at, (turn, u64::try_from(turn).unwrap_or(u64::MAX))))
        .collect();
    scope_calls(&mut messages, &turns).map_err(unreadable)?;
    let tools = chat_tool_list(&call.tools).map_err(unreadable)?;
    Ok((messages, tools))
}

/// The policy-version span the trajectory's totals record
/// (`final_metrics.extra.sven.policy_version_span`).
fn recorded_span(trajectory: &Trajectory) -> Option<(u64, u64)> {
    let span: [u64; 2] = trajectory
        .final_metrics
        .as_ref()?
        .extra
        .as_ref()?
        .pointer("/sven/policy_version_span")
        .and_then(|v| serde_json::from_value(v.clone()).ok())?;
    (span[0] <= span[1]).then_some((span[0], span[1]))
}

/// The records of a rollout whose trajectory is `trajectory` and whose
/// session's request log held `calls` for its policy, in order, drawn under
/// `sampler`.
#[must_use]
pub fn project_calls(
    trajectory: &Trajectory,
    calls: &[ModelCall],
    sampler: &Sampler,
) -> CallProjection {
    let traced = requests(trajectory);
    let mut projection = CallProjection {
        calls: calls.len(),
        tokens: traced
            .iter()
            .map(|t| {
                t.ids
                    .as_ref()
                    .ok()
                    .and_then(Option::as_ref)
                    .map_or(0, |(_, c)| c.len()) as u64
            })
            .sum(),
        truncated: traced.iter().any(|t| t.truncated),
        ..CallProjection::default()
    };
    let miss = |projection: &mut CallProjection, reason| {
        *projection.proxy.entry(reason).or_insert(0) += 1;
    };
    if traced.len() != calls.len() {
        for _ in 0..calls.len().max(traced.len()) {
            miss(&mut projection, ProxyReason::Unpaired);
        }
        return projection;
    }
    let mut digest: Option<String> = None;
    let mut versions = Vec::with_capacity(calls.len());
    for (call, traced) in calls.iter().zip(&traced) {
        let turn = exact_turn(traced, sampler).and_then(|(d, turn, version)| {
            if digest.get_or_insert_with(|| d.clone()) != &d {
                return Err(ProxyReason::MixedTokenizer);
            }
            Ok((d, turn, version))
        });
        let built = turn.and_then(|t| messages_of(call, trajectory).map(|m| (t, m)));
        match built {
            Ok(((digest, turn, version), (messages, tools))) => {
                versions.push(version);
                projection.records.push(CallRecord {
                    messages,
                    tools,
                    sampled: SampledTurns {
                        tokenizer_digest: digest,
                        turns: vec![Some(turn)],
                    },
                    policy_version: version,
                    sampler: *sampler,
                    truncated: false,
                });
            }
            Err(reason) => miss(&mut projection, reason),
        }
    }
    let computed = (versions.len() == calls.len()).then(|| {
        let oldest = versions.iter().copied().min().unwrap_or(0);
        (oldest, versions.iter().copied().max().unwrap_or(0))
    });
    projection.span = recorded_span(trajectory).or(computed);
    let truncated = projection.truncated;
    for record in &mut projection.records {
        record.truncated = truncated;
    }
    projection
}
