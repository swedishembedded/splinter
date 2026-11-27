// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training on whole agent sessions of the
// tokens a policy sampled, for its clients. If your team needs expertise in
// turning multi-call agent episodes into exact and cheap training samples,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! One training record per agent session, from its exact call records, when
//! the session is append-only; otherwise the reason it stays one record per
//! call.
//!
//! A session's calls each re-send the context the earlier calls built. When
//! every call's request is the previous call's request, its sampled answer
//! and the turns between, token for token (the session is *append-only*),
//! the session is one sequence: the last request and its answer, every
//! answer the session sampled supervised in place with that call's own ids
//! and log-probabilities. brain trains such a record (with answers packed)
//! to exactly the gradient of its per-call records under `token-sum`, and
//! forwards the shared context once instead of once per call.
//!
//! The record is brain's packed rollout record: every sampled answer
//! `"train": true` with its call's `token_ids`, `prompt_token_ids`,
//! `behavior_logprobs` (and top-K when sampled so); `reward`, `group`,
//! `policy_version` (the oldest call's), `sampler` and `tokenizer_digest`
//! once on the record. brain refuses a record whose session is not
//! append-only (rule `session_not_append_only`), so it is never made here:
//! a session merges only when every call is exact, all were drawn under one
//! sampler and one tokenizer, every call `k + 1`'s prompt ids start with
//! call `k`'s prompt ids followed by its generated ids, and every call's
//! request and answer are the first messages of the next call's request.
//! Anything else is an [`Unmergeable`] naming the call (and the position)
//! where the property fails, and its calls are trained one record each:
//! nothing is merged by approximation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::chat::WireMessage;

use crate::calls::{CallProjection, CallRecord, Sampler};
use crate::record::{Record, RecordBody, RecordMetadata, RolloutOutcome};
use crate::sampled::{SampledTurn, SampledTurns};
use crate::Objective;

/// The view name session records carry in their metadata.
pub const SESSION_VIEW: &str = "online-sessions";

/// Why a session is trained one record per call, by kind: what a dataset
/// counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackReason {
    /// Some call is not exact (no trace, another temperature, unpaired, ...):
    /// the per-call projection already says why.
    NotExact,
    /// A call names no prompt ids, no generated ids, or not one behaviour
    /// log-probability per generated id.
    MissingIds,
    /// The calls were drawn under different sampling controls.
    SamplerDiffers,
    /// The calls' ids are of different tokenizers.
    TokenizerDiffers,
    /// A call's prompt ids do not start with the session so far.
    NotAppendOnly,
    /// A call's request does not start with the previous call's request and
    /// answer, message for message.
    MessagesDiverge,
    /// The calls offered different tools: one record holds one list.
    ToolsDiffer,
    /// The merged sequence is longer than the training block.
    LongerThanBlock,
}

impl FallbackReason {
    /// The reason's name, as a dataset report writes it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::NotExact => "not_exact",
            Self::MissingIds => "missing_ids",
            Self::SamplerDiffers => "sampler_differs",
            Self::TokenizerDiffers => "tokenizer_differs",
            Self::NotAppendOnly => "not_append_only",
            Self::MessagesDiverge => "messages_diverge",
            Self::ToolsDiffer => "tools_differ",
            Self::LongerThanBlock => "longer_than_block",
        }
    }
}

/// Why one session cannot be one record, naming where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unmergeable {
    /// The session's calls are not all exact.
    NotExact,
    /// Call `call` (0-based) names no prompt or generated ids, or not one
    /// log-probability per generated id.
    MissingIds {
        /// The call.
        call: usize,
    },
    /// Call `call` was drawn under other sampling controls than call 0.
    SamplerDiffers {
        /// The call.
        call: usize,
    },
    /// Call `call`'s ids are of another tokenizer than call 0's.
    TokenizerDiffers {
        /// The call.
        call: usize,
    },
    /// Call `call`'s prompt ids are not the session so far (call `call - 1`'s
    /// prompt ids and generated ids, then the turns between): they first
    /// differ at `position` (the prompt's length when it ends before the
    /// session so far does).
    NotAppendOnly {
        /// The call.
        call: usize,
        /// The first position where the prompt and the session differ.
        position: usize,
    },
    /// Call `call`'s request does not start with call `call - 1`'s request
    /// and answer: they first differ at message `message`.
    MessagesDiverge {
        /// The call.
        call: usize,
        /// The first message that differs.
        message: usize,
    },
    /// Call `call` offered other tools than the last call.
    ToolsDiffer {
        /// The call.
        call: usize,
    },
    /// The merged sequence holds `tokens` ids, more than the block's `block`.
    LongerThanBlock {
        /// The merged sequence's length.
        tokens: u64,
        /// The training block.
        block: u64,
    },
}

impl Unmergeable {
    /// What kind of reason it is.
    #[must_use]
    pub fn reason(&self) -> FallbackReason {
        match self {
            Self::NotExact => FallbackReason::NotExact,
            Self::MissingIds { .. } => FallbackReason::MissingIds,
            Self::SamplerDiffers { .. } => FallbackReason::SamplerDiffers,
            Self::TokenizerDiffers { .. } => FallbackReason::TokenizerDiffers,
            Self::NotAppendOnly { .. } => FallbackReason::NotAppendOnly,
            Self::MessagesDiverge { .. } => FallbackReason::MessagesDiverge,
            Self::ToolsDiffer { .. } => FallbackReason::ToolsDiffer,
            Self::LongerThanBlock { .. } => FallbackReason::LongerThanBlock,
        }
    }
}

impl std::fmt::Display for Unmergeable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotExact => write!(f, "not every call of the session is exact"),
            Self::MissingIds { call } => write!(
                f,
                "call {call} names no prompt or generated ids, or not one log-probability per generated id"
            ),
            Self::SamplerDiffers { call } => {
                write!(f, "call {call} was drawn under other sampling controls than call 0")
            }
            Self::TokenizerDiffers { call } => {
                write!(f, "call {call}'s ids are of another tokenizer than call 0's")
            }
            Self::NotAppendOnly { call, position } => write!(
                f,
                "the session is not append-only: call {call}'s prompt ids first differ from the session so far at position {position}"
            ),
            Self::MessagesDiverge { call, message } => write!(
                f,
                "call {call}'s request does not start with call {}'s request and answer: message {message} differs",
                call - 1
            ),
            Self::ToolsDiffer { call } => {
                write!(f, "call {call} offered other tools than the session's last call")
            }
            Self::LongerThanBlock { tokens, block } => write!(
                f,
                "the merged session holds {tokens} ids, more than the training block of {block}"
            ),
        }
    }
}

/// The first call whose prompt ids do not extend the session so far, and the
/// first position where they differ; `None` when every call's prompt ids
/// start with the previous call's prompt ids followed by its generated ids
/// (the session is append-only). This is the property brain holds a merged
/// record to (rule `session_not_append_only`).
#[must_use]
pub fn first_divergence(turns: &[&SampledTurn]) -> Option<(usize, usize)> {
    turns.windows(2).enumerate().find_map(|(at, pair)| {
        let (before, next) = (pair[0], pair[1]);
        let so_far = before
            .prompt_token_ids
            .iter()
            .chain(&before.token_ids)
            .copied();
        let prompt = &next.prompt_token_ids;
        let length = before.prompt_token_ids.len() + before.token_ids.len();
        match so_far.zip(prompt.iter().copied()).position(|(a, b)| a != b) {
            Some(position) => Some((at + 1, position)),
            None if prompt.len() < length => Some((at + 1, prompt.len())),
            None => None,
        }
    })
}

/// Whether `turn` names everything a merged answer must: prompt ids,
/// generated ids and one behaviour log-probability per generated id.
fn complete(turn: &SampledTurn) -> bool {
    !turn.prompt_token_ids.is_empty()
        && !turn.token_ids.is_empty()
        && turn.token_ids.len() == turn.behavior_logprobs.len()
}

/// Checks the token-level properties of a session's sampled answers, in
/// call order, each with its tokenizer digest: every one complete, one
/// tokenizer, and append-only.
///
/// # Errors
/// The first property that fails, naming the call.
pub fn check_turns(turns: &[(&str, &SampledTurn)]) -> Result<(), Unmergeable> {
    for (call, (digest, turn)) in turns.iter().enumerate() {
        if !complete(turn) {
            return Err(Unmergeable::MissingIds { call });
        }
        if *digest != turns[0].0 {
            return Err(Unmergeable::TokenizerDiffers { call });
        }
    }
    let sampled: Vec<&SampledTurn> = turns.iter().map(|(_, t)| *t).collect();
    match first_divergence(&sampled) {
        Some((call, position)) => Err(Unmergeable::NotAppendOnly { call, position }),
        None => Ok(()),
    }
}

/// The length of the sequence each sampled answer is trained in: its prompt
/// ids and its generated ids.
fn sequence(turn: &SampledTurn) -> u64 {
    (turn.prompt_token_ids.len() + turn.token_ids.len()) as u64
}

/// A message as two requests compare it: what was sent, not whether a
/// record supervises it.
fn as_sent(message: &WireMessage) -> WireMessage {
    WireMessage {
        train: false,
        loss_weight: None,
        ..message.clone()
    }
}

/// The first message where `earlier` (a call's request and answer) is not
/// the start of `later` (a later call's request); `None` when it is.
fn first_message_difference(earlier: &[WireMessage], later: &[WireMessage]) -> Option<usize> {
    earlier
        .iter()
        .zip(later)
        .position(|(a, b)| as_sent(a) != as_sent(b))
        .or_else(|| (later.len() < earlier.len()).then_some(later.len()))
}

/// One agent session as one training record: the last request and its
/// answer, every answer the session sampled supervised with its call's ids.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionRecord {
    /// The last call's request, then its answer; the answer of every call
    /// supervised where it stands.
    pub messages: Vec<WireMessage>,
    /// The tools every call offered.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    /// Each call's sampled answer, in call order: one per supervised message.
    pub sampled: SampledTurns,
    /// The oldest policy version that decoded a call of the session.
    pub policy_version: u64,
    /// The controls every call was drawn under.
    pub sampler: Sampler,
    /// The session stopped at a length limit.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// The ids the per-call records would forward, summed over the calls.
    pub per_call_tokens: u64,
}

impl SessionRecord {
    /// The session `projection` is, as one record, when it is exact and
    /// append-only.
    ///
    /// # Errors
    /// Why the session stays one record per call ([`Unmergeable`]).
    pub fn merge(projection: &CallProjection) -> Result<Self, Unmergeable> {
        if !projection.exact() {
            return Err(Unmergeable::NotExact);
        }
        Self::of_calls(&projection.records)
    }

    /// The session whose exact call records are `calls`, in request order,
    /// as one record.
    ///
    /// # Errors
    /// Why the session stays one record per call ([`Unmergeable`]).
    pub fn of_calls(calls: &[CallRecord]) -> Result<Self, Unmergeable> {
        let (Some(first), Some(last)) = (calls.first(), calls.last()) else {
            return Err(Unmergeable::NotExact);
        };
        let mut turns = Vec::with_capacity(calls.len());
        for (call, record) in calls.iter().enumerate() {
            let [Some(turn)] = record.sampled.turns.as_slice() else {
                return Err(Unmergeable::MissingIds { call });
            };
            if record.sampler != first.sampler {
                return Err(Unmergeable::SamplerDiffers { call });
            }
            if record.tools != last.tools {
                return Err(Unmergeable::ToolsDiffer { call });
            }
            turns.push((record.sampled.tokenizer_digest.as_str(), turn));
        }
        check_turns(&turns)?;
        for (call, pair) in calls.windows(2).enumerate() {
            let request = &pair[1].messages[..pair[1].messages.len() - 1];
            if let Some(message) = first_message_difference(&pair[0].messages, request) {
                return Err(Unmergeable::MessagesDiverge {
                    call: call + 1,
                    message,
                });
            }
        }
        // Each call's answer is the last of its messages, so it stands at
        // that index of every later request; it is supervised in place as
        // it was answered.
        let mut messages = last.messages.clone();
        for record in calls {
            let at = record.messages.len() - 1;
            messages[at] = record.messages[at].clone();
            messages[at].train = true;
        }
        Ok(Self {
            messages,
            tools: last.tools.clone(),
            sampled: SampledTurns {
                tokenizer_digest: first.sampled.tokenizer_digest.clone(),
                turns: turns.iter().map(|(_, t)| Some((*t).clone())).collect(),
            },
            policy_version: calls.iter().map(|c| c.policy_version).min().unwrap_or(0),
            sampler: first.sampler,
            truncated: calls.iter().any(|c| c.truncated),
            per_call_tokens: turns.iter().map(|(_, t)| sequence(t)).sum(),
        })
    }

    /// The ids the merged record forwards: the last request's prompt ids and
    /// its generated ids.
    #[must_use]
    pub fn tokens(&self) -> u64 {
        self.sampled
            .turns
            .last()
            .and_then(Option::as_ref)
            .map_or(0, sequence)
    }

    /// The same record, when its sequence fits a training block of `block`
    /// ids (any length when `None`).
    ///
    /// # Errors
    /// [`Unmergeable::LongerThanBlock`].
    pub fn within(self, block: Option<u64>) -> Result<Self, Unmergeable> {
        match block {
            Some(block) if self.tokens() > block => Err(Unmergeable::LongerThanBlock {
                tokens: self.tokens(),
                block,
            }),
            _ => Ok(self),
        }
    }

    /// The training record of the session once it earned `reward` in the
    /// group `group` as rollout `rollout` (keys as [`CallRecord::record`]
    /// takes them): brain's packed rollout record, to be trained with
    /// answers packed and a block holding [`Self::tokens`].
    #[must_use]
    pub fn record(&self, reward: f64, group: &str, rollout: &str) -> Record {
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
            metadata: RecordMetadata::of(SESSION_VIEW, Objective::PolicyGradient),
        }
    }
}

/// How the sessions of a dataset were written: merged into one record each,
/// or one record per call and why, with the ids each form forwards.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionReport {
    /// Sessions written as one record.
    pub merged: u64,
    /// Sessions written one record per call, by reason.
    pub per_call: BTreeMap<FallbackReason, u64>,
    /// The ids the merged sessions' records forward.
    pub merged_tokens: u64,
    /// The ids the same sessions' per-call records would have forwarded.
    pub per_call_tokens: u64,
}

impl SessionReport {
    /// Counts a merged session.
    pub fn merged(&mut self, session: &SessionRecord) {
        self.merged += 1;
        self.merged_tokens += session.tokens();
        self.per_call_tokens += session.per_call_tokens;
    }

    /// Counts a session written one record per call, for `why`.
    pub fn fell_back(&mut self, why: &Unmergeable) {
        *self.per_call.entry(why.reason()).or_insert(0) += 1;
    }

    /// Whether no session was counted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.merged == 0 && self.per_call.is_empty()
    }
}
