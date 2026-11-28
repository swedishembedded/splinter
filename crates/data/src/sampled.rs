// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What the policy actually sampled, carried to the trainer.
//!
//! sven records, on each agent step, the ids the sampler was given, the ids it
//! generated and one log-probability per generated id (ATIF `step.metrics`),
//! with the tokenizer's digest and the kind of the log-probabilities in
//! `metrics.extra.sven`. A rewarded record that carries them
//! ([`SampledTurns`]) lets brain train on the sampled ids themselves and weigh
//! them against the behaviour policy's own log-probabilities, instead of
//! re-rendering the text (which can tokenise differently) and approximating the
//! behaviour policy by the start of the run. An action whose step has no usable
//! trace is still written, without per-token fields, and counted under why
//! ([`ProxyReason`], [`TokenReport`]); a trace is never partly attached.

use std::collections::BTreeMap;

use atif::{StepOrigin, Trajectory};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The `logprob_kind` sven records for the policy's own full-distribution
/// figures: the only kind a policy gradient may weigh by.
pub(crate) const POLICY_KIND: &str = "policy";

/// One supervised assistant message's sampled tokens, as brain's training
/// record reads them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampledTurn {
    /// The ids the sampler was given: the context the answer was made from.
    pub prompt_token_ids: Vec<u32>,
    /// The ids it generated, as the sampler produced them: the end-of-turn
    /// marker is among them when it stopped on one (a generation cut at its
    /// length limit has none, and the tokenizer is not known here to check).
    pub token_ids: Vec<u32>,
    /// The policy's log-probability of each generated id, one per id.
    pub behavior_logprobs: Vec<f32>,
    /// The sampler's `K` most probable ids at each generated position, one
    /// list per id, when the run asked for them (brain's `behavior_top_ids`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_top_ids: Option<Vec<Vec<u32>>>,
    /// Their whole-row log-probabilities at the sampling temperature, beside
    /// `behavior_top_ids` (brain's `behavior_top_logprobs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_top_logprobs: Option<Vec<Vec<f32>>>,
}

/// The sampled tokens of a record: its tokenizer and, for each supervised
/// message in order, its turn when it has one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampledTurns {
    /// The tokenizer the ids are of (`sha256:<hex>`); every turn of a record
    /// agrees on it.
    pub tokenizer_digest: String,
    /// One entry per supervised message of the record, in order.
    pub turns: Vec<Option<SampledTurn>>,
}

/// Why an action carries no sampled tokens and is trained against the
/// start-of-run proxy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProxyReason {
    /// Its step recorded no ids, log-probabilities or tokenizer.
    NoTrace,
    /// The log-probabilities are the sampler's (after top-k/top-p), not the
    /// policy's.
    SamplerLogprobs,
    /// A log-probability per generated id was not recorded, or there is no
    /// generated id.
    CountMismatch,
    /// An id is not a token id (negative or beyond 32 bits).
    InvalidIds,
    /// A log-probability is not a finite number at most zero.
    InvalidLogprobs,
    /// The scrub changed text of the conversation, which the ids decode to
    /// unscrubbed.
    Scrubbed,
    /// The step's ids are of another tokenizer than its record's others.
    MixedTokenizer,
    /// One recorded step became several actions, so the ids belong to none of
    /// them alone.
    SplitStep,
    /// The record's system prompt was replaced after it was projected
    /// ([`crate::Projection::with_system_prompt`]): the ids are of the prompt
    /// the policy was sent, which the record no longer holds.
    PromptReplaced,
    /// The conversation was rebuilt from the trajectory, not read from the
    /// requests the model was sent, so its context is not the one the ids
    /// were sampled after.
    ContextReconstructed,
    /// The log-probabilities are at another temperature than the sampling
    /// temperature, or name none.
    TemperatureMismatch,
    /// The request names no policy version (or one that is not a whole
    /// number): its staleness cannot be bounded.
    NoPolicyVersion,
    /// The sampler's top-K is malformed: one list without the other, a row
    /// count other than the generated ids', rows of different `K`, an id
    /// twice in a row, or a log-probability that is not one.
    TopMismatch,
    /// The requests the trajectory recorded and the request log's calls are
    /// not one per call, so no call can be paired with its context.
    Unpaired,
    /// The request as logged cannot be read as chat messages.
    UnreadableRequest,
    /// The answer says nothing and calls nothing.
    EmptyAnswer,
    /// The answer's tool-call arguments are text that is not JSON: the call
    /// is context as the model wrote it, never a supervised action.
    UnparsedArguments,
    /// The server's record of what it sampled (`extra.sven.sampled`) names
    /// other ids than the step's trace, or cannot be read.
    SampledDiffers,
}

/// How many rewarded actions carry the tokens the policy sampled, and why the
/// others do not.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenReport {
    /// Actions written with their exact sampled tokens.
    pub exact: usize,
    /// Actions written without, by reason.
    pub proxy: BTreeMap<ProxyReason, usize>,
}

impl TokenReport {
    /// Whether no action was counted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.exact == 0 && self.proxy.is_empty()
    }

    pub(crate) fn count(&mut self, outcome: &Result<(String, SampledTurn), ProxyReason>) {
        match outcome {
            Ok(_) => self.exact += 1,
            Err(reason) => *self.proxy.entry(*reason).or_insert(0) += 1,
        }
    }
}

/// The tokenizer digest and sampled turn of the agent step `step_id` of
/// `trajectory`, or why it has none usable.
pub(crate) fn trace_of(
    trajectory: &Trajectory,
    step_id: u64,
) -> Result<(String, SampledTurn), ProxyReason> {
    let mut steps = trajectory
        .steps
        .iter()
        .filter(|s| s.source == StepOrigin::Agent && s.step_id == step_id);
    let step = steps.next().ok_or(ProxyReason::NoTrace)?;
    let metrics = step.metrics.as_ref().ok_or(ProxyReason::NoTrace)?;
    let sven = |key: &str| metrics.extra.as_ref()?.pointer(&format!("/sven/{key}"));
    let (Some((prompt, completion)), Some(logprobs)) = (
        trained_ids(
            sampled_record(step),
            metrics.prompt_token_ids.as_deref(),
            metrics.completion_token_ids.as_deref(),
        )?,
        &metrics.logprobs,
    ) else {
        return Err(ProxyReason::NoTrace);
    };
    let digest = sven("tokenizer_digest")
        .and_then(|d| d.as_str())
        .ok_or(ProxyReason::NoTrace)?;
    match sven("logprob_kind").and_then(|k| k.as_str()) {
        Some(POLICY_KIND) => {}
        Some(_) => return Err(ProxyReason::SamplerLogprobs),
        None => return Err(ProxyReason::NoTrace),
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
    Ok((
        digest.to_string(),
        SampledTurn {
            prompt_token_ids: prompt,
            token_ids: completion,
            // The trainer reads f32; the policy's own figures are f32 in sven.
            #[allow(clippy::cast_possible_truncation)]
            behavior_logprobs: logprobs.iter().map(|l| *l as f32).collect(),
            behavior_top_ids: None,
            behavior_top_logprobs: None,
        },
    ))
}

/// The traces of the actions of one record, in order, made consistent: an
/// action whose ids are of another tokenizer than the first usable one's is a
/// proxy ([`ProxyReason::MixedTokenizer`]).
pub(crate) fn agreeing(
    traces: Vec<Result<(String, SampledTurn), ProxyReason>>,
) -> Vec<Result<(String, SampledTurn), ProxyReason>> {
    let first = traces
        .iter()
        .find_map(|t| t.as_ref().ok().map(|(digest, _)| digest.clone()));
    traces
        .into_iter()
        .map(|trace| match (&first, trace) {
            (Some(first), Ok((digest, _))) if digest != *first => Err(ProxyReason::MixedTokenizer),
            (_, trace) => trace,
        })
        .collect()
}

/// The sampled tokens of a record made of the actions whose traces are
/// `traces`, or `None` when none has any.
pub(crate) fn turns_of(
    traces: Vec<Result<(String, SampledTurn), ProxyReason>>,
) -> Option<SampledTurns> {
    let digest = traces
        .iter()
        .find_map(|t| t.as_ref().ok().map(|(d, _)| d.clone()))?;
    Some(SampledTurns {
        tokenizer_digest: digest,
        turns: traces
            .into_iter()
            .map(|t| t.ok().map(|(_, turn)| turn))
            .collect(),
    })
}

/// A step's prompt ids and generated ids.
pub(crate) type StepIds = (Vec<u32>, Vec<u32>);

/// The server's record of what it sampled for an agent step's reply, as sven
/// keeps it: `extra.sven.sampled` of the step (beside its metrics), the
/// exact text the server sent.
pub(crate) fn sampled_record(step: &atif::TraceStep) -> Option<&Value> {
    step.extra.as_ref()?.get("sven")?.get("sampled")
}

/// Ids as a trace records them, each a token id.
pub(crate) fn token_ids(ids: &[i64]) -> Result<Vec<u32>, ProxyReason> {
    ids.iter()
        .map(|id| u32::try_from(*id).map_err(|_| ProxyReason::InvalidIds))
        .collect()
}

/// The prompt and generated ids an agent step trains: its trace's `prompt`
/// and `completion` ids, when the server's record of what it sampled
/// (`record`, [`sampled_record`]) names them too, or the step kept no
/// record. `None` when the step traced no ids.
///
/// The record names the generated ids (`token_ids`) and the prompt by its
/// length (`context_len`), its digest (`context_hash`, the sha256 of the ids
/// as little-endian u32) and the generation header it ends with
/// (`prefill_token_ids`).
///
/// # Errors
/// [`ProxyReason::SampledDiffers`] when the record cannot be read or names
/// other ids than the trace (whose log-probabilities belong to the trace's
/// ids); [`ProxyReason::InvalidIds`] when a trace id is not a token id.
pub(crate) fn trained_ids(
    record: Option<&Value>,
    prompt: Option<&[i64]>,
    completion: Option<&[i64]>,
) -> Result<Option<StepIds>, ProxyReason> {
    let (Some(prompt), Some(completion)) = (prompt, completion) else {
        return Ok(None);
    };
    let ids = (token_ids(prompt)?, token_ids(completion)?);
    let Some(record) = record else {
        return Ok(Some(ids));
    };
    let record: Value = match record {
        Value::String(text) => {
            serde_json::from_str(text).map_err(|_| ProxyReason::SampledDiffers)?
        }
        other => other.clone(),
    };
    if names(&record, &ids).unwrap_or(false) {
        Ok(Some(ids))
    } else {
        Err(ProxyReason::SampledDiffers)
    }
}

/// Whether the server's sampled `record` names the step's `(prompt,
/// generated)` ids; `None` when it lacks a field.
fn names(record: &Value, (prompt, generated): &StepIds) -> Option<bool> {
    use sha2::Digest as _;
    let ids = |key: &str| decode_ids(record.get(key)?.as_str()?).ok();
    let mut hasher = sha2::Sha256::new();
    for id in prompt {
        hasher.update(id.to_le_bytes());
    }
    let hash = format!("sha256:{:x}", hasher.finalize());
    Some(
        ids("token_ids")? == *generated
            && prompt.ends_with(&ids("prefill_token_ids")?)
            && record.get("context_len")?.as_u64()? == prompt.len() as u64
            && record.get("context_hash")?.as_str()? == hash,
    )
}

/// The ids of a reply's `sampled` record (a brain server's `token_ids` and
/// `prefill_token_ids`): each id an unsigned LEB128 varint, the run
/// base64-encoded. The generated ids are the ones the server splices into
/// every later prompt.
///
/// # Errors
/// The text is not base64, a varint is cut off or longer than an id.
pub fn decode_ids(text: &str) -> Result<Vec<u32>, String> {
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(text)
        .map_err(|e| format!("sampled ids: {e}"))?;
    let mut ids = Vec::with_capacity(bytes.len() / 2);
    let (mut value, mut shift) = (0u64, 0u32);
    for b in bytes {
        value |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            ids.push(u32::try_from(value).map_err(|_| "sampled ids: an id past u32".to_string())?);
            (value, shift) = (0, 0);
        } else {
            shift += 7;
            if shift > 28 {
                return Err("sampled ids: a varint longer than five bytes".to_string());
            }
        }
    }
    if shift != 0 {
        return Err("sampled ids: the text ends inside a varint".to_string());
    }
    Ok(ids)
}
