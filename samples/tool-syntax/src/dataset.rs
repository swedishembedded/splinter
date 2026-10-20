// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Turning a verified performance into training data.
//!
//! The output is `generic-messages-v2`: one packed conversation per line,
//! `train` per message deciding which spans are supervised. brain's parser is
//! strict about it, and this module is strict about the things the parser
//! cannot see.
//!
//! # What this refuses to emit, and why
//!
//! **An episode that was not verified.** Training on an unverified trajectory
//! teaches whatever the model happened to do, which on a task it fails is
//! precisely the behaviour being trained out of it. Only a [`Verdict`] that
//! says solved gets through, and a [`Verdict`] can only come from evaluating
//! a task's declared predicates.
//!
//! **Context as a supervised span.** Only the assistant's decision is trained on.
//! Supervising a tool result teaches the model to produce tool output itself -
//! to hallucinate the answer it was supposed to go and fetch, which is again
//! the exact failure the task exists to correct.
//!
//! # Provenance is not decoration
//!
//! Every record carries how it was produced. A trajectory a stronger model
//! generated, or one a script drove, is excellent supervised data and is *not*
//! an on-policy rollout; mixing them and forgetting which was which is how a
//! later claim about self-improvement quietly stops being true. The metadata
//! travels with the record so the question stays answerable afterwards.

use serde::{Deserialize, Serialize};
use splinter_sdk::vocabulary::chat::{WireFunction, WireMessage, WireToolCall};

use crate::Verdict;

/// How a trajectory came to exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// The model being trained produced it.
    OnPolicy,
    /// A different, usually stronger, model produced it.
    Teacher,
    /// A deterministic solver drove it. Verified and useful, but it
    /// demonstrates one path rather than a policy.
    Scripted,
}

/// Why an episode produced no training record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Excluded {
    /// The verifier did not accept the work.
    NotSolved {
        /// The declared predicates that did not hold.
        unmet: Vec<String>,
    },
    /// The transcript contains a message this harness could not represent, so
    /// any example drawn from it would be missing context the model saw.
    OpaqueTurn {
        /// Role of the message that could not be represented; `(none)` when
        /// a captured request message carried no role at all.
        role: String,
    },
    /// Nothing in the episode would be supervised.
    NothingToLearn,
}

impl std::fmt::Display for Excluded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Excluded::NotSolved { unmet } => {
                write!(
                    f,
                    "the verifier did not accept it; unmet: {}",
                    unmet.join(", ")
                )
            }
            Excluded::OpaqueTurn { role } => write!(
                f,
                "the transcript contains a {role} message this harness cannot represent, so an \
                 example drawn from it would be missing a step the model actually saw"
            ),
            Excluded::NothingToLearn => f.write_str(
                "no assistant turn would be supervised, so the record would train on nothing",
            ),
        }
    }
}

/// One packed conversation, ready to serialise as a JSONL line.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    /// The conversation in order; only assistant messages carry
    /// `train: true`.
    pub messages: Vec<WireMessage>,
    /// Tool schemas exactly as the captured request sent them, so the chat
    /// template renders the same tools preamble at training time as at
    /// inference. Empty for a record derived from stored history, which
    /// does not keep them.
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    /// How the record was produced; never rendered into the prompt.
    pub metadata: RecordMetadata,
}

/// Where a record came from: the family it belongs to, who produced the
/// trajectory, and what was hidden while it ran.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordMetadata {
    /// The task family. Splits are keyed on this, never on the episode: two
    /// episodes of one family share a workspace, a request and a solution
    /// shape, so splitting by episode leaks near-duplicate context across
    /// train and test and reports memorisation as generalisation.
    pub family: String,
    /// Who produced the trajectory, so on-policy rollouts stay separable
    /// from teacher and scripted demonstrations.
    pub provenance: Provenance,
    /// What was hidden for this episode. Recorded after the fact, for
    /// auditing the mix - never shown to a model.
    pub hidden_state: String,
}

/// A message the model was shown but is not trained on.
///
/// Everything that is not an assistant decision is context: the system
/// prompt, the request, and every tool result. Supervising any of them would
/// teach the model to produce them.
fn context(role: &str, text: &str) -> WireMessage {
    WireMessage {
        role: role.to_string(),
        content: text.to_string(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        train: false,
    }
}

/// Serialise records as JSONL, one packed conversation per line.
pub fn to_jsonl(records: &[Record]) -> Result<String, serde_json::Error> {
    let mut out = String::new();
    for record in records {
        out.push_str(&serde_json::to_string(record)?);
        out.push('\n');
    }
    Ok(out)
}

/// Build the training record of one verified performance: the whole episode as
/// one packed conversation, every assistant decision supervised and every
/// other message context.
///
/// The conversation is the one the served model is asked to continue at each
/// step: system prompt, request, then for each action an assistant turn
/// carrying exactly one tool call (ids `call_0`, `call_1`, ... unique within
/// the conversation) followed by the tool message that answers that call and
/// no other, and finally the closing answer. brain's chat fine-tuner encodes
/// such a record one example per supervised answer, each rendered the way
/// the serving renderer renders the same messages when the model is asked for
/// that answer (earlier assistant turns as context, the supervised one last),
/// so no decision is trained against context the model never sees.
///
/// The system prompt and tool schemas come from a request the agent really
/// sent, so a record renders at training time the way the prompt renders at
/// inference. The observations come from the real tool executor. The
/// decisions, and only the decisions, are the demonstration's.
pub fn records_from_performance(
    family: &str,
    hidden_state: &str,
    prompt: &serde_json::Value,
    request: &str,
    performed: &[(String, String, String)],
    closing: &str,
    verdict: &Verdict,
) -> Result<Vec<Record>, Excluded> {
    if !verdict.solved() {
        return Err(Excluded::NotSolved {
            unmet: verdict.failed().to_vec(),
        });
    }
    if performed.is_empty() {
        return Err(Excluded::NothingToLearn);
    }

    let system = prompt
        .pointer("/messages/0/content")
        .and_then(|c| c.as_str())
        .ok_or(Excluded::OpaqueTurn {
            role: "system".into(),
        })?;
    let tools = prompt
        .get("tools")
        .and_then(|t| t.as_array())
        .cloned()
        .unwrap_or_default();

    let mut messages = vec![context("system", system), context("user", request)];
    for (index, (tool, arguments, observation)) in performed.iter().enumerate() {
        let id = format!("call_{index}");
        messages.push(WireMessage {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: vec![WireToolCall {
                id: Some(id.clone()),
                kind: "function".into(),
                function: WireFunction {
                    name: tool.clone(),
                    arguments: arguments.clone(),
                },
            }],
            tool_call_id: None,
            train: true,
        });
        messages.push(WireMessage {
            tool_call_id: Some(id),
            ..context("tool", observation)
        });
    }
    messages.push(WireMessage {
        train: true,
        ..context("assistant", closing)
    });

    Ok(vec![Record {
        messages,
        tools,
        metadata: RecordMetadata {
            family: family.to_string(),
            provenance: Provenance::Scripted,
            hidden_state: hidden_state.to_string(),
        },
    }])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::PredicateSet;

    fn verdict(pass: bool) -> Verdict {
        let set = PredicateSet::new(["done"]).expect("one predicate");
        let observed: BTreeMap<String, bool> = [("done".to_string(), pass)].into_iter().collect();
        set.evaluate(&observed).expect("evaluated")
    }

    fn prompt() -> serde_json::Value {
        serde_json::json!({
            "messages": [{"role": "system", "content": "you are an agent"}],
            "tools": [{"type": "function", "function": {"name": "shell"}}],
        })
    }

    fn performed() -> Vec<(String, String, String)> {
        vec![
            (
                "shell".into(),
                r#"{"command":"./svctl status"}"#.into(),
                "active deployment: staging".into(),
            ),
            (
                "read_file".into(),
                r#"{"path":"config/staging.json"}"#.into(),
                "L1: {}".into(),
            ),
        ]
    }

    fn records(verdict: &Verdict) -> Result<Vec<Record>, Excluded> {
        records_from_performance(
            "f",
            "staging",
            &prompt(),
            "enable retries",
            &performed(),
            "done",
            verdict,
        )
    }

    #[test]
    fn every_decision_and_only_decisions_are_supervised() {
        // Supervising context teaches the model to produce it - the system
        // prompt, the request, a tool result - instead of acting; leaving a
        // decision unsupervised teaches nothing about it.
        let records = records(&verdict(true)).expect("a solved performance yields records");
        assert_eq!(records.len(), 1);
        let roles: Vec<&str> = records[0]
            .messages
            .iter()
            .map(|m| m.role.as_str())
            .collect();
        assert_eq!(
            roles,
            [
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant"
            ]
        );
        for message in &records[0].messages {
            assert_eq!(
                message.train,
                message.role == "assistant",
                "only the decisions may be trained on, got {message:?}"
            );
        }
        assert_eq!(records[0].tools.len(), 1, "the prompt's tools travel along");
    }

    #[test]
    fn each_tool_result_answers_exactly_the_call_before_it() {
        // The invariant the corrupted CE4 data broke: unique call ids, and a
        // result that answers one earlier call and no other.
        let records = records(&verdict(true)).expect("records");
        let messages = &records[0].messages;
        let mut ids = Vec::new();
        for (at, message) in messages.iter().enumerate() {
            for call in &message.tool_calls {
                ids.push(call.id.clone().expect("a call id"));
            }
            if message.role == "tool" {
                let before = &messages[at - 1];
                assert_eq!(before.tool_calls.len(), 1, "one call per decision");
                assert_eq!(message.tool_call_id, before.tool_calls[0].id);
            }
        }
        let unique: std::collections::BTreeSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "call ids repeat: {ids:?}");
    }

    #[test]
    fn an_unverified_performance_yields_nothing() {
        let err = records(&verdict(false)).expect_err("an unsolved task must not become data");
        assert!(matches!(err, Excluded::NotSolved { .. }), "{err}");
    }

    #[test]
    fn tool_arguments_survive_verbatim() {
        // Re-serialising would normalise key order and spacing away from what
        // the model is meant to emit.
        let records = records(&verdict(true)).expect("records");
        assert_eq!(
            records[0].messages[2].tool_calls[0].function.arguments,
            r#"{"command":"./svctl status"}"#
        );
    }

    #[test]
    fn a_record_serialises_as_one_jsonl_line_with_train_on_every_message() {
        let jsonl = to_jsonl(&records(&verdict(true)).expect("records")).expect("serialises");
        assert_eq!(jsonl.lines().count(), 1);
        let parsed: serde_json::Value = serde_json::from_str(jsonl.trim()).expect("valid json");
        for message in parsed["messages"].as_array().expect("messages") {
            assert!(
                message.get("train").is_some(),
                "every message needs train: {message}"
            );
        }
    }
}
