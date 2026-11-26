// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning data that teaches an agent what
// worked rather than every attempt it made, for its clients. If your team
// needs expertise in supervising agent trajectories, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which tool calls of a recorded conversation were answered with an error,
//! and the supervision that leaves the actions made only of such calls out
//! of the loss.
//!
//! The error status of a result is in neither record alone. The ATIF
//! trajectory keeps each tool's output as the tool returned it, with no
//! status; the request log keeps what the model was shown, which for a
//! result the tool flagged as an error (MCP `isError`) is that output marked
//! with Sven's prefix `error: ` (and possibly truncated, with a time-left line
//! or a repeat note after it). The marker alone is no evidence: an output may
//! itself begin with `error: ` (a command's own message) while the call
//! succeeded, and then the request shows it unmarked. So a result is an error
//! when what the model was shown is the trajectory's output with the marker
//! put in front of it: once the marker is taken off, the shown text agrees
//! with the output at least as far as it did with the marker left on.

use std::collections::{BTreeSet, HashMap};

use atif::{MessageBody, Trajectory};
use serde_json::Value;
use splinter_core::chat::WireMessage;

use crate::record::RecordBody;

/// The prefix Sven puts before a tool result flagged as an error.
pub(crate) const ERROR_MARKER: &str = "error: ";

/// Whether `shown`, a tool result as the model was sent it, is the tool's
/// own `output` marked as an error.
pub(crate) fn shown_as_error(shown: &str, output: &str) -> bool {
    let Some(rest) = shown.strip_prefix(ERROR_MARKER) else {
        return false;
    };
    common_prefix(rest, output) >= common_prefix(shown, output)
}

fn common_prefix(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

/// The calls of `chain` (a conversation built from `request`, the longest
/// request, with its call ids scoped `s<step>-<id>`) whose result the model
/// was shown as an error, by their scoped ids. A result the trajectory does
/// not hold, or holds as no text, is no evidence of an error.
pub(crate) fn failed_calls(
    trajectory: &Trajectory,
    request: &[Value],
    chain: &[WireMessage],
) -> BTreeSet<String> {
    // A result answers the latest call of its id: a runtime may file it on a
    // step of its own after the step that made the call, and a model may
    // number the calls of each response from zero.
    let mut caller: HashMap<&str, u64> = HashMap::new();
    let mut outputs: HashMap<(u64, &str), String> = HashMap::new();
    for step in &trajectory.steps {
        for call in step.tool_calls.iter().flatten() {
            caller.insert(call.tool_call_id.as_str(), step.step_id);
        }
        for result in step.observation.iter().flat_map(|o| &o.results) {
            let (Some(id), Some(body)) = (&result.source_call_id, &result.content) else {
                continue;
            };
            if let (Some(&step), Some(text)) = (caller.get(id.as_str()), text_of(body)) {
                outputs.insert((step, id.as_str()), text);
            }
        }
    }
    let shown = request
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("tool"))
        .filter_map(|m| {
            let content = m.get("content")?;
            let id = content.get("tool_call_id")?.as_str()?;
            Some((id, sent_text(content.get("content")?)?))
        });
    let scoped = chain
        .iter()
        .filter(|m| m.role == "tool")
        .filter_map(|m| m.tool_call_id.as_deref());
    // The conversation keeps every tool message of the request, in order.
    shown
        .zip(scoped)
        .filter_map(|((id, text), scoped)| {
            let step = scoped
                .strip_prefix('s')?
                .strip_suffix(id)?
                .strip_suffix('-')?
                .parse::<u64>()
                .ok()?;
            let output = outputs.get(&(step, id))?;
            shown_as_error(&text, output).then(|| scoped.to_owned())
        })
        .collect()
}

/// The text of a tool result as sent: a string, or the text parts of a list.
fn sent_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect(),
        ),
        _ => None,
    }
}

fn text_of(body: &MessageBody) -> Option<String> {
    match body {
        MessageBody::Text(text) => Some(text.clone()),
        MessageBody::Segments(segments) => Some(
            segments
                .iter()
                .filter_map(|s| match s {
                    atif::ContentSegment::Text { text } => Some(text.as_str()),
                    atif::ContentSegment::Image { .. } => None,
                })
                .collect(),
        ),
    }
}

/// Takes every supervised assistant action of `record` whose tool calls were
/// all answered with an error (`failed`, by call id) out of the loss: it
/// stays in the context, unsupervised. An action that recovers after an
/// error, any action with one call that succeeded or was not answered, and
/// the final answer stay supervised. Returns how many actions it took out.
pub fn mask_failed_actions(record: &mut RecordBody, failed: &BTreeSet<String>) -> usize {
    let RecordBody::Chat { messages, .. } = record else {
        return 0;
    };
    let mut masked = 0;
    for message in messages.iter_mut().filter(|m| {
        m.role == "assistant"
            && m.train
            && !m.tool_calls.is_empty()
            && m.tool_calls
                .iter()
                .all(|c| c.id.as_ref().is_some_and(|id| failed.contains(id)))
    }) {
        message.train = false;
        message.loss_weight = None;
        masked += 1;
    }
    masked
}

#[cfg(test)]
mod tests {
    use super::shown_as_error;

    #[test]
    fn the_marker_counts_only_when_it_was_put_before_the_output() {
        assert!(shown_as_error(
            "error: no such file\n\nTime: 1 min left",
            "no such file"
        ));
        assert!(shown_as_error("error: ", ""));
        // Truncated to its tail: the marker before text that is not the output's start.
        assert!(shown_as_error(
            "error: \n[... 3 lines omitted ...]\n",
            "[{\"long\": 1}]"
        ));
        assert!(!shown_as_error("no such file", "no such file"));
        // The output itself starts with the marker and was shown as it was.
        assert!(!shown_as_error(
            "error: bad flag\n\nTime: 1 min left",
            "error: bad flag"
        ));
        // ... and when that output is flagged as an error, it is marked again.
        assert!(shown_as_error("error: error: bad flag", "error: bad flag"));
    }
}
