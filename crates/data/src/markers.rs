// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-training pipelines that train only on
// what an agent really saw and said, for its clients. If your team needs
// expertise in turning agent trajectories into fine-tuning data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a trajectory says about itself that decides whether it can be a
//! training conversation: the markers sven's trajectory export places on
//! a step, and the form of the tool schemas it records.
//!
//! * A system step carrying `extra.context_management` of a compaction
//!   (or any transformation that replaces or truncates the history) means
//!   the steps before it are not what the agent saw afterwards.
//! * A tool call carrying `extra.raw_arguments`, or flagged
//!   `extra.unparsed_arguments`, was sent by the model as text that is not
//!   JSON; its recorded `arguments` are a stand-in.
//! * `agent.tool_definitions` are OpenAI function-calling entries; a
//!   record's `tools` are the same entries in brain's form.

use atif::{ContextManagement, StepOrigin, ToolInvocation, TraceStep, Trajectory};
use serde_json::Value;

use crate::trajectory::Unprojectable;

/// The key of a tool call's `extra` holding the argument text the model
/// sent when it was not valid JSON.
const RAW_ARGUMENTS_KEY: &str = "raw_arguments";

/// The key of a tool call's `extra` flagging arguments that did not parse.
const UNPARSED_ARGUMENTS_KEY: &str = "unparsed_arguments";

/// Whether `step` marks the agent's context being compacted: the history
/// before it was replaced or cut, so the conversation the agent acted in
/// after it is not the steps that lead the trajectory.
pub(crate) fn is_compaction(step: &TraceStep) -> bool {
    step.source == StepOrigin::System
        && step
            .extra
            .as_ref()
            .and_then(ContextManagement::from_extra)
            .is_some_and(|c| {
                matches!(c.kind.as_str(), "compaction" | "pruning")
                    || matches!(c.boundary.as_str(), "replace" | "truncate")
            })
}

/// Whether `step` is a system step that is the harness's prompt, not a
/// marker of something done to the context.
pub(crate) fn is_harness_prompt(step: &TraceStep) -> bool {
    step.source == StepOrigin::System
        && step
            .extra
            .as_ref()
            .and_then(ContextManagement::from_extra)
            .is_none()
}

/// Whether `call`'s arguments are a stand-in for text the model sent that
/// was not JSON.
pub(crate) fn has_unparsed_arguments(call: &ToolInvocation) -> bool {
    call.extra.as_ref().is_some_and(|extra| {
        extra.get(RAW_ARGUMENTS_KEY).is_some()
            || extra.get(UNPARSED_ARGUMENTS_KEY).and_then(Value::as_bool) == Some(true)
    })
}

/// The tool schemas `trajectory` records, as a chat record carries them:
/// each an OpenAI function-calling entry. An entry already in that form is
/// kept as it is; a bare `{name, description, parameters}` function schema
/// is wrapped; anything else is refused, naming its index.
pub(crate) fn chat_tools(trajectory: &Trajectory) -> Result<Vec<Value>, Unprojectable> {
    trajectory
        .agent
        .tool_definitions
        .iter()
        .flatten()
        .enumerate()
        .map(|(index, definition)| {
            let object = definition
                .as_object()
                .ok_or(Unprojectable::ToolDefinition { index })?;
            let wrapped = object.get("type").and_then(Value::as_str) == Some("function")
                && object.get("function").is_some_and(Value::is_object);
            if wrapped {
                Ok(definition.clone())
            } else if object.get("name").is_some_and(Value::is_string) {
                Ok(serde_json::json!({"type": "function", "function": definition}))
            } else {
                Err(Unprojectable::ToolDefinition { index })
            }
        })
        .collect()
}
