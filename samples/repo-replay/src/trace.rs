// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What an agent did, as the gate reads it: the tool calls with their
//! arguments and results, in order, and what it said between them.
//!
//! It is read from sven's trajectory (the authoritative record of the run),
//! never from anything the agent reports about itself.

use serde_json::Value;
use splinter_sdk::agent::sven::atif::{MessageBody, StepOrigin, Trajectory};

/// One tool call and what it returned.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    /// The trajectory step that made it.
    pub step: u64,
    /// The tool's name (`read_file`, `edit_file`, `shell`, ...).
    pub tool: String,
    /// The arguments the model sent.
    pub args: Value,
    /// What the tool returned, as text; empty when no result was recorded.
    pub result: String,
}

impl Call {
    /// A string argument.
    #[must_use]
    pub fn arg(&self, name: &str) -> Option<&str> {
        self.args.get(name).and_then(Value::as_str)
    }

    /// The command line of a shell call.
    #[must_use]
    pub fn command(&self) -> Option<&str> {
        self.arg("shell_command").or_else(|| self.arg("command"))
    }

    /// Every string value of the arguments joined, where text can hide.
    #[must_use]
    pub fn args_text(&self) -> String {
        fn collect(v: &Value, out: &mut Vec<String>) {
            match v {
                Value::String(s) => out.push(s.clone()),
                Value::Array(a) => a.iter().for_each(|x| collect(x, out)),
                Value::Object(o) => o.values().for_each(|x| collect(x, out)),
                _ => {}
            }
        }
        let mut parts = Vec::new();
        collect(&self.args, &mut parts);
        parts.join("\n")
    }
}

/// The calls of a run and the text the agent said.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    /// Every tool call, in order.
    pub calls: Vec<Call>,
    /// The agent's own words, one entry per agent step that had any.
    pub said: Vec<String>,
}

fn text_of(body: &MessageBody) -> String {
    match body {
        MessageBody::Text(t) => t.clone(),
        MessageBody::Segments(_) => String::new(),
    }
}

impl Trace {
    /// The trace of `trajectory`.
    #[must_use]
    pub fn of(trajectory: &Trajectory) -> Self {
        let mut trace = Self::default();
        for step in trajectory
            .steps
            .iter()
            .filter(|s| s.source == StepOrigin::Agent)
        {
            let said = text_of(&step.message);
            if !said.trim().is_empty() {
                trace.said.push(said);
            }
            for call in step.tool_calls.iter().flatten() {
                let result = step
                    .observation
                    .iter()
                    .flat_map(|o| &o.results)
                    .find(|r| r.source_call_id.as_deref() == Some(&call.tool_call_id))
                    .and_then(|r| r.content.as_ref())
                    .map(text_of)
                    .unwrap_or_default();
                trace.calls.push(Call {
                    step: step.step_id,
                    tool: call.function_name.clone(),
                    args: call.arguments.clone(),
                    result,
                });
            }
        }
        trace
    }
}
