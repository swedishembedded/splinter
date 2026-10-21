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
use splinter_sdk::agent::sven::atif::{StepOrigin, Trajectory};
use splinter_sdk::vocabulary::tool_calls::{body_text, tool_calls};

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

impl Trace {
    /// The trace of `trajectory`.
    #[must_use]
    pub fn of(trajectory: &Trajectory) -> Self {
        let said = trajectory
            .steps
            .iter()
            .filter(|s| s.source == StepOrigin::Agent)
            .map(|s| body_text(&s.message))
            .filter(|said| !said.trim().is_empty())
            .collect();
        let calls = tool_calls(trajectory)
            .into_iter()
            .map(|call| Call {
                step: call.step,
                tool: call.tool,
                args: call.arguments,
                result: call.output,
            })
            .collect();
        Self { calls, said }
    }
}
