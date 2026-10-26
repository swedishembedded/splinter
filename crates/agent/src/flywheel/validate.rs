// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-driven training-task generation that
// is judged by what the tasks do when run, for its clients. If your team
// needs expertise in synthetic task generation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The validate harness: one finished attempt, one focus, one typed decision.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sven_sdk::CancelToken;

use super::{copy_tree, io, Call, FlywheelError, Runner, SvenLaunch};

/// A validator's decision about one attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The attempt is worth training on.
    Keep,
    /// It is not.
    Discard,
}

/// A validator's typed decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decision {
    /// Keep or discard.
    pub decision: Verdict,
    /// What the decision rests on.
    #[serde(default)]
    pub evidence: String,
    /// Why, when it is a discard.
    #[serde(default)]
    pub rejection_reasons: Vec<String>,
}

/// The validate harness, run on one finished attempt at a time.
#[derive(Clone, Debug)]
pub struct Validate {
    runner: Runner,
    harness: PathBuf,
}

/// The focus of a validator: what it judges an attempt for.
pub const FOCI: [&str; 3] = ["outcome", "process", "teaching"];

/// What a validator is shown of one attempt.
#[derive(Clone, Copy, Debug)]
pub struct Attempt<'a> {
    /// The task as the solver read it.
    pub task: &'a str,
    /// The directory of the finished work.
    pub workspace: &'a Path,
    /// The hidden tests and what they printed on this work.
    pub tests: &'a str,
    /// The solver's conversation, rendered.
    pub trajectory: &'a str,
}

impl Validate {
    /// The validate harness `harness`.
    #[must_use]
    pub fn new(launch: SvenLaunch, harness: PathBuf) -> Self {
        Self {
            runner: Runner::new(launch),
            harness,
        }
    }

    /// Stops the run in progress, and every later one, once `cancel` is
    /// cancelled.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.runner.cancel = Some(cancel);
        self
    }

    /// The decision for `attempt` under `focus`.
    ///
    /// # Errors
    ///
    /// [`FlywheelError::NoDecision`] when the run ended without one (a
    /// failed validator is not a discard), and the errors of a run that
    /// cannot be made.
    pub async fn decide(
        &self,
        attempt: &Attempt<'_>,
        focus: &str,
        label: &str,
        deadline: Duration,
    ) -> Result<Decision, FlywheelError> {
        let work = tempfile::Builder::new()
            .prefix("validate-")
            .tempdir()
            .map_err(io("making the validator's workspace"))?;
        let ws = work.path();
        let write = |name: &str, text: &str| {
            std::fs::write(ws.join(name), text).map_err(io(format!("writing {name}")))
        };
        write("task.md", attempt.task)?;
        write("tests.md", attempt.tests)?;
        write("trajectory.md", attempt.trajectory)?;
        copy_tree(attempt.workspace, &ws.join("workspace"), &[".git"])?;
        let finished = self
            .runner
            .run(&Call {
                harness: &self.harness,
                workspace: ws,
                inputs: &json!({ "focus": focus }),
                deadline,
                label,
            })
            .await?;
        let record = &finished.record;
        let how = || {
            format!(
                "{}, status {}, conclusion {}",
                record.killed.as_deref().unwrap_or("ended"),
                record.status.as_deref().unwrap_or("unknown"),
                record.conclusion.as_deref().unwrap_or("unknown")
            )
        };
        let decision = finished
            .report
            .as_ref()
            .and_then(|r| r.pointer("/output/decision"))
            .ok_or_else(|| FlywheelError::NoDecision(how()))?;
        serde_json::from_value(decision.clone())
            .map_err(|e| FlywheelError::NoDecision(format!("{e}; {}", how())))
    }
}

/// A JSON value as the text it carries: a string as is, anything else as JSON.
fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Conversations (ATIF trajectories) as the validator reads them: one
/// section per conversation, each step's message, tool calls and results.
#[must_use]
pub fn conversations_text(trajectories: &[Value]) -> String {
    /// The most characters of one tool result shown.
    const RESULT_CHARS: usize = 4000;
    let mut out = Vec::new();
    for (number, atif) in trajectories.iter().enumerate() {
        out.push(format!(
            "## Conversation {} of {}\n",
            number + 1,
            trajectories.len()
        ));
        for step in atif
            .get("steps")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let source = step.get("source").and_then(Value::as_str).unwrap_or("?");
            if let Some(message) = step
                .get("message")
                .map(text_of)
                .filter(|m| !m.trim().is_empty())
            {
                out.push(format!("### {source}\n\n{}\n", message.trim()));
            }
            for call in step
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                out.push(format!(
                    "**tool call** `{}`\n```json\n{}\n```\n",
                    call.get("function_name")
                        .and_then(Value::as_str)
                        .unwrap_or("?"),
                    serde_json::to_string_pretty(call.get("arguments").unwrap_or(&Value::Null))
                        .unwrap_or_default()
                ));
            }
            for result in step
                .pointer("/observation/results")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let content = result.get("content").map(text_of).unwrap_or_default();
                out.push(format!(
                    "**tool result**\n```\n{}\n```\n",
                    content.chars().take(RESULT_CHARS).collect::<String>()
                ));
            }
        }
    }
    out.join("\n")
}
