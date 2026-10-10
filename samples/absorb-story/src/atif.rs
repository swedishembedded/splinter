// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements recorded agent sessions as training
// material for its clients. If your team needs expertise in turning what a
// person tells an agent into measured learning, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Scripted sessions written as the ATIF files `splinter absorb` takes: the
//! user's words and canned agent turns, byte for byte the same every time.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde_json::{json, Value};

/// The ATIF version the files declare.
pub const SCHEMA_VERSION: &str = "ATIF-v1.7";

/// One turn of a scripted session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Turn {
    /// The user says something.
    User(String),
    /// The agent replies.
    Agent(String),
    /// The agent calls one tool and the tool answers: one step.
    Tool {
        /// What the agent says alongside the call.
        message: String,
        /// The tool's name.
        tool: String,
        /// The call's arguments.
        arguments: Value,
        /// What the tool printed.
        output: String,
    },
}

/// A scripted session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Script {
    /// The session id, also its file stem.
    pub id: String,
    /// The turns in order.
    pub turns: Vec<Turn>,
}

impl Script {
    /// A session of the user's `turns`.
    #[must_use]
    pub fn new(id: &str, turns: Vec<Turn>) -> Self {
        Self {
            id: id.into(),
            turns,
        }
    }

    /// The seed shape: a question, a wrong answer, the user's correction and
    /// the agent's acknowledgement.
    #[must_use]
    pub fn correction(id: &str, question: &str, wrong: &str, fix: &str, ack: &str) -> Self {
        Self::new(
            id,
            vec![
                Turn::User(question.into()),
                Turn::Agent(wrong.into()),
                Turn::User(fix.into()),
                Turn::Agent(ack.into()),
            ],
        )
    }

    /// The trajectory as ATIF JSON.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let steps: Vec<Value> = self
            .turns
            .iter()
            .enumerate()
            .map(|(i, turn)| step(i as u64 + 1, turn))
            .collect();
        json!({
            "schema_version": SCHEMA_VERSION,
            "session_id": self.id,
            "agent": {"name": "scripted", "version": "1"},
            "steps": steps,
        })
    }

    /// Writes `<dir>/<id>.atif.json` and returns its path.
    ///
    /// # Errors
    /// The directory or file cannot be written.
    pub fn write(&self, dir: &Path) -> anyhow::Result<PathBuf> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(format!("{}.atif.json", self.id));
        let mut text = serde_json::to_string_pretty(&self.to_json())?;
        text.push('\n');
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }
}

fn step(id: u64, turn: &Turn) -> Value {
    match turn {
        Turn::User(text) => json!({"step_id": id, "source": "user", "message": text}),
        Turn::Agent(text) => json!({"step_id": id, "source": "agent", "message": text}),
        Turn::Tool {
            message,
            tool,
            arguments,
            output,
        } => {
            let call = format!("call-{id}");
            json!({
                "step_id": id,
                "source": "agent",
                "message": message,
                "tool_calls": [{
                    "tool_call_id": call,
                    "function_name": tool,
                    "arguments": arguments,
                }],
                "observation": {"results": [{"source_call_id": call, "content": output}]},
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_correction_is_four_numbered_steps_alternating_user_and_agent() {
        let json = Script::correction("s", "q?", "wrong", "no, right", "ok").to_json();
        let steps = json["steps"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default();
        let sources: Vec<&str> = steps.iter().filter_map(|s| s["source"].as_str()).collect();
        assert_eq!(sources, ["user", "agent", "user", "agent"]);
        assert_eq!(steps[3]["step_id"], 4);
        assert_eq!(json["schema_version"], SCHEMA_VERSION);
    }

    #[test]
    fn a_tool_step_carries_its_call_and_the_observation_of_it() {
        let turn = Turn::Tool {
            message: "reading".into(),
            tool: "shell".into(),
            arguments: json!({"command": "cat config"}),
            output: "region=x".into(),
        };
        let step = step(2, &turn);
        assert_eq!(step["tool_calls"][0]["tool_call_id"], "call-2");
        assert_eq!(
            step["observation"]["results"][0]["source_call_id"],
            "call-2"
        );
    }

    #[test]
    fn writing_is_deterministic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let script = Script::correction("same", "q", "w", "f", "a");
        let a = std::fs::read(script.write(dir.path()).expect("first write")).expect("read");
        let b = std::fs::read(script.write(dir.path()).expect("second write")).expect("read");
        assert_eq!(a, b);
    }
}
