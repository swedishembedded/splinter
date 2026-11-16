// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements programmatic expert policies that produce
// verified agent demonstrations for training small models, for its clients.
// If your team needs expertise in demonstration data for tool-using agents,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The deterministic expert: the next action for a conversation the agent
//! runtime sent, decided from what the tools have shown so far.
//!
//! The expert knows each family's procedure, not the answer: every value it
//! writes is read out of a tool result in the conversation, so a
//! demonstration it produces is a correct use of the tools, and a tool that
//! misbehaves makes the episode fail verification instead of producing a
//! fabricated success. It reads the tools the request offers and refuses
//! (with the name) when one it needs is missing or takes other arguments.

use anyhow::{anyhow, bail, Context as _};
use serde_json::Value;

use crate::tasks::{Family, Instance, ANSWER_FILE};

/// What the expert says next.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Call `tool` with `arguments`.
    Call {
        /// The tool.
        tool: String,
        /// Its arguments.
        arguments: Value,
    },
    /// Finish with this text.
    Finish(String),
}

impl Action {
    /// The action as the OpenAI assistant message answering `messages`; a
    /// call's arguments are JSON text, as on the wire, and its id is unique in
    /// the conversation (one per earlier tool result).
    #[must_use]
    pub fn message(&self, messages: &[Value]) -> Value {
        let id = format!("call_{}", tool_results(messages).len());
        match self {
            Self::Finish(text) => serde_json::json!({ "role": "assistant", "content": text }),
            Self::Call { tool, arguments } => serde_json::json!({
                "role": "assistant", "content": null, "tool_calls": [{
                    "id": id, "type": "function",
                    "function": { "name": tool, "arguments": arguments.to_string() } }] }),
        }
    }
}

/// The tool results in `messages`, in order (OpenAI chat format).
fn tool_results(messages: &[Value]) -> Vec<String> {
    messages
        .iter()
        .filter(|m| m["role"] == "tool")
        .map(|m| content_text(&m["content"]))
        .collect()
}

/// A message's content as text, whether a string or a list of text parts.
#[must_use]
pub fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// The lines of a file a read showed, without the reader's `L<n>:` prefixes.
fn file_lines(result: &str) -> Vec<String> {
    result
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix('L')?;
            let (n, text) = rest.split_once(':')?;
            n.chars()
                .all(|c| c.is_ascii_digit())
                .then(|| text.to_string())
        })
        .collect()
}

/// The first line of a command's output that is a whole number.
fn first_number(result: &str) -> Option<String> {
    result
        .lines()
        .map(|l| l.trim().trim_start_matches("error:").trim())
        .find(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_digit()))
        .map(String::from)
}

/// The argument name the tool `name` takes for `role`, checked against its
/// schema in the request: the expert never calls a tool the agent does not
/// offer, or with arguments it does not take.
fn argument(tools: &[Value], name: &str, candidates: &[&str]) -> anyhow::Result<String> {
    let tool = tools
        .iter()
        .find(|t| t["function"]["name"] == name)
        .ok_or_else(|| anyhow!("the agent offers no {name} tool, which the expert needs"))?;
    let props = &tool["function"]["parameters"]["properties"];
    candidates
        .iter()
        .find(|c| props.get(**c).is_some())
        .map(|c| (*c).to_string())
        .ok_or_else(|| anyhow!("the {name} tool takes none of {candidates:?}"))
}

/// A JSON object of the given fields.
fn object(fields: &[(String, String)]) -> Value {
    Value::Object(
        fields
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect(),
    )
}

fn read(tools: &[Value], path: &str) -> anyhow::Result<Action> {
    Ok(Action::Call {
        tool: "read_file".into(),
        arguments: object(&[(argument(tools, "read_file", &["path"])?, path.to_string())]),
    })
}

fn shell(tools: &[Value], command: &str) -> anyhow::Result<Action> {
    Ok(Action::Call {
        tool: "shell".into(),
        arguments: object(&[(
            argument(tools, "shell", &["shell_command", "command"])?,
            command.to_string(),
        )]),
    })
}

fn write_answer(tools: &[Value], value: &str) -> anyhow::Result<Action> {
    let path = argument(tools, "write_file", &["path"])?;
    let text = argument(tools, "write_file", &["text", "content"])?;
    Ok(Action::Call {
        tool: "write_file".into(),
        arguments: object(&[
            (path, ANSWER_FILE.to_string()),
            (text, format!("{value}\n")),
        ]),
    })
}

/// The input files of a combine instance, in the order its request names
/// them.
fn combine_inputs(instance: &Instance) -> anyhow::Result<(String, String)> {
    let mut named: Vec<(usize, &String)> = instance
        .files
        .keys()
        .filter(|k| k.starts_with("inputs/"))
        .filter_map(|k| instance.instruction.find(k.as_str()).map(|at| (at, k)))
        .collect();
    named.sort();
    match named.as_slice() {
        [(_, a), (_, b)] => Ok(((*a).clone(), (*b).clone())),
        _ => bail!("{}: the request does not name two input files", instance.id),
    }
}

/// The expert's next action on `instance`, given the request's `messages`
/// and `tools`.
///
/// # Errors
/// A tool the procedure needs is not offered, or a tool result does not hold
/// what the procedure reads from it.
pub fn next(instance: &Instance, messages: &[Value], tools: &[Value]) -> anyhow::Result<Action> {
    let seen = tool_results(messages);
    let step = seen.len();
    let answer_written = |value: &str| {
        Action::Finish(format!(
            "The answer is {value}. It is written to {ANSWER_FILE}."
        ))
    };
    match instance.family {
        Family::Combine => {
            let (a, b) = combine_inputs(instance)?;
            let value_of = |i: usize| -> anyhow::Result<String> {
                file_lines(&seen[i])
                    .iter()
                    .find_map(|l| first_number(l))
                    .with_context(|| format!("reading a number from {:?}", seen[i]))
            };
            match step {
                0 => read(tools, &a),
                1 => read(tools, &b),
                2 => shell(
                    tools,
                    &format!("./calc combine {} {}", value_of(0)?, value_of(1)?),
                ),
                3 => write_answer(
                    tools,
                    &first_number(&seen[2])
                        .with_context(|| format!("./calc printed {:?}", seen[2]))?,
                ),
                _ => Ok(answer_written(&first_number(&seen[2]).unwrap_or_default())),
            }
        }
        Family::Lookup => {
            let item = instance
                .instruction
                .split_whitespace()
                .find(|w| {
                    instance.files["registry.txt"]
                        .lines()
                        .any(|l| l.split(':').next() == Some(w.trim_end_matches('.')))
                })
                .map(|w| w.trim_end_matches('.').to_string())
                .with_context(|| {
                    format!("{}: the request names no registered item", instance.id)
                })?;
            let code = || -> anyhow::Result<String> {
                file_lines(&seen[0])
                    .iter()
                    .find_map(|l| {
                        l.strip_prefix(&format!("{item}: "))
                            .map(|c| c.trim().to_string())
                    })
                    .with_context(|| format!("finding {item} in {:?}", seen[0]))
            };
            match step {
                0 => read(tools, "registry.txt"),
                1 => write_answer(tools, &code()?),
                _ => Ok(answer_written(&code()?)),
            }
        }
        Family::Count => {
            let word = ["ERROR", "WARN", "TIMEOUT"]
                .into_iter()
                .find(|w| instance.instruction.contains(w))
                .with_context(|| format!("{}: the request names no word to count", instance.id))?;
            let count =
                || first_number(&seen[0]).with_context(|| format!("grep printed {:?}", seen[0]));
            match step {
                // grep exits 1 when it counts nothing; the count is still
                // the answer.
                0 => shell(tools, &format!("grep -c {word} log.txt || true")),
                1 => write_answer(tools, &count()?),
                _ => Ok(answer_written(&count()?)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{instances, Split};
    use serde_json::json;

    fn tools() -> Vec<Value> {
        let tool = |name: &str, props: Value| json!({ "type": "function", "function": { "name": name, "parameters": { "properties": props } } });
        vec![
            tool("read_file", json!({ "path": {} })),
            tool("write_file", json!({ "path": {}, "text": {} })),
            tool("shell", json!({ "shell_command": {} })),
        ]
    }

    #[test]
    fn the_expert_reads_its_values_from_the_tools() {
        // seed: any seed; the procedure must hold for every instance.
        let instance = instances(4, Split::Train, 1).remove(0);
        let mut messages = vec![json!({ "role": "user", "content": instance.instruction })];
        let (a, b) = combine_inputs(&instance).unwrap();
        assert_eq!(
            next(&instance, &messages, &tools()).unwrap(),
            read(&tools(), &a).unwrap()
        );
        messages.push(json!({ "role": "tool", "content": "L1:17" }));
        assert_eq!(
            next(&instance, &messages, &tools()).unwrap(),
            read(&tools(), &b).unwrap()
        );
        messages.push(json!({ "role": "tool", "content": "L1:5" }));
        assert_eq!(
            next(&instance, &messages, &tools()).unwrap(),
            shell(&tools(), "./calc combine 17 5").unwrap(),
            "the values come from the tool results, not from the instance"
        );
    }

    #[test]
    fn every_call_of_a_conversation_has_its_own_id() {
        let call = Action::Call {
            tool: "shell".into(),
            arguments: json!({}),
        };
        let first = call.message(&[]);
        let second = call.message(&[json!({ "role": "tool", "content": "x" })]);
        assert_ne!(first["tool_calls"][0]["id"], second["tool_calls"][0]["id"]);
    }

    #[test]
    fn a_missing_tool_is_refused_by_name() {
        // seed: any seed.
        let instance = instances(4, Split::Train, 3).remove(2);
        let err = next(&instance, &[], &tools()[..2]).unwrap_err();
        assert!(err.to_string().contains("shell"), "{err}");
    }
}
