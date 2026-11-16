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
//! Every demonstration follows one general procedure: inspect the workspace
//! first (list its files), read what the request names, discover how a
//! helper is called from its own `--help` before calling it, act, read the
//! result back to verify it, then answer. The expert knows the plan of an
//! instance, never its answer: every value it writes is read out of a tool
//! result, in the file's format, so a demonstration is a correct use of the
//! tools and a tool that misbehaves makes the episode fail verification
//! instead of producing a fabricated success. It reads the tools the request
//! offers and refuses (with the name) when one it needs is missing.

use anyhow::{anyhow, bail, Context as _};
use serde_json::Value;

use crate::formats::{unnumbered, Format};
use crate::tasks::{Instance, Plan, ANSWER_FILE};

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

fn write(tools: &[Value], path: &str, text: &str) -> anyhow::Result<Action> {
    let p = argument(tools, "write_file", &["path"])?;
    let t = argument(tools, "write_file", &["text", "content"])?;
    Ok(Action::Call {
        tool: "write_file".into(),
        arguments: object(&[(p, path.to_string()), (t, text.to_string())]),
    })
}

/// The value at `key` in a read result of a file in `format`.
fn value_in(result: &str, format: Format, key: &str) -> anyhow::Result<String> {
    format
        .parse(&unnumbered(result))
        .into_iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
        .with_context(|| format!("no {key} in {result:?}"))
}

/// The first line of a command's output that is a whole number.
fn first_number(result: &str) -> Option<String> {
    result
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_digit()))
        .map(String::from)
}

/// What a plan writes, from the results of its reads: a path and its text.
type WriteOf = Box<dyn Fn(&[String]) -> anyhow::Result<(String, String)>>;

/// The calls of a plan, given the tool results seen so far: a call needs
/// only the results before it. `Ok(None)` when every call is made; the last
/// call reads back what was written, and the value written is returned with
/// the final answer.
fn plan_step(
    plan: &Plan,
    seen: &[String],
    tools: &[Value],
) -> anyhow::Result<Result<Action, String>> {
    // Inspecting the workspace comes first, always.
    if seen.is_empty() {
        return shell(tools, "ls -R").map(Ok);
    }
    let r = &seen[1..];
    let step = r.len();
    // The reads a plan makes, then what it writes (path, text), then the
    // read-back of that path.
    let (reads, write_of): (Vec<String>, WriteOf) = match plan {
        Plan::Read { slot } => {
            let slot = slot.clone();
            (
                vec![slot.file.clone()],
                Box::new(move |r| {
                    Ok((
                        ANSWER_FILE.into(),
                        format!("{}\n", value_in(&r[0], slot.format, &slot.key)?),
                    ))
                }),
            )
        }
        Plan::Chain { start, next } => {
            let mut files = vec![start.file.clone()];
            files.extend(next.iter().map(|(f, _)| f.clone()));
            let (start, next) = (start.clone(), next.clone());
            (
                files,
                Box::new(move |r| {
                    let mut key = value_in(&r[0], start.format, &start.key)?;
                    for (i, (_, format)) in next.iter().enumerate() {
                        key = value_in(&r[i + 1], *format, &key)?;
                    }
                    Ok((ANSWER_FILE.into(), format!("{key}\n")))
                }),
            )
        }
        Plan::Edit { target, source } => {
            let (target, source) = (target.clone(), source.clone());
            (
                vec![target.file.clone(), source.file.clone()],
                Box::new(move |r| {
                    let value = value_in(&r[1], source.format, &source.key)?;
                    let mut entries = target.format.parse(&unnumbered(&r[0]));
                    let slot = entries
                        .iter_mut()
                        .find(|(k, _)| *k == target.key)
                        .with_context(|| format!("no {} in {:?}", target.key, r[0]))?;
                    slot.1 = value;
                    Ok((target.file.clone(), target.format.render(&entries)))
                }),
            )
        }
        Plan::Count { file, word } => {
            // The count comes from a command, not a read.
            if step == 0 {
                return shell(tools, &format!("grep -c {word} {file} || true")).map(Ok);
            }
            let count = first_number(&r[0]).with_context(|| format!("grep printed {:?}", r[0]))?;
            return finish(tools, step - 1, ANSWER_FILE, &format!("{count}\n"), &count);
        }
        Plan::Compute { helper, a, b } => {
            let mut files = vec![a.file.clone()];
            if b.file != a.file {
                files.push(b.file.clone());
            }
            let n = files.len();
            if step < n {
                return read(tools, &files[step]).map(Ok);
            }
            let va = value_in(&r[0], a.format, &a.key)?;
            let vb = value_in(&r[n - 1], b.format, &b.key)?;
            if step == n {
                return shell(tools, &format!("./{helper} --help")).map(Ok);
            }
            if step == n + 1 {
                let call = crate::skills::usage_call(&r[n], &va, &vb)
                    .with_context(|| format!("no usage in {:?}", r[n]))?;
                return shell(tools, &call).map(Ok);
            }
            let out = first_number(&r[n + 1])
                .with_context(|| format!("./{helper} printed {:?}", r[n + 1]))?;
            return finish(tools, step - n - 2, ANSWER_FILE, &format!("{out}\n"), &out);
        }
    };
    if step < reads.len() {
        return read(tools, &reads[step]).map(Ok);
    }
    let (path, text) = write_of(&r[..reads.len()])?;
    let said = if path == ANSWER_FILE {
        text.trim().to_string()
    } else {
        format!("{path} is updated")
    };
    finish(tools, step - reads.len(), &path, &text, &said)
}

/// The tail every plan ends with: write, read back, answer. `step` counts
/// from the write.
fn finish(
    tools: &[Value],
    step: usize,
    path: &str,
    text: &str,
    said: &str,
) -> anyhow::Result<Result<Action, String>> {
    match step {
        0 => write(tools, path, text).map(Ok),
        1 => read(tools, path).map(Ok),
        _ => Ok(Err(said.to_string())),
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
    Ok(match plan_step(&instance.plan, &seen, tools)? {
        Ok(action) => action,
        Err(said) => Action::Finish(closing(&instance.id, &said)),
    })
}

/// The final reply, in one of several phrasings chosen by the instance, so no
/// stock sentence dominates the demonstrations.
fn closing(id: &str, said: &str) -> String {
    let pick = id.bytes().fold(0usize, |h, b| {
        h.wrapping_mul(31).wrapping_add(usize::from(b))
    });
    if said.ends_with("updated") {
        let forms = [
            "Done: {s}.",
            "{s} as asked.",
            "I changed it: {s}.",
            "{s}; the other entries are unchanged.",
        ];
        return forms[pick % forms.len()].replace("{s}", said);
    }
    let forms = [
        "Done: {s} is in answer.txt.",
        "{s}",
        "answer.txt now holds {s}.",
        "The result is {s}, written to answer.txt.",
        "Wrote {s} to answer.txt.",
    ];
    forms[pick % forms.len()].replace("{s}", said)
}

/// Runs one tool call in the workspace `ws` and returns its result as the
/// agent runtime shows it (a read numbered by line, a write's byte count, a
/// command's output with its status line).
///
/// # Errors
/// A file cannot be read or written, or a command cannot be started.
pub fn execute(ws: &std::path::Path, tool: &str, arguments: &Value) -> anyhow::Result<String> {
    let arg = |k: &str| arguments[k].as_str().unwrap_or_default().to_string();
    Ok(match tool {
        "read_file" => std::fs::read_to_string(ws.join(arg("path")))?
            .lines()
            .enumerate()
            .map(|(i, l)| format!("L{}:{l}", i + 1))
            .collect::<Vec<_>>()
            .join("\n"),
        "write_file" => {
            let path = ws.join(arg("path"));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, arg("text"))?;
            format!("wrote {} bytes to {}", arg("text").len(), arg("path"))
        }
        _ => {
            let out = std::process::Command::new("sh")
                .arg("-c")
                .arg(arg("shell_command"))
                .current_dir(ws)
                .output()?;
            crate::world::shell_result(
                &String::from_utf8_lossy(&out.stdout),
                out.status.code().unwrap_or(-1),
            )
        }
    })
}

/// The tools as the agent runtime offers them, for running the expert
/// outside an episode.
#[must_use]
pub fn wire_tools() -> Vec<Value> {
    crate::collect::tool_sigs()
        .iter()
        .map(|t| {
            let props: serde_json::Map<String, Value> = t.params.iter().map(|p| (p.clone(), serde_json::json!({}))).collect();
            serde_json::json!({ "type": "function", "function": { "name": t.name, "parameters": { "properties": props } } })
        })
        .collect()
}

/// The expert's whole episode on `instance` in the workspace `ws`, tools run
/// for real: the turns as `(action, result)`, the last a finish.
///
/// # Errors
/// As [`next`] and [`execute`], or the procedure does not finish.
pub fn solve(
    instance: &Instance,
    ws: &std::path::Path,
) -> anyhow::Result<Vec<(Action, Option<String>)>> {
    let tools = wire_tools();
    let mut messages = vec![serde_json::json!({ "role": "user", "content": instance.instruction })];
    let mut turns = Vec::new();
    for _ in 0..16 {
        let action = next(instance, &messages, &tools)?;
        match &action {
            Action::Finish(_) => {
                turns.push((action, None));
                return Ok(turns);
            }
            Action::Call { tool, arguments } => {
                let result = execute(ws, tool, arguments)?;
                messages.push(serde_json::json!({ "role": "tool", "content": result }));
                turns.push((action, Some(result)));
            }
        }
    }
    bail!("{}: the procedure did not finish", instance.id)
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
    fn the_expert_inspects_first_and_reads_its_values_from_the_tools() {
        // seed: any seed; the procedure must hold for every instance.
        let instance = instances(4, Split::Train, 3).remove(2);
        assert_eq!(instance.skill, crate::skills::Skill::Compute);
        let mut messages = vec![json!({ "role": "user", "content": instance.instruction })];
        assert_eq!(
            next(&instance, &messages, &tools()).unwrap(),
            shell(&tools(), "ls -R").unwrap(),
            "inspect first"
        );
        let Plan::Compute { helper, a, b } = &instance.plan else {
            unreachable!()
        };
        messages.push(json!({ "role": "tool", "content": "listing" }));
        // Every operand file holds the operands at 17 and 5 in this test,
        // whatever the instance's own values are.
        let fake = |format: crate::formats::Format| {
            format.render(&[(a.key.clone(), "17".into()), (b.key.clone(), "5".into())])
        };
        let mut reads = vec![a.clone()];
        if b.file != a.file {
            reads.push(b.clone());
        }
        for slot in &reads {
            assert_eq!(
                next(&instance, &messages, &tools()).unwrap(),
                read(&tools(), &slot.file).unwrap()
            );
            messages.push(json!({ "role": "tool", "content": fake(slot.format) }));
        }
        assert_eq!(
            next(&instance, &messages, &tools()).unwrap(),
            shell(&tools(), &format!("./{helper} --help")).unwrap()
        );
        messages.push(
            json!({ "role": "tool", "content": format!("usage: ./{helper} --left A --right B") }),
        );
        assert_eq!(
            next(&instance, &messages, &tools()).unwrap(),
            shell(&tools(), &format!("./{helper} --left 17 --right 5")).unwrap(),
            "the values come from the tool results and the call from the helper's own usage"
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
