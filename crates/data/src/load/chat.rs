// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that read any dataset
// the same way, for its clients. If your team needs expertise in loading,
// sampling and versioning datasets for agent training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The supervised chat record of the trainers.

use serde_json::Value;

use super::Record;

/// One line of a chat dataset: `messages` (a message with `train: true` is
/// supervised), `tools`, and `metadata` naming the record's `task` (or its
/// `group`). The line is kept as it was read, so a dataset written back is the
/// dataset read.
#[derive(Clone, Debug)]
pub struct ChatRecord {
    line: String,
    value: Value,
}

impl Record for ChatRecord {
    fn parse(line: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        if !value
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|m| !m.is_empty())
        {
            return Err("is not a chat record (no `messages`)".into());
        }
        Ok(Self {
            line: line.to_owned(),
            value,
        })
    }

    fn line(&self) -> &str {
        &self.line
    }
}

impl ChatRecord {
    fn messages(&self) -> &[Value] {
        self.value
            .get("messages")
            .and_then(Value::as_array)
            .map_or(&[], Vec::as_slice)
    }

    /// The task the record was made on (`metadata.task`).
    #[must_use]
    pub fn task(&self) -> Option<&str> {
        self.value.pointer("/metadata/task").and_then(Value::as_str)
    }

    /// What the record's split is drawn by: its `metadata.group`, else its task.
    #[must_use]
    pub fn group(&self) -> Option<&str> {
        self.value
            .pointer("/metadata/group")
            .and_then(Value::as_str)
            .or_else(|| self.task())
    }

    /// The indices of the supervised messages (`train: true`).
    #[must_use]
    pub fn supervised(&self) -> Vec<usize> {
        self.messages()
            .iter()
            .enumerate()
            .filter(|(_, m)| m.get("train").and_then(Value::as_bool) == Some(true))
            .map(|(i, _)| i)
            .collect()
    }

    /// Whether any message keeps its turn's reasoning.
    #[must_use]
    pub fn has_reasoning(&self) -> bool {
        self.messages()
            .iter()
            .any(|m| m.get("reasoning_content").is_some())
    }

    /// The system prompt the record opens with.
    #[must_use]
    pub fn system_prompt(&self) -> Option<&str> {
        self.messages()
            .first()
            .filter(|m| m.get("role").and_then(Value::as_str) == Some("system"))
            .and_then(|m| m.get("content").and_then(Value::as_str))
    }

    /// The tool schema, in canonical JSON: what two records compare it by.
    ///
    /// # Errors
    /// The schema cannot be written canonically.
    pub fn tools_canonical(&self) -> Result<String, String> {
        splinter_core::digest::canonical_json(self.value.get("tools").unwrap_or(&Value::Null))
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .map_err(|e| e.to_string())
    }

    /// The tool calls the record makes, in order: the tool's name and its
    /// arguments (keys sorted, runs of whitespace one space), the arguments
    /// named in `ignore` left out. Two records of one task that make the same
    /// calls are one solution.
    ///
    /// # Errors
    /// A call has no name, or its arguments cannot be written canonically.
    pub fn calls(&self, ignore: &[String]) -> Result<Vec<(String, String)>, String> {
        let mut calls = Vec::new();
        for message in self.messages() {
            for call in message
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let name = call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .ok_or("has a tool call with no name")?;
                let raw = call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let mut arguments: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
                if let Some(object) = arguments.as_object_mut() {
                    object.retain(|key, _| !ignore.contains(key));
                }
                let canonical = splinter_core::digest::canonical_json(&arguments)
                    .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                    .map_err(|e| e.to_string())?;
                calls.push((
                    name.to_owned(),
                    canonical.split_whitespace().collect::<Vec<_>>().join(" "),
                ));
            }
        }
        Ok(calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn line(task: &str, intent: &str, train: bool) -> String {
        json!({
            "messages": [
                {"role": "system", "content": "P"},
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "ok", "train": train, "tool_calls": [{
                    "id": "c", "type": "function",
                    "function": {"name": "vim", "arguments": json!({"commands": ["wq"], "intent": intent}).to_string()}
                }]},
            ],
            "tools": [{"name": "vim"}],
            "metadata": {"task": task},
        })
        .to_string()
    }

    #[test]
    fn a_chat_record_reads_its_task_prompt_supervision_and_calls() {
        let text = line("t1", "save", true);
        let record = ChatRecord::parse(&text).unwrap();
        assert_eq!(record.line(), text, "kept byte for byte");
        assert_eq!(record.task(), Some("t1"));
        assert_eq!(record.group(), Some("t1"));
        assert_eq!(record.system_prompt(), Some("P"));
        assert_eq!(record.supervised(), [2]);
        assert!(!record.has_reasoning());
        let ignoring = record.calls(&["intent".to_owned()]).unwrap();
        let whole = record.calls(&[]).unwrap();
        assert_eq!(
            ignoring,
            [("vim".to_owned(), r#"{"commands":["wq"]}"#.to_owned())]
        );
        assert!(whole[0].1.contains("intent"));
    }

    #[test]
    fn a_line_that_is_not_a_chat_record_is_refused() {
        assert!(ChatRecord::parse("not json").is_err());
        assert!(ChatRecord::parse(r#"{"text": "x"}"#).is_err());
        assert!(ChatRecord::parse(r#"{"messages": []}"#).is_err());
    }
}
