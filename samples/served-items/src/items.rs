// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that grade a language model's
// structured answers exactly, without a judge model, for its clients. If your
// team needs expertise in measurement instruments that cannot mistake a
// tooling fault for a learning result you can procure our services by sending
// an email to info@swedishembedded.com.

//! A task family read from a file of items: each item is one user message
//! and the JSON object that answers it. The reply is read from its last
//! fenced block (else the whole reply), parsed as JSON and compared with the
//! reference exactly; a list of strings compares as a set, since such answers
//! name sets. Nothing is graded by a loss.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_sdk::campaign::brain::{Graded, TaskFamily};
use splinter_sdk::campaign::CampaignError;

/// One item as the file holds it.
#[derive(Clone, Debug, Deserialize)]
pub struct Item {
    /// The item's id.
    pub id: String,
    /// The one user message put to the model.
    pub prompt: String,
    /// The JSON object that answers it.
    pub answer: Value,
    /// The item's part of the split (`held_out`, `train_pool`, ...).
    pub role: String,
}

/// How a reply to an item was judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The answer equals the reference.
    Correct,
    /// An answer parses and differs from the reference.
    Wrong,
    /// No JSON object could be read from the reply.
    Malformed,
}

/// The items of one file.
pub struct Items {
    items: BTreeMap<String, Item>,
    order: Vec<String>,
}

/// The body of the last fenced block of `reply`; `None` when it has none.
fn last_fenced_block(reply: &str) -> Option<&str> {
    let mut last = None;
    let mut rest = reply;
    while let Some(open) = rest.find("```") {
        let after = &rest[open + 3..];
        let Some(newline) = after.find('\n') else {
            break;
        };
        let body = &after[newline + 1..];
        let Some(close) = body.find("```") else {
            break;
        };
        last = Some(&body[..close]);
        rest = &body[close + 3..];
    }
    last
}

/// `value` with every list of strings sorted.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), canonical(v))).collect())
        }
        Value::Array(list) => {
            let mut items: Vec<Value> = list.iter().map(canonical).collect();
            if items.iter().all(Value::is_string) {
                items.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
            }
            Value::Array(items)
        }
        other => other.clone(),
    }
}

/// The verdict on `reply` against `reference`.
#[must_use]
pub fn judge(reply: &str, reference: &Value) -> Verdict {
    let text = last_fenced_block(reply).unwrap_or(reply);
    match serde_json::from_str::<Value>(text.trim()) {
        Ok(answer @ Value::Object(_)) if canonical(&answer) == canonical(reference) => {
            Verdict::Correct
        }
        Ok(Value::Object(_)) => Verdict::Wrong,
        _ => Verdict::Malformed,
    }
}

impl Items {
    /// The items of the JSONL file at `path`.
    ///
    /// # Errors
    /// The file cannot be read, a line is not an item, or an id repeats.
    pub fn open(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut items = BTreeMap::new();
        let mut order = Vec::new();
        for (n, line) in text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
        {
            let item: Item = serde_json::from_str(line)
                .with_context(|| format!("{} line {}", path.display(), n + 1))?;
            anyhow::ensure!(
                item.answer.is_object(),
                "{} line {}: the answer is not a JSON object",
                path.display(),
                n + 1
            );
            order.push(item.id.clone());
            anyhow::ensure!(
                items.insert(item.id.clone(), item).is_none(),
                "{} line {}: a repeated id",
                path.display(),
                n + 1
            );
        }
        Ok(Self { items, order })
    }

    /// The ids of the items whose role is `role`, or `role` itself read as a
    /// comma-separated list of ids, in file order.
    ///
    /// # Errors
    /// An id names no item, or the selection is empty.
    pub fn select(&self, role: &str) -> Result<Vec<String>> {
        let by_role: Vec<String> = self
            .order
            .iter()
            .filter(|id| self.items[*id].role == role)
            .cloned()
            .collect();
        let ids = if by_role.is_empty() {
            role.split(',').map(str::to_string).collect::<Vec<_>>()
        } else {
            by_role
        };
        for id in &ids {
            anyhow::ensure!(
                self.items.contains_key(id),
                "{id:?} is neither a role nor an item id"
            );
        }
        Ok(ids)
    }

    /// The item `id`.
    ///
    /// # Errors
    /// No item has the id.
    pub fn get(&self, id: &str) -> Result<&Item, CampaignError> {
        self.items
            .get(id)
            .ok_or_else(|| CampaignError::Config(format!("{id:?} is not an item")))
    }
}

impl TaskFamily for Items {
    fn describe(&self) -> Value {
        json!({"family": "items", "items": self.order.len()})
    }

    fn pool(&self) -> Vec<String> {
        self.order.clone()
    }

    fn dev(&self) -> Vec<String> {
        self.order.clone()
    }

    fn retention(&self) -> Vec<String> {
        Vec::new()
    }

    fn prompt(&self, task: &str) -> Result<Vec<(String, String)>, CampaignError> {
        Ok(vec![("user".to_string(), self.get(task)?.prompt.clone())])
    }

    fn grade(&self, task: &str, completion: &str) -> Result<Graded, CampaignError> {
        let passed = judge(completion, &self.get(task)?.answer) == Verdict::Correct;
        Ok(Graded {
            reward: f64::from(u8::from(passed)),
            passed,
            feedback: None,
            privileged: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_is_the_last_fenced_object_compared_exactly_with_sets_unordered() {
        let reference = json!({"outcome": "no_effect", "computed": {"fee": "1.50"}});
        let reply = "First ```json\n{\"outcome\": \"x\"}\n``` then\n```json\n{\"computed\": {\"fee\": \"1.50\"}, \"outcome\": \"no_effect\"}\n```";
        assert_eq!(judge(reply, &reference), Verdict::Correct);
        assert_eq!(
            judge(
                "{\"outcome\": \"no_effect\", \"computed\": {\"fee\": 1.5}}",
                &reference
            ),
            Verdict::Wrong,
            "a number is not the two-place decimal asked for"
        );
        assert_eq!(
            judge("```json\n{\"outcome\": \n```", &reference),
            Verdict::Malformed
        );
        assert_eq!(
            judge("[1, 2]", &reference),
            Verdict::Malformed,
            "an answer is an object"
        );
        let set = json!({"missing": ["b", "a"]});
        assert_eq!(
            judge("```\n{\"missing\": [\"a\", \"b\"]}\n```", &set),
            Verdict::Correct
        );
        assert_eq!(
            judge("```\n{\"missing\": [\"a\"]}\n```", &set),
            Verdict::Wrong
        );
    }

    #[test]
    fn items_are_selected_by_role_or_by_id_and_an_unknown_id_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("items.jsonl");
        std::fs::write(
            &path,
            "{\"id\":\"a\",\"prompt\":\"p\",\"answer\":{\"x\":1},\"role\":\"held_out\"}\n{\"id\":\"b\",\"prompt\":\"q\",\"answer\":{\"x\":2},\"role\":\"train_pool\"}\n",
        )
        .unwrap();
        let items = Items::open(&path).unwrap();
        assert_eq!(items.select("held_out").unwrap(), ["a"]);
        assert_eq!(items.select("b,a").unwrap(), ["b", "a"]);
        assert!(items.select("c").is_err());
        assert_eq!(
            items.prompt("b").unwrap(),
            [("user".to_string(), "q".to_string())]
        );
    }
}
