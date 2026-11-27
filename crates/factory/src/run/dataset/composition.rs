// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training datasets whose make-up is measured,
// not assumed, before a model is trained on them, for its clients. If your
// team needs expertise in curating agent fine-tuning data, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What an SFT dataset is made of (`composition.json`): how many records,
//! trials and tasks, how many different behaviours it teaches, and how much
//! of it the loss reads. Two arms of an experiment built from one set of
//! attempts are compared by it before either is trained.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::chat::WireMessage;
use splinter_core::digest::{digest_of, Digest};
use splinter_data::RecordBody;

use super::Admitted;

/// The composition file of an SFT dataset product.
pub const COMPOSITION_FILE: &str = "composition.json";

/// What an SFT dataset is made of.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Composition {
    /// Records in both files.
    pub records: usize,
    /// Records in `train.jsonl`.
    pub train_records: usize,
    /// Records in `validation.jsonl`.
    pub validation_records: usize,
    /// Distinct trials (pass and slot) the records are of.
    pub trials: usize,
    /// Distinct tasks the records are of.
    pub tasks: usize,
    /// Distinct action sequences: each record's supervised tool calls in
    /// order, each by its tool name and its arguments with their keys sorted
    /// and their spacing normalised.
    pub action_sequences: usize,
    /// Assistant messages the loss reads (`train: true`).
    pub supervised_messages: u64,
    /// Assistant messages kept as context (`train: false`).
    pub masked_messages: u64,
    /// The characters of the supervised messages: text, reasoning and each
    /// call's name and arguments (the trainer's tokens are not counted here).
    pub supervised_characters: u64,
    /// Actions kept as context because every call they made failed
    /// (`supervise: productive`; none otherwise).
    pub failed_actions_masked: u64,
    /// Which actions are supervised (`all`, `productive`).
    pub supervise: String,
    /// The most trials kept per task, when limited.
    pub per_task: Option<usize>,
    /// The negative control the dataset was built as, if any.
    pub control: Option<String>,
}

/// How an SFT dataset was built, beside its records.
pub(super) struct Build<'a> {
    pub(super) train_records: usize,
    pub(super) validation_records: usize,
    pub(super) failed_actions_masked: u64,
    pub(super) supervise: &'a str,
    pub(super) per_task: Option<usize>,
    pub(super) control: Option<&'a str>,
}

/// What one record's conversation supervises: the measures
/// [`Composition`] sums over a dataset, for one record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordMeasure {
    /// Assistant messages the loss reads (`train: true`).
    pub supervised_messages: u64,
    /// Assistant messages kept as context (`train: false`).
    pub masked_messages: u64,
    /// The characters of the supervised messages: text, reasoning and each
    /// call's name and arguments.
    pub supervised_characters: u64,
    /// The digest of the supervised tool calls in order, each by its name
    /// and its arguments with their keys sorted and their spacing
    /// normalised: two records with one digest teach one action sequence.
    pub action_sequence: Digest,
}

/// The measures of the conversation `messages`.
#[must_use]
pub fn measure(messages: &[WireMessage]) -> RecordMeasure {
    let mut sequence: Vec<(&str, String)> = Vec::new();
    let (mut supervised, mut masked, mut characters) = (0u64, 0u64, 0u64);
    for m in messages.iter().filter(|m| m.role == "assistant") {
        if !m.train {
            masked += 1;
            continue;
        }
        supervised += 1;
        characters += count(&m.content) + m.reasoning_content.as_deref().map_or(0, count);
        for call in &m.tool_calls {
            characters += count(&call.function.name) + count(&call.function.arguments);
            sequence.push((&call.function.name, normalised(&call.function.arguments)));
        }
    }
    RecordMeasure {
        supervised_messages: supervised,
        masked_messages: masked,
        supervised_characters: characters,
        action_sequence: digest_of(&sequence),
    }
}

/// The composition of the records of `admitted`, built as `build` says.
pub(super) fn composition(admitted: &[Admitted], build: &Build<'_>) -> Composition {
    let mut trials = BTreeSet::new();
    let mut tasks = BTreeSet::new();
    let mut sequences: BTreeSet<Digest> = BTreeSet::new();
    let (mut supervised, mut masked, mut characters) = (0u64, 0u64, 0u64);
    for a in admitted {
        trials.insert((a.source.as_str(), a.trial.slot.to_string()));
        tasks.insert(a.trial.task.as_str());
        let RecordBody::Chat { messages, .. } = &a.record.body else {
            continue;
        };
        let m = measure(messages);
        supervised += m.supervised_messages;
        masked += m.masked_messages;
        characters += m.supervised_characters;
        sequences.insert(m.action_sequence);
    }
    Composition {
        records: admitted.len(),
        train_records: build.train_records,
        validation_records: build.validation_records,
        trials: trials.len(),
        tasks: tasks.len(),
        action_sequences: sequences.len(),
        supervised_messages: supervised,
        masked_messages: masked,
        supervised_characters: characters,
        failed_actions_masked: build.failed_actions_masked,
        supervise: build.supervise.to_owned(),
        per_task: build.per_task,
        control: build.control.map(str::to_owned),
    }
}

fn count(text: &str) -> u64 {
    text.chars().count() as u64
}

/// `arguments` with object keys sorted and no spacing; as written when they
/// are not JSON.
fn normalised(arguments: &str) -> String {
    serde_json::from_str::<Value>(arguments)
        .map_or_else(|_| arguments.to_owned(), |v| sorted(&v).to_string())
}

fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let ordered: BTreeMap<&String, Value> =
                map.iter().map(|(k, v)| (k, sorted(v))).collect();
            Value::Object(ordered.into_iter().map(|(k, v)| (k.clone(), v)).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

impl Composition {
    /// The composition as rows of the dataset's report.
    pub(super) fn rows(&self) -> Vec<(&'static str, String)> {
        let mut rows = vec![
            ("records", self.records.to_string()),
            ("trials", self.trials.to_string()),
            ("tasks", self.tasks.to_string()),
            (
                "distinct action sequences",
                self.action_sequences.to_string(),
            ),
            (
                "supervised assistant messages",
                self.supervised_messages.to_string(),
            ),
            (
                "assistant messages kept as context",
                self.masked_messages.to_string(),
            ),
            (
                "supervised characters",
                self.supervised_characters.to_string(),
            ),
            ("supervise", self.supervise.clone()),
        ];
        if let Some(limit) = self.per_task {
            rows.push(("per task", limit.to_string()));
        }
        if let Some(control) = &self.control {
            rows.push(("control", control.clone()));
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::normalised;

    #[test]
    fn arguments_that_differ_only_in_key_order_or_spacing_are_one() {
        assert_eq!(
            normalised(r#"{"b": [1, {"y": 2, "x": 1}], "a": "s"}"#),
            normalised(r#"{"a":"s","b":[1,{"x":1,"y":2}]}"#)
        );
        assert_ne!(
            normalised(r#"{"a": [1, 2]}"#),
            normalised(r#"{"a": [2, 1]}"#)
        );
        assert_eq!(normalised("not json"), "not json");
    }
}
