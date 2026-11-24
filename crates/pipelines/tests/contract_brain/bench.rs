// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that the training records one system writes are the records
// another system's trainer reads, for its clients. If your team needs expertise in training-data contracts between
// agent runtimes and trainers then you can procure our services by sending an email to info@swedishembedded.com.

//! A served brain over a tokenizer-only model, and the independent tokenizer the expectations are computed with.
//!
//! The model is brain's own contract fixture (`training-records/v2/model`: one id per letter, three markers, a chat
//! template that writes a turn as its role, its content and the end-of-turn marker) widened to every printable ASCII
//! character and extended to write a call's name and arguments, so the records Splinter writes (a system prompt
//! with punctuation, a tool call) are tokenizable. One character is one token, which is what lets a test count
//! the tokens of an action without a tokenizer library. `check_dataset` reads nothing else of a model.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};
use splinter_model::served_train::{check_dataset, CheckSpec, DatasetReport};
use splinter_model::{ManagedBrain, ServiceSettings};

/// The model store's name of the base the records are read against.
const BASE: &str = "contract/chars";
pub const END_OF_TURN: i64 = 151_645;
pub const TURN_START: i64 = 151_644;
const END_OF_TEXT: i64 = 151_643;

/// A turn is its role and a newline, its content, what its calls say and the end-of-turn marker: calls are written as the
/// function's name followed by its arguments' JSON text, so a supervised call is countable character by character.
const TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}\
{% if message.tool_calls %}{% for call in message.tool_calls %}{{ call.function.name }}{{ call.function.arguments }}\
{% endfor %}{% endif %}<|im_end|>{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// The id of each character the toy tokenizer knows: printable ASCII and the newline, in code order.
fn vocabulary() -> BTreeMap<char, i64> {
    std::iter::once('\n')
        .chain(' '..='~')
        .enumerate()
        .map(|(id, c)| (c, id as i64))
        .collect()
}

/// How the byte-pair tokenizer spells `c` in its vocabulary: printable characters as themselves, a space and a
/// newline as the printable stand-ins of GPT-2's byte mapping (a vocabulary key with a bare space is never matched,
/// and the character then silently has no token).
fn vocabulary_key(c: char) -> char {
    match c {
        ' ' => '\u{120}',
        '\n' => '\u{10a}',
        other => other,
    }
}

/// The ids of `text`: one per character.
pub fn encode(text: &str) -> Vec<i64> {
    let vocabulary = vocabulary();
    text.chars()
        .map(|c| {
            *vocabulary
                .get(&c)
                .unwrap_or_else(|| panic!("{c:?} is not in the toy tokenizer"))
        })
        .collect()
}

/// The ids of one turn as the template writes it: its start marker, role, content, calls and end marker.
pub fn turn(role: &str, body: &str) -> Vec<i64> {
    let mut ids = vec![TURN_START];
    ids.extend(encode(&format!("{role}\n{body}")));
    ids.push(END_OF_TURN);
    ids
}

/// What a supervised answer is trained as: its text and the end-of-turn marker that closes it.
pub fn answer(body: &str) -> Vec<i64> {
    let mut ids = encode(body);
    ids.push(END_OF_TURN);
    ids
}

/// The ids that open an answer: the start marker and the role.
pub fn answer_prompt() -> Vec<i64> {
    let mut ids = vec![TURN_START];
    ids.extend(encode("assistant\n"));
    ids
}

/// The tokenizer-only model: the fixture's files, widened; the digest of its `tokenizer.json`.
fn write_model(dir: &Path) -> String {
    std::fs::create_dir_all(dir).unwrap();
    let vocab: serde_json::Map<String, Value> = vocabulary()
        .into_iter()
        .map(|(c, id)| (vocabulary_key(c).to_string(), json!(id)))
        .collect();
    let tokenizer = serde_json::to_vec_pretty(&json!({
        "model": {"vocab": vocab, "merges": []},
        "added_tokens": [
            {"content": "<|endoftext|>", "id": END_OF_TEXT},
            {"content": "<|im_start|>", "id": TURN_START},
            {"content": "<|im_end|>", "id": END_OF_TURN},
        ],
    }))
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), &tokenizer).unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_vec_pretty(&json!({"chat_template": TEMPLATE})).unwrap(),
    )
    .unwrap();
    // The store lists a directory without weights only as a compound model; its one role is the directory itself.
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec_pretty(&json!({"id": BASE, "family": "toy", "roles": {"dir": "."}}))
            .unwrap(),
    )
    .unwrap();
    format!("sha256:{:x}", Sha256::digest(&tokenizer))
}

/// A brain server on the CPU whose model store holds the toy tokenizer.
pub struct Bench {
    pub dir: tempfile::TempDir,
    /// The `sha256:` digest of the model's `tokenizer.json`, which a sampled record names.
    pub tokenizer: String,
    brain: ManagedBrain,
}

impl Bench {
    /// Starts the first `brain` on `PATH` over a fresh model store.
    pub fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        let tokenizer = write_model(&models.join(BASE));
        let binary = splinter_orchestrator::Config::from_env()
            .brain_binary
            .expect("the contract test needs a `brain` binary on PATH");
        let brain = ManagedBrain::start(
            &ServiceSettings {
                binary,
                models,
                state: dir.path().join("brain-service"),
                device: Some("cpu".into()),
                backend: Some("cpu".into()),
                max_batch: Some(1),
                serve_adapters: false,
                adapter_base: None,
            },
            &sven_sdk::CancelToken::default(),
        )
        .unwrap();
        Self {
            dir,
            tokenizer,
            brain,
        }
    }

    /// Brain's report of the dataset at `path` read as `spec` says, as its trainer would read it.
    pub fn check(&self, path: &Path, spec: &CheckSpec<'_>) -> DatasetReport {
        check_dataset(self.brain.endpoints(), path, spec).unwrap()
    }

    /// [`Bench::check`] of `kind` records, one example per answer.
    pub fn check_as(
        &self,
        path: &Path,
        kind: splinter_model::served_train::DatasetKind,
    ) -> DatasetReport {
        let mut spec = CheckSpec::chat(BASE, false, false);
        spec.kind = kind;
        self.check(path, &spec)
    }

    /// A path for a dataset named `name`.
    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
}
