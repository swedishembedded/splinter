// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures the task generator specs share: a scripted generator model, a
//! stored source, a python environment in the process sandbox, and the
//! generator wired to them.

// Each spec binary uses a subset of these helpers.
#![allow(dead_code)]

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use splinter_agent::solve::Model;
use splinter_knowledge::tasks::{ModelTaskGenerator, SourceText};
use splinter_record::clock::FixedClock;
use splinter_record::source::{CapturedSource, Origin, PartContent, SourceId};
use splinter_record::sources::SourceStore;
use splinter_record::StateRoot;
use splinter_sandbox::{Limits, ProcessSandbox, RuntimeEnvironment, RuntimeRegistry};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

/// A generator model that answers each request with the next reply of its
/// script, and keeps the requests it was sent.
pub struct Scripted {
    replies: Mutex<VecDeque<String>>,
    pub seen: Mutex<Vec<CompletionRequest>>,
}

impl Scripted {
    /// A model that replies `replies`, in order.
    pub fn new(replies: Vec<String>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
            seen: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "generator-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.seen.lock().unwrap().push(req);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| panic!("the script has no reply left"));
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// A scratch directory removed when dropped.
pub struct Scratch(pub PathBuf);

impl Scratch {
    /// A fresh scratch directory named for `test`.
    pub fn new(test: &str) -> Self {
        let path = std::env::temp_dir().join(format!("splinter-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The part name the fixtures' documents are stored under.
pub const PART: &str = "notes.md";

/// `text` stored as a Markdown document in a source store under `scratch`;
/// the store and the loaded text of its one part.
pub fn stored(scratch: &Scratch, text: &str) -> (SourceStore, SourceId, SourceText) {
    let store = SourceStore::open(&StateRoot::new(scratch.0.join("state")));
    let captured = CapturedSource::new(
        Origin::Document {
            path: "/notes/notes.md".into(),
        },
        vec![PartContent {
            name: PART.into(),
            media_type: "text/markdown".into(),
            bytes: text.as_bytes().to_vec(),
        }],
        &FixedClock::new("2026-09-30T08:00:00.000Z"),
    )
    .unwrap();
    let id = store.put_source(&captured).unwrap();
    let source = SourceText::load(&store, &id, PART).unwrap();
    (store, id, source)
}

/// python3 in the process sandbox.
pub fn python(scratch: &Scratch) -> RuntimeEnvironment {
    let dir = scratch.0.join("sandbox");
    std::fs::create_dir_all(&dir).unwrap();
    let sandbox = ProcessSandbox::new(&dir, BTreeMap::new(), &[], Limits::default());
    RuntimeEnvironment::new(&RuntimeRegistry::builtin(), "python3", Arc::new(sandbox)).unwrap()
}

/// A generator on `model` over `store`, offered `runtimes`.
pub fn generator(
    model: Arc<Scripted>,
    store: SourceStore,
    runtimes: Vec<RuntimeEnvironment>,
) -> ModelTaskGenerator {
    ModelTaskGenerator::new(Model::new(model, "scripted/generator-1"), store)
        .with_runtimes(runtimes)
}

/// One generated task as the model writes it.
pub fn entry(instruction: &str, reference: &str, section: usize, quote: Option<&str>) -> Value {
    json!({
        "instruction": instruction,
        "reference": reference,
        "evidence": [{ "section": section, "quote": quote }],
    })
}

/// `task` naming `subject` as what it is about.
pub fn about(mut task: Value, subject: &str) -> Value {
    task["subject"] = json!(subject);
    task
}

/// The model's reply holding `tasks`.
pub fn reply(tasks: Vec<Value>) -> String {
    json!({ "tasks": tasks }).to_string()
}
