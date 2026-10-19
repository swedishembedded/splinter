// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that work inside a confined
// directory and leave a complete, replayable record of what they did, for
// its clients. If your team needs expertise in agent runtimes or in
// measuring what an agent really did, you can procure our services by

//! Spec: a worker may be given knowledge to retrieve from. The memory it reads
//! is frozen when the worker starts, it cannot write to it, and the attempt
//! records which snapshot it read and which units it retrieved, so a result is
//! about one memory and the units an answer was built from are in the record.
//! A worker without knowledge is offered no memory tool at all.

#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_agent::knowledge::Knowledge;
use splinter_agent::solve::SolveOptions;
use splinter_agent::work::{WorkOptions, Worker};
use sven_sdk::memory::{MemoryStore, SqliteMemory};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};

struct Scripted {
    seen: Mutex<Vec<CompletionRequest>>,
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "worker-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let results = req
            .messages
            .iter()
            .filter(|m| matches!(m.content, MessageContent::ToolResult { .. }))
            .count();
        let events = match results {
            0 => call("r", r#"{"action":"recall","query":"rotation schedule"}"#),
            1 => call(
                "w",
                r#"{"action":"remember","content":"written during the attempt"}"#,
            ),
            _ => vec![ResponseEvent::TextDelta("done".into()), ResponseEvent::Done],
        };
        self.seen.lock().unwrap().push(req);
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

fn call(id: &str, arguments: &str) -> Vec<ResponseEvent> {
    vec![
        ResponseEvent::ToolCall {
            index: 0,
            id: id.into(),
            name: "semantic_memory".into(),
            arguments: arguments.into(),
        },
        ResponseEvent::Done,
    ]
}

fn options() -> WorkOptions {
    WorkOptions::new(SolveOptions::new(Duration::from_secs(120)))
}

async fn knowledge() -> Knowledge {
    let store: Arc<dyn MemoryStore> = Arc::new(SqliteMemory::open_in_memory().unwrap());
    let knowledge = Knowledge::new(store, "ops-notes");
    knowledge
        .add_document(
            "runbook",
            "Runbook",
            "# Keys\n\nThe deploy key rotation schedule is every friday at noon.\n",
        )
        .await
        .unwrap();
    knowledge
}

#[tokio::test]
async fn a_worker_with_knowledge_retrieves_from_a_frozen_snapshot_and_records_it() {
    let dir = tempfile::tempdir().unwrap();
    let knowledge = knowledge().await;
    let model = Arc::new(Scripted {
        seen: Mutex::new(Vec::new()),
    });
    let mut worker =
        Worker::start_with_knowledge(dir.path(), model.clone(), options(), &[], &knowledge)
            .await
            .unwrap();
    let frozen = worker
        .memory_snapshot()
        .expect("a frozen worker names it")
        .to_owned();

    let report = worker
        .send("when does the deploy key rotate?")
        .await
        .unwrap();

    let used = report.memory.as_ref().expect("the attempt used memory");
    assert_eq!(used.snapshot_id.as_deref(), Some(frozen.as_str()));
    assert_eq!(used.retrievals, 1);
    assert_eq!(used.units.len(), 1, "{used:?}");
    assert!(used.units[0].inserted);
    assert_eq!(
        report
            .attempt_report(Duration::from_secs(1))
            .memory
            .as_ref(),
        Some(used),
        "the attempt report carries the provenance"
    );
    // The write the model tried during the attempt did not reach the memory.
    assert_eq!(knowledge.source_count().await.unwrap(), 1);
    let seen = model.seen.lock().unwrap();
    assert!(
        seen[0].tools.iter().any(|t| t.name == "semantic_memory"),
        "the memory tool is offered to a worker with knowledge"
    );
}

#[tokio::test]
async fn a_worker_without_knowledge_is_offered_no_memory_and_reports_none() {
    let dir = tempfile::tempdir().unwrap();
    let model = Arc::new(Scripted {
        seen: Mutex::new(Vec::new()),
    });
    let mut worker = Worker::start(dir.path(), model.clone(), options()).unwrap();
    assert!(worker.memory_snapshot().is_none());
    let report = worker.send("anything").await.unwrap();
    assert!(report.memory.is_none());
    let seen = model.seen.lock().unwrap();
    assert!(seen[0].tools.iter().all(|t| t.name != "semantic_memory"));
}
