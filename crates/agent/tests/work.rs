// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that work inside a confined
// directory and leave a complete, replayable record of what they did, for
// its clients. If your team needs expertise in agent runtimes or in
// measuring what an agent really did, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a worker edits the files of one directory through sven's own
//! tools and stays inside it. Tools that reach the network or another
//! model are not offered. A run that asks for more tool calls than its cap
//! allows is stopped and says so. Every event reaches the observer, in
//! order. A suspended worker continues from its conversation.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_agent::solve::SolveOptions;
use splinter_agent::work::{Observed, Observer, WorkOptions, Worker, WITHHELD_TOOLS};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};
use sven_sdk::{RunConclusion, SessionEvent};

/// A model that answers from a script over the request it is sent, and
/// keeps every request so the test can inspect what it was offered.
struct Scripted {
    script: fn(&CompletionRequest) -> Vec<ResponseEvent>,
    seen: Mutex<Vec<CompletionRequest>>,
}

impl Scripted {
    fn new(script: fn(&CompletionRequest) -> Vec<ResponseEvent>) -> Arc<Self> {
        Arc::new(Self {
            script,
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
        "worker-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let events = (self.script)(&req);
        self.seen.lock().unwrap().push(req);
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

fn text(reply: &str) -> Vec<ResponseEvent> {
    vec![ResponseEvent::TextDelta(reply.into()), ResponseEvent::Done]
}

fn call(id: &str, name: &str, arguments: serde_json::Value) -> Vec<ResponseEvent> {
    vec![
        ResponseEvent::ToolCall {
            index: 0,
            id: id.into(),
            name: name.into(),
            arguments: arguments.to_string(),
        },
        ResponseEvent::Done,
    ]
}

fn tool_results(req: &CompletionRequest) -> usize {
    req.messages
        .iter()
        .filter(|m| matches!(m.content, MessageContent::ToolResult { .. }))
        .count()
}

fn has_tool_result(req: &CompletionRequest) -> bool {
    tool_results(req) > 0
}

fn options() -> WorkOptions {
    WorkOptions::new(SolveOptions::new(Duration::from_secs(120)))
}

fn observed_into(log: &Arc<Mutex<Vec<SessionEvent>>>) -> Observer {
    let log = Arc::clone(log);
    Arc::new(move |seen: Observed<'_>| {
        if let Observed::Event(event) = seen {
            log.lock().unwrap().push(event.clone());
        }
    })
}

#[tokio::test]
async fn a_worker_edits_a_file_in_its_workspace_and_the_observer_sees_every_call() {
    let dir = tempfile::tempdir().unwrap();
    let model = Scripted::new(|req| {
        if has_tool_result(req) {
            text("wrote it")
        } else {
            call(
                "c1",
                "write_file",
                serde_json::json!({"path": "notes.txt", "text": "hello", "append": false}),
            )
        }
    });
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut with_observer = options();
    with_observer.observer = Some(observed_into(&log));
    let mut worker = Worker::start(dir.path(), model, with_observer).unwrap();

    let report = worker.send("write notes.txt").await.unwrap();

    assert_eq!(report.solution.conclusion, RunConclusion::Success);
    assert_eq!(report.solution.final_output.as_deref(), Some("wrote it"));
    assert_eq!(report.tool_calls, 1);
    assert!(!report.stopped_by_tool_cap);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("notes.txt")).unwrap(),
        "hello"
    );
    let events = log.lock().unwrap();
    let started = events
        .iter()
        .position(|e| matches!(e, SessionEvent::ToolCallStarted(c) if c.name == "write_file"));
    let finished = events.iter().position(
        |e| matches!(e, SessionEvent::ToolCallFinished { tool_name, is_error: false, .. } if tool_name == "write_file"),
    );
    assert!(
        started.is_some() && finished.is_some() && started < finished,
        "the observer saw the call start and then finish: {events:?}"
    );
}

#[tokio::test]
async fn a_worker_is_not_offered_the_tools_that_reach_the_network_or_another_model() {
    let dir = tempfile::tempdir().unwrap();
    let model = Scripted::new(|_| text("nothing to do"));
    let mut worker = Worker::start(dir.path(), model.clone(), options()).unwrap();
    worker.send("anything").await.unwrap();

    let seen = model.seen.lock().unwrap();
    let offered: Vec<&str> = seen[0].tools.iter().map(|t| t.name.as_str()).collect();
    for withheld in WITHHELD_TOOLS {
        assert!(
            !offered.contains(withheld),
            "{withheld} is offered: {offered:?}"
        );
    }
    for needed in ["read_file", "write_file", "edit_file", "shell", "grep"] {
        assert!(
            offered.contains(&needed),
            "{needed} is missing: {offered:?}"
        );
    }
}

#[tokio::test]
async fn a_worker_cannot_write_outside_its_workspace() {
    let outer = tempfile::tempdir().unwrap();
    let inner = outer.path().join("repo");
    std::fs::create_dir(&inner).unwrap();
    // The same call shape twice: once through the jail's wall, once within
    // it, so the refusal is the jail's and not a malformed call.
    let model = Scripted::new(|req| match tool_results(req) {
        0 => call(
            "c1",
            "write_file",
            serde_json::json!({"path": "../escaped.txt", "text": "x", "append": false}),
        ),
        1 => call(
            "c2",
            "write_file",
            serde_json::json!({"path": "inside.txt", "text": "x", "append": false}),
        ),
        _ => text("done"),
    });
    let mut worker = Worker::start(&inner, model, options()).unwrap();
    worker.send("write outside, then inside").await.unwrap();

    assert!(
        inner.join("inside.txt").exists(),
        "a write within the workspace works"
    );
    assert!(
        !outer.path().join("escaped.txt").exists(),
        "the file tools refuse a path outside the workspace"
    );
}

#[tokio::test]
async fn a_run_that_asks_for_more_tool_calls_than_its_cap_is_stopped_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let model = Scripted::new(|_| {
        call(
            "c",
            "find_file",
            serde_json::json!({"pattern": "*.nothing"}),
        )
    });
    let mut capped = options();
    capped.max_tool_calls = Some(2);
    let mut worker = Worker::start(dir.path(), model, capped).unwrap();

    let report = worker.send("search forever").await.unwrap();

    assert!(report.stopped_by_tool_cap);
    assert_eq!(report.solution.conclusion, RunConclusion::Cancelled);
    assert!(
        report.tool_calls > 2 && report.tool_calls <= 4,
        "stopped soon after the cap, not run to the end: {}",
        report.tool_calls
    );
}

#[tokio::test]
async fn a_suspended_worker_continues_its_conversation() {
    let dir = tempfile::tempdir().unwrap();
    let first = Scripted::new(|_| text("first reply"));
    let mut worker = Worker::start(dir.path(), first, options()).unwrap();
    worker.send("first question").await.unwrap();
    let state = serde_json::to_string(&worker.suspend()).unwrap();

    let second = Scripted::new(|_| text("second reply"));
    let mut resumed = Worker::resume(
        dir.path(),
        second.clone(),
        options(),
        serde_json::from_str(&state).unwrap(),
    )
    .unwrap();
    let report = resumed.send("second question").await.unwrap();

    assert_eq!(
        report.solution.final_output.as_deref(),
        Some("second reply")
    );
    let seen = second.seen.lock().unwrap();
    let said: String = seen[0]
        .messages
        .iter()
        .filter_map(|m| match &m.content {
            MessageContent::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        said.contains("first question") && said.contains("first reply"),
        "the resumed worker remembers the conversation: {said}"
    );
}

#[tokio::test]
async fn a_run_that_repeats_one_call_and_gets_the_same_answer_is_stopped_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "same").unwrap();
    // The model asks to read the same file forever; every answer is identical.
    let model = Scripted::new(|req| {
        call(
            &format!("c{}", tool_results(req)),
            "read_file",
            serde_json::json!({"path": "a.txt"}),
        )
    });
    let mut bounded = options();
    bounded.max_repeated_results = Some(3);
    let mut worker = Worker::start(dir.path(), model, bounded).unwrap();

    let report = worker.send("look at a.txt").await.unwrap();

    assert!(report.stopped_by_repetition);
    assert!(!report.stopped_by_tool_cap);
    // The cancel is seen between events, so the next call may already have
    // been requested: at most one beyond the bound.
    assert!(
        (3..=4).contains(&report.tool_calls),
        "stopped at the third identical answer, not {}",
        report.tool_calls
    );
}

#[tokio::test]
async fn a_repeated_call_whose_answer_changes_is_not_a_repetition() {
    let dir = tempfile::tempdir().unwrap();
    // Write a different text, then read the file, again and again: the read
    // is the same call each time but the answer differs.
    let model = Scripted::new(|req| {
        let n = tool_results(req);
        if n >= 8 {
            text("done")
        } else if n.is_multiple_of(2) {
            call(
                &format!("w{n}"),
                "write_file",
                serde_json::json!({"path": "a.txt", "text": format!("v{n}"), "append": false}),
            )
        } else {
            call(
                &format!("r{n}"),
                "read_file",
                serde_json::json!({"path": "a.txt"}),
            )
        }
    });
    let mut bounded = options();
    bounded.max_repeated_results = Some(2);
    let mut worker = Worker::start(dir.path(), model, bounded).unwrap();

    let report = worker.send("revise a.txt").await.unwrap();

    assert!(!report.stopped_by_repetition);
    assert_eq!(report.solution.conclusion, RunConclusion::Success);
}
