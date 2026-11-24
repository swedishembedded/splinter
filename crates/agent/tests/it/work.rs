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
    assert!(
        report
            .repeated_call
            .as_deref()
            .is_some_and(|c| c.starts_with("read_file") && c.contains("a.txt")),
        "{:?}",
        report.repeated_call
    );
    assert_eq!(
        report.tool_calls, 3,
        "stopped at the third identical answer, not {}",
        report.tool_calls
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_repetition_stop_does_not_wait_for_a_slow_event_consumer() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "same").unwrap();
    let model = Scripted::new(|req| {
        call(
            &format!("c{}", tool_results(req)),
            "read_file",
            serde_json::json!({"path": "a.txt"}),
        )
    });
    let mut bounded = options();
    bounded.max_repeated_results = Some(3);
    // A consumer that falls behind, as any task does on a loaded machine: the
    // stop is decided before the next model request, not when events are read.
    bounded.observer = Some(Arc::new(|_| {
        std::thread::sleep(std::time::Duration::from_millis(40));
    }));
    let mut worker = Worker::start(dir.path(), model, bounded).unwrap();

    let report = worker.send("look at a.txt").await.unwrap();

    assert!(report.stopped_by_repetition);
    assert_eq!(
        report.tool_calls, 3,
        "the request after the third identical answer is the one that is stopped"
    );
}

#[tokio::test]
async fn a_call_repeated_with_only_cosmetic_differences_is_the_same_call_and_the_stop_is_labelled()
{
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "same").unwrap();
    // The same read each time, but with another `description`, another key
    // order and padded whitespace in the path argument: one call to sven's
    // identity. sven's own note on a repeated result changes the result text,
    // which the count must look through.
    let model = Scripted::new(|req| {
        let n = tool_results(req);
        let args = if n.is_multiple_of(2) {
            serde_json::json!({"path": "a.txt", "description": format!("look {n}")})
        } else {
            serde_json::json!({"description": "again", "path": "a.txt"})
        };
        call(&format!("c{n}"), "read_file", args)
    });
    let mut bounded = options();
    bounded.max_repeated_results = Some(3);
    let mut worker = Worker::start(dir.path(), model, bounded).unwrap();

    let report = worker.send("look at a.txt").await.unwrap();

    assert!(report.stopped_by_repetition);
    assert_eq!(report.tool_calls, 3);
    let trajectory = serde_json::to_string(&report.solution.trajectory).unwrap();
    assert!(
        trajectory.contains("repetition stop: read_file"),
        "the aborted turn names the reason: {trajectory}"
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

#[tokio::test]
async fn a_worker_asked_for_the_policy_trace_keeps_the_sampled_tokens_in_its_trajectory() {
    fn traced(req: &CompletionRequest) -> Vec<ResponseEvent> {
        let mut events = if req.want_logprobs {
            vec![
                ResponseEvent::PromptTokens { ids: vec![1, 2, 3] },
                ResponseEvent::Tokens {
                    ids: vec![7, 8],
                    logprobs: vec![-0.5, -0.25],
                },
            ]
        } else {
            Vec::new()
        };
        events.extend(text("done"));
        events
    }
    let dir = tempfile::tempdir().unwrap();
    let plain = Scripted::new(traced);
    let mut worker = Worker::start(dir.path(), plain.clone(), options()).unwrap();
    worker.send("say done").await.unwrap();
    assert!(
        !plain.seen.lock().unwrap()[0].want_logprobs,
        "the trace is asked for only when wanted"
    );

    let model = Scripted::new(traced);
    let mut wanting = options();
    wanting.solve.want_logprobs = true;
    let mut worker = Worker::start(dir.path(), model.clone(), wanting).unwrap();
    worker.send("say done").await.unwrap();
    assert!(model.seen.lock().unwrap()[0].want_logprobs);
    let trajectory = serde_json::to_value(worker.trajectory()).unwrap();
    let metrics = trajectory["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|step| step.get("metrics").filter(|m| m["logprobs"].is_array()))
        .unwrap_or_else(|| panic!("no step carries the sampled tokens: {trajectory}"));
    assert_eq!(metrics["completion_token_ids"], serde_json::json!([7, 8]));
    assert_eq!(metrics["logprobs"], serde_json::json!([-0.5, -0.25]));
    assert_eq!(metrics["prompt_token_ids"], serde_json::json!([1, 2, 3]));
}

/// The calls of a run and their results as the trajectory records them: each
/// call's id with how many results name it.
fn calls_and_results(trajectory: &atif::Trajectory) -> Vec<(String, usize)> {
    let steps = &trajectory.steps;
    steps
        .iter()
        .flat_map(|s| s.tool_calls.iter().flatten())
        .map(|c| {
            let results = steps
                .iter()
                .filter_map(|s| s.observation.as_ref())
                .flat_map(|o| o.results.iter())
                .filter(|r| r.source_call_id.as_deref() == Some(c.tool_call_id.as_str()))
                .count();
            (c.tool_call_id.clone(), results)
        })
        .collect()
}

/// Two calls in one turn and a third after them: the trajectory records
/// every call once, each with exactly one result that names it, in the order
/// they were made, and every call took effect.
#[tokio::test]
async fn several_calls_are_each_recorded_once_with_the_one_result_that_answers_it() {
    let dir = tempfile::tempdir().unwrap();
    let model = Scripted::new(|req| match tool_results(req) {
        0 => vec![
            ResponseEvent::ToolCall {
                index: 0,
                id: "c1".into(),
                name: "write_file".into(),
                arguments: serde_json::json!({"path": "a.txt", "text": "a", "append": false})
                    .to_string(),
            },
            ResponseEvent::ToolCall {
                index: 1,
                id: "c2".into(),
                name: "write_file".into(),
                arguments: serde_json::json!({"path": "b.txt", "text": "b", "append": false})
                    .to_string(),
            },
            ResponseEvent::Done,
        ],
        2 => call(
            "c3",
            "write_file",
            serde_json::json!({"path": "c.txt", "text": "c", "append": false}),
        ),
        _ => text("wrote all three"),
    });
    let mut worker = Worker::start(dir.path(), model, options()).unwrap();
    let report = worker.send("write three files").await.unwrap();

    assert_eq!(report.solution.conclusion, RunConclusion::Success);
    assert_eq!(report.tool_calls, 3);
    assert_eq!(
        calls_and_results(&report.solution.trajectory),
        [("c1".into(), 1), ("c2".into(), 1), ("c3".into(), 1)]
    );
    for (file, body) in [("a.txt", "a"), ("b.txt", "b"), ("c.txt", "c")] {
        assert_eq!(
            std::fs::read_to_string(dir.path().join(file)).unwrap(),
            body
        );
    }
}

/// A call whose arguments are not JSON fails, and says so: its one result is
/// an error the model is told of, no file is written, and no result is made
/// up for it.
#[tokio::test]
async fn a_call_with_arguments_that_do_not_parse_fails_openly_and_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let model = Scripted::new(|req| {
        if has_tool_result(req) {
            text("that failed")
        } else {
            vec![
                ResponseEvent::ToolCall {
                    index: 0,
                    id: "c1".into(),
                    name: "write_file".into(),
                    arguments: "{\"path\": \"a.txt\", \"text\": ".into(),
                },
                ResponseEvent::Done,
            ]
        }
    });
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut observed = options();
    observed.observer = Some(observed_into(&log));
    let mut worker = Worker::start(dir.path(), model, observed).unwrap();
    let report = worker.send("write a file").await.unwrap();

    assert_eq!(
        calls_and_results(&report.solution.trajectory),
        [("c1".into(), 1)]
    );
    assert!(!dir.path().join("a.txt").exists(), "nothing was written");
    let failed = log
        .lock()
        .unwrap()
        .iter()
        .any(|e| matches!(e, SessionEvent::ToolCallFinished { is_error: true, .. }));
    assert!(failed, "the call is reported as failed");
}

/// A model that makes one call, then never answers in time.
struct Stalling;

#[async_trait::async_trait]
impl ModelProvider for Stalling {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "stalling-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let events = if has_tool_result(&req) {
            tokio::time::sleep(Duration::from_secs(30)).await;
            text("too late")
        } else {
            call(
                "c1",
                "write_file",
                serde_json::json!({"path": "half.txt", "text": "half", "append": false}),
            )
        };
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

/// A run its deadline ends is a timeout, not a success or an error, and keeps
/// what it did before: the call it made, with its result, is in the
/// trajectory, and its effect is in the tree, for the grader to judge.
#[tokio::test]
async fn a_run_its_deadline_ends_is_a_timeout_that_keeps_its_partial_trajectory() {
    let dir = tempfile::tempdir().unwrap();
    let short = WorkOptions::new(SolveOptions::new(Duration::from_secs(2)));
    let mut worker = Worker::start(dir.path(), Arc::new(Stalling), short).unwrap();
    let report = worker.send("write half a file").await.unwrap();

    assert_eq!(report.solution.conclusion, RunConclusion::Timeout);
    assert_eq!(
        calls_and_results(&report.solution.trajectory),
        [("c1".into(), 1)]
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("half.txt")).unwrap(),
        "half"
    );
    assert_ne!(report.solution.final_output.as_deref(), Some("too late"));
}

const LONG_RUN_CALLS: usize = 80;

fn writes_many(req: &CompletionRequest) -> Vec<ResponseEvent> {
    let done = tool_results(req);
    if done < LONG_RUN_CALLS {
        call(
            &format!("c{done}"),
            "write_file",
            serde_json::json!({"path": format!("f{done}.txt"), "text": format!("{done}"), "append": false}),
        )
    } else {
        text("wrote them all")
    }
}

/// A long run keeps all of itself: every one of its many calls is in the
/// trajectory once, in order, with its one result, and each took effect;
/// nothing is dropped to keep the record short.
#[tokio::test]
async fn a_long_run_keeps_every_call_and_result_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut worker = Worker::start(dir.path(), Scripted::new(writes_many), options()).unwrap();
    let report = worker.send("write many files").await.unwrap();

    assert_eq!(report.solution.conclusion, RunConclusion::Success);
    let expected: Vec<(String, usize)> =
        (0..LONG_RUN_CALLS).map(|n| (format!("c{n}"), 1)).collect();
    assert_eq!(calls_and_results(&report.solution.trajectory), expected);
    assert_eq!(report.dropped_events, 0);
    for n in 0..LONG_RUN_CALLS {
        assert_eq!(
            std::fs::read_to_string(dir.path().join(format!("f{n}.txt"))).unwrap(),
            n.to_string()
        );
    }
}
