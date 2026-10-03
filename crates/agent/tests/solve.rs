// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent solvers whose every run is replayable
// evidence, for its clients. If your team needs expertise in agent
// environments or learning from agent experience, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a task is solved through sven in exactly the environment it names.
//! Closed-book offers the model no tools at all; a runtime environment
//! offers exactly one, `run_code`, whose real output reaches the model and
//! the trajectory. A task is never solved in an environment other than the
//! one it records. A model's stream idle limit reaches sven: a first chunk
//! slower than the limit fails the run, one within it does not. An
//! experience's code calls replay in its environment to the results it
//! recorded; a recorded result that differs is named, field by field, and
//! a replay anywhere but the recorded environment is refused. Every
//! request a solve sends carries Splinter's own system prompt as its one
//! system turn, whatever the environment, with nothing appended to it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;

use splinter_agent::replay::{replay, CallReplay, ReplayError};
use splinter_agent::solve::{
    solve, solve_prompted, SolveError, SolveOptions, RUN_CODE, SYSTEM_PROMPT,
};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Provenance, Task};
use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, RuntimeEnvironment, RuntimeRegistry,
};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream, Role,
};
use sven_sdk::RunConclusion;

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

    /// The tool names offered on each request.
    fn offered(&self) -> Vec<Vec<String>> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.tools.iter().map(|t| t.name.clone()).collect())
            .collect()
    }
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "solver-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let events = (self.script)(&req);
        self.seen.lock().unwrap().push(req);
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

/// A model whose first chunk arrives only after `silence`, as a local
/// model's prefill does.
struct SlowFirstChunk {
    silence: Duration,
}

#[async_trait::async_trait]
impl ModelProvider for SlowFirstChunk {
    fn name(&self) -> &str {
        "slow"
    }
    fn model_name(&self) -> &str {
        "prefill-1"
    }
    async fn complete(&self, _req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let silence = self.silence;
        let first = futures::stream::once(async move {
            tokio::time::sleep(silence).await;
            Ok(ResponseEvent::TextDelta("42".into()))
        });
        Ok(Box::pin(
            first.chain(futures::stream::iter([Ok(ResponseEvent::Done)])),
        ))
    }
}

fn text(reply: &str) -> Vec<ResponseEvent> {
    vec![ResponseEvent::TextDelta(reply.into()), ResponseEvent::Done]
}

/// The text of the last tool result in the request, if there is one.
fn last_tool_result(req: &CompletionRequest) -> Option<String> {
    req.messages.iter().rev().find_map(|m| match &m.content {
        MessageContent::ToolResult { content, .. } => content.as_text().map(str::to_string),
        _ => None,
    })
}

fn options() -> SolveOptions {
    SolveOptions::new(Duration::from_secs(120))
}

fn task(environment: Environment, instruction: &str) -> Task {
    Task::new("compute", vec![], environment, instruction, vec![]).unwrap()
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn python(scratch: &Scratch) -> ResolvedEnvironment {
    std::fs::create_dir_all(&scratch.0).unwrap();
    let sandbox = ProcessSandbox::new(&scratch.0, BTreeMap::new(), &[], Limits::default());
    ResolvedEnvironment::Runtime(
        RuntimeEnvironment::new(&RuntimeRegistry::builtin(), "python3", Arc::new(sandbox)).unwrap(),
    )
}

#[tokio::test]
async fn a_closed_book_solve_offers_the_model_no_tools() {
    let model = Scripted::new(|_| text("42"));
    let task = task(Environment::closed_book(), "What is six times seven?");
    let solution = solve(
        &task,
        &ResolvedEnvironment::ClosedBook,
        model.clone(),
        options(),
    )
    .await
    .unwrap();
    let offered = model.offered();
    assert!(!offered.is_empty(), "the model was asked");
    assert!(
        offered.iter().all(Vec::is_empty),
        "closed-book offers no tools: {offered:?}"
    );
    assert_eq!(solution.conclusion, RunConclusion::Success);
    assert_eq!(solution.final_output.as_deref(), Some("42"));
    let experience = solution
        .into_experience(
            task.clone(),
            Provenance::new(
                "scripted/solver-1",
                &FixedClock::new("2026-09-30T08:00:00.000Z"),
            ),
        )
        .unwrap();
    assert_eq!(experience.environment, Environment::closed_book());
}

#[tokio::test]
async fn a_runtime_solve_runs_the_models_code_and_answers_with_its_output() {
    let scratch = Scratch(
        std::env::temp_dir().join(format!("splinter-solve-runtime-{}", std::process::id())),
    );
    let environment = python(&scratch);
    let task = task(
        environment.record().unwrap(),
        "Compute six times seven with Python.",
    );
    let model = Scripted::new(|req| match last_tool_result(req) {
        None => vec![
            ResponseEvent::ToolCall {
                index: 0,
                id: "call-1".into(),
                name: RUN_CODE.into(),
                arguments: serde_json::json!({"code": "print(6 * 7)"}).to_string(),
            },
            ResponseEvent::Done,
        ],
        Some(result) => {
            let result: serde_json::Value = serde_json::from_str(&result).unwrap();
            text(&format!(
                "The program printed {}",
                result["stdout"].as_str().unwrap().trim()
            ))
        }
    });
    let solution = solve(&task, &environment, model.clone(), options())
        .await
        .unwrap();

    let offered = model.offered();
    assert_eq!(offered.len(), 2, "a call, then the answer: {offered:?}");
    assert!(
        offered.iter().all(|tools| tools == &[RUN_CODE.to_string()]),
        "exactly one tool: {offered:?}"
    );
    assert_eq!(solution.conclusion, RunConclusion::Success);
    assert_eq!(
        solution.final_output.as_deref(),
        Some("The program printed 42")
    );

    let steps = &solution.trajectory.steps;
    let called = steps
        .iter()
        .flat_map(|s| s.tool_calls.iter().flatten())
        .find(|c| c.function_name == RUN_CODE)
        .expect("the trajectory records the call");
    assert_eq!(called.arguments["code"], "print(6 * 7)");
    let observed = steps
        .iter()
        .filter_map(|s| s.observation.as_ref())
        .flat_map(|o| o.results.iter())
        .find(|r| r.source_call_id.as_deref() == Some(called.tool_call_id.as_str()))
        .expect("the trajectory records the call's result");
    let observed = serde_json::to_string(&observed.content).unwrap();
    assert!(observed.contains("42"), "{observed}");

    let experience = solution
        .into_experience(
            task.clone(),
            Provenance::new(
                "scripted/solver-1",
                &FixedClock::new("2026-09-30T08:00:00.000Z"),
            ),
        )
        .unwrap();
    assert_eq!(experience.environment, environment.record().unwrap());
    assert!(
        std::fs::read_dir(&scratch.0).unwrap().next().is_none(),
        "every call's directory was removed"
    );

    // Replayed in its own environment, the call reproduces what it
    // observed.
    let replayed = replay(&experience, &environment).unwrap();
    assert_eq!(replayed.calls.len(), 1);
    assert_eq!(replayed.calls[0].replay, CallReplay::Reproduced);
    assert!(replayed.reproduced());

    // A recorded result edited after the fact no longer replays.
    let mut edited = experience.clone();
    for entry in edited
        .trajectory
        .steps
        .iter_mut()
        .filter_map(|s| s.observation.as_mut())
        .flat_map(|o| o.results.iter_mut())
    {
        let text = entry.content.as_ref().unwrap().as_text().unwrap();
        entry.content = Some(sven_sdk::atif::MessageBody::text(text.replace("42", "41")));
    }
    let replayed = replay(&edited, &environment).unwrap();
    assert_eq!(
        replayed.calls[0].replay,
        CallReplay::Diverged {
            fields: vec!["stdout"]
        }
    );
    assert!(!replayed.reproduced());

    let elsewhere = replay(&experience, &ResolvedEnvironment::ClosedBook);
    assert!(
        matches!(elsewhere, Err(ReplayError::EnvironmentMismatch { .. })),
        "{elsewhere:?}"
    );
}

#[tokio::test]
async fn a_task_is_never_solved_in_another_environment() {
    let scratch = Scratch(
        std::env::temp_dir().join(format!("splinter-solve-mismatch-{}", std::process::id())),
    );
    let model = Scripted::new(|_| text("42"));
    let closed = task(Environment::closed_book(), "What is six times seven?");
    let refused = solve(&closed, &python(&scratch), model.clone(), options()).await;
    assert!(
        matches!(refused, Err(SolveError::EnvironmentMismatch { .. })),
        "{refused:?}"
    );
    assert!(model.offered().is_empty(), "the model was never asked");
}

#[tokio::test]
async fn the_stream_idle_limit_bounds_the_silence_before_a_first_chunk() {
    let task = task(Environment::closed_book(), "What is six times seven?");
    let model = Arc::new(SlowFirstChunk {
        silence: Duration::from_millis(2500),
    });
    let with_idle = |limit: Duration| {
        let mut options = options();
        options.stream_idle = Some(limit);
        options
    };

    let stale = solve(
        &task,
        &ResolvedEnvironment::ClosedBook,
        model.clone(),
        with_idle(Duration::from_secs(1)),
    )
    .await;
    let failed = match stale {
        Err(SolveError::Engine(e)) => e.to_string(),
        Ok(solution) => format!("{:?}: {:?}", solution.conclusion, solution.final_output),
        Err(e) => format!("{e:?}"),
    };
    assert!(
        failed.contains("idle"),
        "a first chunk slower than the limit fails the run: {failed}"
    );

    let patient = solve(
        &task,
        &ResolvedEnvironment::ClosedBook,
        model,
        with_idle(Duration::from_secs(10)),
    )
    .await
    .unwrap();
    assert_eq!(patient.conclusion, RunConclusion::Success);
    assert_eq!(patient.final_output.as_deref(), Some("42"));
}

#[tokio::test]
async fn every_request_runs_under_splinters_system_prompt() {
    let model = Scripted::new(|_| text("42"));
    let task = task(Environment::closed_book(), "What is six times seven?");
    solve(
        &task,
        &ResolvedEnvironment::ClosedBook,
        model.clone(),
        options(),
    )
    .await
    .unwrap();
    solve_prompted(
        &task,
        "Six sevens are forty-two.\n\nWhat is six times seven?",
        &ResolvedEnvironment::ClosedBook,
        model.clone(),
        options(),
    )
    .await
    .unwrap();
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "one request per solve");
    for request in seen.iter() {
        let system: Vec<&str> = request
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.as_text().unwrap_or_default())
            .collect();
        assert_eq!(system, [SYSTEM_PROMPT], "{:?}", request.messages);
        assert_eq!(request.messages[0].role, Role::System);
        assert_eq!(request.system_dynamic_suffix, None);
    }
}
