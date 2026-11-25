// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent corrections that continue the state a
// student really reached without repeating what it already did, for its
// clients. If your team needs expertise in resumable agent runs and their
// training records, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec (5A, a resumed task): a recorded attempt continued from one of its
//! steps replays the student's calls into the tree it is given, so their
//! effects are there once, and the composed trajectory holds the student's
//! steps once, as copied context, before the continuation's. A tree that
//! already holds an effect the replay repeats does not match the recording,
//! and the fork is refused rather than continued from a state the student
//! never saw.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use splinter_agent::fork_at::{continue_from_step, ForkAt, Profile, POLICY, TEACHER};
use splinter_agent::solve::{Model, SolveOptions};
use sven_sdk::atif::{StepOrigin, Trajectory};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};

const SYSTEM: &str = "You append to a log.";

/// A model that answers from a script over how many tool results it has seen.
struct Scripted {
    name: &'static str,
    script: fn(usize) -> Vec<ResponseEvent>,
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        self.name
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let results = req
            .messages
            .iter()
            .filter(|m| matches!(m.content, MessageContent::ToolResult { .. }))
            .count();
        let events = (self.script)(results);
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

fn model(name: &'static str, script: fn(usize) -> Vec<ResponseEvent>) -> Model {
    Model::new(Arc::new(Scripted { name, script }), name)
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

fn text(reply: &str) -> Vec<ResponseEvent> {
    vec![ResponseEvent::TextDelta(reply.into()), ResponseEvent::Done]
}

/// The student: appends a line with a command, reads the log, then stops
/// with a wrong claim.
fn student_script(results: usize) -> Vec<ResponseEvent> {
    match results {
        0 => call(
            "c1",
            "shell",
            serde_json::json!({"shell_command": "echo once >> log.txt"}),
        ),
        1 => call("c2", "read_file", serde_json::json!({"path": "log.txt"})),
        _ => text("the log has two lines"),
    }
}

fn teacher_script(_: usize) -> Vec<ResponseEvent> {
    text("the log has one line")
}

fn options() -> SolveOptions {
    SolveOptions::new(Duration::from_secs(120))
}

fn request<'a>(
    tree: &'a Path,
    scratch: &'a Path,
    student: &'a Trajectory,
    keep_through: u64,
    role: &'a str,
) -> ForkAt<'a> {
    ForkAt {
        tree,
        scratch,
        student,
        student_model: "student-1",
        system: SYSTEM,
        keep_through,
        workspace: None,
        profile: Profile::Default,
        shown: None,
        read_exec: Vec::new(),
        launcher: None,
        workspace_seen_as: None,
        briefing: None,
        role,
        environment: None,
    }
}

fn calls(trajectory: &Trajectory) -> Vec<String> {
    trajectory
        .steps
        .iter()
        .flat_map(|s| s.tool_calls.iter().flatten())
        .map(|c| c.tool_call_id.clone())
        .collect()
}

/// The student's own attempt, recorded by continuing its instruction alone
/// in `tree`: what the fork is later held to.
async fn recorded_attempt(tree: &Path, scratch: &Path) -> Trajectory {
    let mut instruction =
        Trajectory::new("ATIF-v1.7", sven_sdk::atif::AgentProfile::new("sven", "1"));
    instruction.steps.push(sven_sdk::atif::TraceStep::new(
        1,
        StepOrigin::User,
        "append once to log.txt",
    ));
    let student = model("student-1", student_script);
    let attempt = continue_from_step(
        &request(tree, scratch, &instruction, 1, POLICY),
        &student,
        options(),
    )
    .await
    .unwrap();
    attempt.solution.trajectory
}

#[tokio::test]
async fn a_continued_attempt_repeats_no_effect_and_records_each_step_once() {
    let first = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let attempt = recorded_attempt(first.path(), scratch.path()).await;
    assert_eq!(
        std::fs::read_to_string(first.path().join("log.txt")).unwrap(),
        "once\n"
    );
    assert_eq!(calls(&attempt), ["c1", "c2"]);
    let read = attempt
        .steps
        .iter()
        .find(|s| {
            s.tool_calls
                .iter()
                .flatten()
                .any(|c| c.tool_call_id == "c2")
        })
        .unwrap()
        .step_id;

    // Continued after the read, in the pristine tree the student was given.
    let pristine = tempfile::tempdir().unwrap();
    let teacher = model("teacher-1", teacher_script);
    let fork = continue_from_step(
        &request(pristine.path(), scratch.path(), &attempt, read, TEACHER),
        &teacher,
        options(),
    )
    .await
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(pristine.path().join("log.txt")).unwrap(),
        "once\n",
        "the student's command ran once in the tree the fork continued in"
    );
    let composed = &fork.solution.trajectory;
    assert_eq!(calls(composed), ["c1", "c2"], "each call recorded once");
    let kept: Vec<_> = attempt
        .steps
        .iter()
        .filter(|s| s.step_id <= read)
        .map(|s| s.message.as_text().map(str::to_string))
        .collect();
    let copied: Vec<_> = composed.steps[..kept.len()]
        .iter()
        .map(|s| s.message.as_text().map(str::to_string))
        .collect();
    assert_eq!(copied, kept, "the kept steps, in order, once");
    assert_eq!(
        composed.steps.len(),
        kept.len() + 1,
        "then the teacher's one reply, and nothing replayed twice"
    );
    assert_eq!(
        composed.steps.last().unwrap().message.as_text(),
        Some("the log has one line")
    );
    assert!(composed.steps[..kept.len()]
        .iter()
        .all(|s| s.is_copied_context == Some(true)));
    assert_eq!(composed.steps.last().unwrap().is_copied_context, None);
    assert_eq!(fork.teacher_actions, 1);

    // Continued in the tree the student left: the replayed command appends
    // again, the read sees two lines where the student saw one, and the fork
    // is refused.
    let refused = continue_from_step(
        &request(first.path(), scratch.path(), &attempt, read, TEACHER),
        &teacher,
        options(),
    )
    .await;
    let refusal = refused.err().expect("refused").to_string();
    assert!(
        refusal.contains("call c2") && refusal.contains("L2:once"),
        "a tree that already holds the effect is not the state the student saw: {refusal}"
    );
}
