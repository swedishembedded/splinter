// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements confined, reproducible agent workspaces for
// its clients. If your team needs expertise in running coding agents safely,
// you can procure our services by sending an email to info@swedishembedded.com.

//! Spec: workspace episodes at their limits: trials of one task run at once
//! keep to their own trees, and a command past its time limit is stopped and
//! the model told so.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::*;

fn writes_one(req: &CompletionRequest) -> Vec<ResponseEvent> {
    match tool_results(req) {
        0 => call(
            "c0",
            "write_file",
            serde_json::json!({"path": "app.txt", "text": "one\n", "append": false}),
        ),
        _ => text("done"),
    }
}

fn writes_two(req: &CompletionRequest) -> Vec<ResponseEvent> {
    match tool_results(req) {
        0 => call(
            "c0",
            "write_file",
            serde_json::json!({"path": "app.txt", "text": "two\n", "append": false}),
        ),
        _ => text("done"),
    }
}

/// Two trials of one task run at once each work in a fresh copy of its
/// snapshot: each patch is that trial's own edit against the snapshot, and
/// neither sees the other's.
#[tokio::test]
async fn trials_of_one_task_run_at_once_each_change_only_their_own_tree() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let protocol = protocol(0);
    let (first, second) = tokio::join!(
        solve(
            &fixture,
            &environment,
            &[],
            Scripted::new(writes_one),
            &protocol
        ),
        solve(
            &fixture,
            &environment,
            &[],
            Scripted::new(writes_two),
            &protocol
        ),
    );
    let diff = |solved: WorkspaceSolution| String::from_utf8(solved.patch.diff).unwrap();
    let (one, two) = (diff(first.unwrap()), diff(second.unwrap()));
    assert!(
        one.contains("-broken") && one.contains("+one") && !one.contains("two"),
        "{one}"
    );
    assert!(
        two.contains("-broken") && two.contains("+two") && !two.contains("one"),
        "{two}"
    );
}

fn sleeps_then_answers(req: &CompletionRequest) -> Vec<ResponseEvent> {
    match tool_results(req) {
        0 => call(
            "c0",
            "shell",
            serde_json::json!({"description": "wait", "shell_command": "sleep 30"}),
        ),
        _ => text("done"),
    }
}

/// A command that outlives its time limit is stopped there, and the result
/// the model reads says so (`[stopped at the time limit]`): the run goes
/// on, and the time limit is reported, not turned into an invented output
/// or a failure of the run.
#[tokio::test]
async fn a_command_past_its_time_limit_is_stopped_and_the_model_is_told_so() {
    let fixture = Fixture::new();
    let sandbox = ProcessSandbox::new(
        fixture.root.path().join("calls"),
        BTreeMap::new(),
        &[],
        Limits {
            wall_time_ms: 300,
            ..Limits::default()
        },
    );
    let environment = fixture.environment_in(Arc::new(sandbox));
    let model = Scripted::new(sleeps_then_answers);
    let started = std::time::Instant::now();
    let solved = solve(&fixture, &environment, &[], model.clone(), &protocol(0))
        .await
        .unwrap();
    assert!(
        started.elapsed() < std::time::Duration::from_secs(20),
        "the limit stopped it"
    );
    assert_eq!(solved.solution.conclusion, RunConclusion::Success);
    let seen = model.seen.lock().unwrap();
    let told = seen
        .last()
        .unwrap()
        .messages
        .iter()
        .any(|m| format!("{:?}", m.content).contains("stopped at the time limit"));
    assert!(told, "the model reads that the command was stopped");
}
