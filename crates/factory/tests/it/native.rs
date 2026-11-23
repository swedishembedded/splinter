// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! `native-sft` keeps the conversation exactly as the agent's model was sent
//! it - its own system prompt and tools - and supervises every action the
//! agent took, on a real Sven trial (request log and ATIF trajectory, kept as
//! recorded).

use std::path::Path;

use splinter_data::request_log::read_request_log;
use splinter_data::{native_record, RecordBody};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/it/fixtures/sven-trial")
            .join(name),
    )
    .unwrap()
}

#[test]
fn a_real_trial_becomes_one_record_with_every_action_supervised_under_its_own_prompt() {
    let trajectory: atif::Trajectory =
        serde_json::from_slice(&fixture("agent/trajectory.json")).unwrap();
    let log = read_request_log(&fixture("agent/requests.jsonl")).unwrap();
    let calls: Vec<_> = log.iter().filter_map(|r| r.call.clone()).collect();
    assert_eq!(calls.len(), 7);
    let native = native_record(&trajectory, &calls).expect("the trial projects");
    let RecordBody::Chat {
        messages, tools, ..
    } = &native.record
    else {
        panic!("a chat record")
    };
    let agent_steps = trajectory
        .steps
        .iter()
        .filter(|s| s.source == atif::StepOrigin::Agent)
        .count();
    assert_eq!(
        native.actions, agent_steps,
        "every action of the agent is supervised"
    );
    assert_eq!(messages.iter().filter(|m| m.train).count(), agent_steps);
    assert!(messages
        .iter()
        .filter(|m| m.train)
        .all(|m| m.role == "assistant"));
    // The system prompt is the one Sven sent, not a replacement.
    let sent = calls[0].messages[0]["content"].as_str().unwrap();
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[0].content, sent);
    assert!(!tools.is_empty(), "the tools the agent was offered");
    assert!(
        native.tools.iter().all(|t| t == "vim-vim"),
        "{:?}",
        native.tools
    );
}

#[test]
fn calls_that_are_not_the_trajectorys_are_refused_rather_than_guessed() {
    let trajectory: atif::Trajectory =
        serde_json::from_slice(&fixture("agent/trajectory.json")).unwrap();
    let log = read_request_log(&fixture("agent/requests.jsonl")).unwrap();
    let mut calls: Vec<_> = log.iter().filter_map(|r| r.call.clone()).collect();
    calls.pop();
    assert!(native_record(&trajectory, &calls).is_err());
}

#[test]
fn a_trial_whose_runtime_files_each_requests_tokens_as_a_step_of_its_own_projects_every_action() {
    // Sven 5.9 files a request's token account as an agent step that says,
    // reasons and calls nothing, beside the step of the reply: three model
    // calls, five agent steps. The empty steps are no actions.
    let read = |name: &str| {
        std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/it/fixtures/sven-5.9-trial")
                .join(name),
        )
        .unwrap()
    };
    let trajectory: atif::Trajectory =
        serde_json::from_slice(&read("agent/trajectory.json")).unwrap();
    let calls: Vec<_> = read_request_log(&read("agent/requests.jsonl"))
        .unwrap()
        .into_iter()
        .filter_map(|r| r.call)
        .collect();
    assert_eq!(calls.len(), 3);
    assert_eq!(
        trajectory
            .steps
            .iter()
            .filter(|s| s.source == atif::StepOrigin::Agent)
            .count(),
        5
    );
    let native = native_record(&trajectory, &calls).expect("the trial projects");
    assert_eq!(native.actions, 3);
    let RecordBody::Chat { messages, .. } = &native.record else {
        panic!("a chat record")
    };
    assert_eq!(messages.iter().filter(|m| m.train).count(), 3);
}
