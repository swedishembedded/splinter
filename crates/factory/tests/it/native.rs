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

#[test]
fn portable_sft_rebuilds_a_conversation_from_the_trajectory_alone() {
    // An agent that leaves no request log (Claude Code) has only its ATIF
    // trajectory: the conversation is rebuilt from it, every action
    // supervised, its instruction the trajectory's own first user turn.
    let trajectory: atif::Trajectory =
        serde_json::from_slice(&fixture("agent/trajectory.json")).unwrap();
    let portable = splinter_data::portable_record(&trajectory).expect("the trial projects");
    let RecordBody::Chat { messages, .. } = &portable.record else {
        panic!("a chat record")
    };
    let agent_steps = trajectory
        .steps
        .iter()
        .filter(|s| s.source == atif::StepOrigin::Agent)
        .count();
    assert_eq!(portable.actions, agent_steps);
    assert_eq!(messages.iter().filter(|m| m.train).count(), agent_steps);
    let first_user = messages.iter().find(|m| m.role == "user").unwrap();
    assert!(
        first_user.content.starts_with("# "),
        "{}",
        first_user.content
    );
}

/// The `sven-trial` fixture with the results of `marked` shown to the model
/// as errors (Sven sends an `isError` result as `error: ` and the output; its
/// trajectory keeps the output as the tool returned it), and the results of
/// `printed` printed by the tool itself starting with `error: ` (a successful
/// result whose own text says so: the trajectory and the request agree).
fn trial_with_results(marked: &[&str], printed: &[&str]) -> splinter_data::NativeRecord {
    let mut log = String::from_utf8(fixture("agent/requests.jsonl")).unwrap();
    let mut trajectory: serde_json::Value =
        serde_json::from_slice(&fixture("agent/trajectory.json")).unwrap();
    for id in marked.iter().chain(printed) {
        let sent = format!("\"tool_call_id\":\"{id}\",\"content\":\"");
        assert!(log.contains(&sent), "the fixture answers {id}");
        log = log.replace(&sent, &format!("{sent}error: "));
    }
    for step in trajectory["steps"].as_array_mut().unwrap() {
        let results = step
            .get_mut("observation")
            .and_then(|o| o.get_mut("results"))
            .and_then(serde_json::Value::as_array_mut);
        for result in results.into_iter().flatten() {
            if printed.contains(&result["source_call_id"].as_str().unwrap_or_default()) {
                let text = result["content"].as_str().unwrap().to_owned();
                result["content"] = serde_json::Value::String(format!("error: {text}"));
            }
        }
    }
    let trajectory: atif::Trajectory = serde_json::from_value(trajectory).unwrap();
    let calls: Vec<_> = read_request_log(log.as_bytes())
        .unwrap()
        .into_iter()
        .filter_map(|r| r.call)
        .collect();
    native_record(&trajectory, &calls).expect("the trial projects")
}

const FIRST_CALL: &str = "call_de01edfe6dd94b92ba12a09923d01212";
const THIRD_CALL: &str = "call_7600a95fe9b243f4a0098f5686f24b2c";

fn supervised_calls(record: &RecordBody) -> Vec<Option<String>> {
    let RecordBody::Chat { messages, .. } = record else {
        panic!("a chat record")
    };
    messages
        .iter()
        .filter(|m| m.role == "assistant" && m.train)
        .map(|m| m.tool_calls.first().and_then(|c| c.id.clone()))
        .collect()
}

#[test]
fn an_action_whose_every_call_was_answered_with_an_error_is_kept_as_context_and_nothing_else_is() {
    let mut native = trial_with_results(&[FIRST_CALL, THIRD_CALL], &[]);
    assert_eq!(
        native.failed_calls,
        [format!("s2-{FIRST_CALL}"), format!("s4-{THIRD_CALL}")]
            .into_iter()
            .collect()
    );
    let before = supervised_calls(&native.record);
    assert_eq!(before.len(), 7, "six calling actions and the final answer");
    let masked = splinter_data::mask_failed_actions(&mut native.record, &native.failed_calls);
    assert_eq!(masked, 2);
    let after = supervised_calls(&native.record);
    let expected: Vec<Option<String>> = before
        .iter()
        .filter(|id| {
            !native
                .failed_calls
                .contains(id.as_deref().unwrap_or_default())
        })
        .cloned()
        .collect();
    assert_eq!(
        after, expected,
        "the recovery and the final answer stay supervised"
    );
    assert_eq!(after.last(), Some(&None), "the final answer is supervised");
    let RecordBody::Chat { messages, .. } = &native.record else {
        panic!("a chat record")
    };
    let masked_turns: Vec<_> = messages
        .iter()
        .filter(|m| m.role == "assistant" && !m.train)
        .collect();
    assert_eq!(masked_turns.len(), 2, "kept in the context");
    assert!(masked_turns.iter().all(|m| m.loss_weight.is_none()));
}

#[test]
fn a_successful_result_whose_own_text_starts_like_an_error_is_no_error() {
    // The marker alone is no evidence: a tool's output may itself begin
    // `error: ` (a command's own message) while the call succeeded.
    let native = trial_with_results(&[THIRD_CALL], &[FIRST_CALL]);
    assert_eq!(
        native.failed_calls,
        [format!("s4-{THIRD_CALL}")].into_iter().collect()
    );
    let clean = trial_with_results(&[], &[]);
    assert!(clean.failed_calls.is_empty());
    let mut record = clean.record.clone();
    assert_eq!(
        splinter_data::mask_failed_actions(&mut record, &clean.failed_calls),
        0
    );
    assert_eq!(record, clean.record, "no error leaves the record as it was");
}
