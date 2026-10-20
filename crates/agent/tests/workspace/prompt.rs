// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that change a repository
// without ever seeing the tests that judge them, for its clients. If your
// team needs expertise in measuring what a code-editing agent really did,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: which system prompt a solver of a repository works under, and what
//! the attempt records of it: the default coding-agent prompt, a version the
//! protocol names, and a prompt the model brings.

use super::*;
use splinter_core::prompt::{
    CodingPrompt, CODING_AGENT_2_PROMPT, CODING_AGENT_2_PROMPT_ID, CODING_AGENT_PROMPT,
    CODING_AGENT_PROMPT_ID,
};

/// The text of the system turn of the last request the model was sent.
fn system_turn(model: &Scripted) -> String {
    let seen = model.seen.lock().unwrap();
    let messages = &seen.last().unwrap().messages;
    let system = messages.iter().find(|m| m.role == Role::System).unwrap();
    system.as_text().unwrap().to_string()
}

/// One episode under `asked`; its trajectory's solve profile.
async fn solve_under(asked: WorkOptions, model: Arc<Scripted>) -> serde_json::Value {
    solve_under_protocol(asked, model, protocol(0)).await
}

/// One episode under `asked` and `protocol`; its trajectory's solve profile.
async fn solve_under_protocol(
    asked: WorkOptions,
    model: Arc<Scripted>,
    protocol: WorkProtocol,
) -> serde_json::Value {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let (task, archive) = (fixture.task(&environment), &fixture.archive);
    let solved = solve_in_workspace(&task, &environment, archive, &[], model, asked, &protocol)
        .await
        .unwrap();
    let extra = solved.solution.trajectory.agent.extra.unwrap();
    extra[SOLVE_PROFILE_KEY].clone()
}

#[tokio::test]
async fn an_episode_runs_under_the_coding_agent_prompt_and_records_its_profile() {
    let model = Scripted::new(edits_then_stops);
    let mut asked = options();
    asked.max_repeated_results = Some(3);
    asked.solve.sampling = Some(serde_json::json!({"top_k": 20, "seed": 7}));

    let profile = solve_under(asked, model.clone()).await;

    assert_eq!(system_turn(&model), CODING_AGENT_PROMPT);
    assert_eq!(profile["system_prompt"]["id"], CODING_AGENT_PROMPT_ID);
    assert_eq!(
        profile["system_prompt"]["digest"],
        serde_json::json!(Digest::of(CODING_AGENT_PROMPT.as_bytes()))
    );
    assert_eq!(profile["tools"]["max_timeout_secs"], 1800);
    assert_eq!(profile["agent"]["note_repeated_results"], true);
    assert_eq!(profile["max_repeated_results"], 3);
    assert_eq!(
        profile["sampling"],
        serde_json::json!({"top_k": 20, "seed": 7})
    );
}

#[tokio::test]
async fn a_protocol_can_name_the_second_coding_agent_prompt_and_the_record_says_which() {
    let model = Scripted::new(edits_then_stops);
    let chosen = WorkProtocol {
        prompt: CodingPrompt::V2,
        ..protocol(0)
    };

    let profile = solve_under_protocol(options(), model.clone(), chosen).await;

    assert_eq!(system_turn(&model), CODING_AGENT_2_PROMPT);
    assert_eq!(profile["system_prompt"]["id"], CODING_AGENT_2_PROMPT_ID);
    assert_eq!(
        profile["system_prompt"]["digest"],
        serde_json::json!(Digest::of(CODING_AGENT_2_PROMPT.as_bytes()))
    );
}

#[tokio::test]
async fn a_prompt_the_model_brings_is_kept_and_recorded_without_the_coding_agent_id() {
    let model = Scripted::new(edits_then_stops);
    let mut asked = options();
    asked.solve.system = Some("You are Maren.".into());

    let profile = solve_under(asked, model.clone()).await;

    assert_eq!(system_turn(&model), "You are Maren.");
    assert!(profile["system_prompt"]["id"].is_null());
}
