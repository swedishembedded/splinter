// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a trajectory the projection into training conversations cannot
//! represent is refused with the step and the reason, and one it can
//! represent (a dialogue, or a single instruction with tool calls) is not.

use atif::{
    AgentProfile, ContentSegment, MessageBody, ObservationEntry, StepObservation, StepOrigin,
    ToolInvocation, TraceStep, Trajectory,
};
use serde_json::json;
use splinter_data::{projection_refusal, Unprojectable};

fn trajectory(steps: Vec<TraceStep>) -> Trajectory {
    let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    trajectory.steps = steps;
    trajectory
}

fn user(step: u64, text: &str) -> TraceStep {
    TraceStep::new(step, StepOrigin::User, text)
}

fn say(step: u64, text: &str) -> TraceStep {
    TraceStep::new(step, StepOrigin::Agent, text)
}

fn call(step: u64, id: &str, name: &str, arguments: serde_json::Value, result: &str) -> TraceStep {
    let mut s = say(step, "");
    s.tool_calls = Some(vec![ToolInvocation::new(id, name).with_arguments(arguments)]);
    s.observation = Some(StepObservation::single(ObservationEntry::for_call(
        id, result,
    )));
    s
}

#[test]
fn a_dialogue_and_a_tool_call_session_are_projectable() {
    let dialogue = trajectory(vec![
        user(1, "What is the port?"),
        say(2, "It is 80."),
        user(3, "No, it is 9090."),
        say(4, "Understood."),
    ]);
    assert_eq!(projection_refusal(&dialogue), None);

    let procedure = trajectory(vec![
        user(1, "Restart the service."),
        call(
            2,
            "c1",
            "shell",
            json!({"cmd": "systemctl restart svc"}),
            "ok",
        ),
        say(3, "Restarted."),
    ]);
    assert_eq!(projection_refusal(&procedure), None);
}

#[test]
fn a_second_user_step_beside_a_tool_call_is_refused_naming_the_step() {
    let mixed = trajectory(vec![
        user(1, "Restart the service."),
        call(2, "c1", "shell", json!({"cmd": "x"}), "ok"),
        user(3, "Thanks, and now status."),
        say(4, "Fine."),
    ]);
    let refusal = projection_refusal(&mixed).unwrap();
    assert!(matches!(
        refusal,
        Unprojectable::ToolCallInDialogue { step: 2 }
    ));
    assert!(refusal.to_string().contains("step 2"), "{refusal}");
}

#[test]
fn image_content_is_refused_naming_the_step() {
    let mut with_image = trajectory(vec![user(1, "Look."), say(2, "ok")]);
    with_image.steps[1].message = MessageBody::Segments(vec![
        ContentSegment::Text { text: "a".into() },
        ContentSegment::image(atif::ImageMediaType::Png, "x.png"),
    ]);
    assert!(matches!(
        projection_refusal(&with_image),
        Some(Unprojectable::Image { step: 2 })
    ));
}
