// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a privileged path's states are its steps with the person's turns
//! taken out, and the student is forked after its actions, the latest first.

use atif::{
    AgentProfile, ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep,
    Trajectory,
};
use serde_json::json;

use super::{fork_points, split_system, unguided};

fn user(id: u64, text: &str) -> TraceStep {
    TraceStep::new(id, StepOrigin::User, text)
}

fn typed(id: u64, text: &str) -> TraceStep {
    let mut step = user(id, text);
    step.extra = Some(json!({"sven": {"origin": "human"}}));
    step
}

fn call(id: u64, call: &str) -> TraceStep {
    let mut step = TraceStep::new(id, StepOrigin::Agent, "");
    step.tool_calls = Some(vec![
        ToolInvocation::new(call, "shell").with_arguments(json!({}))
    ]);
    step.observation = Some(StepObservation::single(ObservationEntry::for_call(
        call, "ok",
    )));
    step
}

fn path() -> Trajectory {
    let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1"));
    trajectory.steps = vec![
        TraceStep::new(1, StepOrigin::System, "the system prompt"),
        user(2, "Write a.txt"),
        call(3, "c1"),
        TraceStep::new(4, StepOrigin::Agent, "Wrote ho."),
        typed(5, "It must say hi."),
        call(6, "c2"),
        call(7, "c3"),
        TraceStep::new(8, StepOrigin::Agent, "Fixed."),
    ];
    trajectory
}

#[test]
fn the_persons_turns_and_the_replies_they_answered_are_taken_out() {
    let states = unguided(&path());
    let texts: Vec<_> = states
        .steps
        .iter()
        .map(|s| s.message.as_text().unwrap_or_default().to_string())
        .collect();
    // The person's turn, and "Wrote ho.", which ended the turn they answered.
    assert_eq!(states.steps.len(), 6);
    assert!(texts
        .iter()
        .all(|t| t != "It must say hi." && t != "Wrote ho."));
    assert_eq!(
        states.steps.iter().map(|s| s.step_id).collect::<Vec<_>>(),
        [1, 2, 3, 4, 5, 6]
    );
    let (system, exported) = split_system(&states);
    assert_eq!(system.as_deref(), Some("the system prompt"));
    assert_eq!(exported.steps[0].message.as_text(), Some("Write a.txt"));
    assert_eq!(exported.steps[0].step_id, 1);
}

#[test]
fn forks_are_after_the_paths_actions_but_its_last_the_latest_first() {
    let (_, exported) = split_system(&unguided(&path()));
    // Write a.txt (1), c1 (2), c2 (3), c3 (4), "Fixed." (5): after c3 the
    // path has done its work.
    assert_eq!(fork_points(&exported), [3, 2]);
}
