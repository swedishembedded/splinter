// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements audits and gates that read what a tool-using
// agent really did, for its clients. If your team needs expertise in judging
// agent trajectories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the calls of a trajectory are read with the output each call itself
//! returned. Parallel calls finish in any order, so a result is paired to its
//! call by the id it names, never by position; a call without a result
//! returned nothing; what the agent said between calls is not a call.

#![allow(clippy::unwrap_used)]

use atif::{
    AgentProfile, ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep,
    Trajectory,
};
use serde_json::json;
use splinter_core::tool_calls::{outputs_by_call, tool_calls};

fn trajectory(steps: Vec<TraceStep>) -> Trajectory {
    let profile = AgentProfile {
        name: "t".into(),
        version: "1".into(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    };
    let mut trajectory = Trajectory::new("ATIF-v1.7", profile);
    trajectory.steps = steps;
    trajectory
}

/// An agent step that made `calls` (id, tool, command) and received `results`
/// (call id, output) in the order given.
fn step(id: u64, calls: &[(&str, &str, &str)], results: &[(&str, &str)]) -> TraceStep {
    let mut step = TraceStep::new(id, StepOrigin::Agent, "working");
    step.tool_calls = Some(
        calls
            .iter()
            .map(|(call, tool, command)| {
                ToolInvocation::new(*call, *tool).with_arguments(json!({ "command": command }))
            })
            .collect(),
    );
    let mut entries = results
        .iter()
        .map(|(call, output)| ObservationEntry::for_call(*call, *output));
    if let Some(first) = entries.next() {
        let mut observation = StepObservation::single(first);
        observation.results.extend(entries);
        step.observation = Some(observation);
    }
    step
}

#[test]
fn parallel_calls_whose_results_arrive_out_of_order_get_their_own_output() {
    let t = trajectory(vec![step(
        2,
        &[("a", "shell", "ls"), ("b", "shell", "cat x")],
        &[("b", "contents of x"), ("a", "listing")],
    )]);
    let calls = tool_calls(&t);
    assert_eq!(calls.len(), 2);
    assert_eq!(
        (calls[0].tool.as_str(), calls[0].output.as_str()),
        ("shell", "listing")
    );
    assert_eq!(calls[1].output, "contents of x");
    assert_eq!(calls[1].arguments["command"], "cat x");
    assert_eq!(calls[1].step, 2);
}

#[test]
fn a_call_with_no_result_returned_nothing() {
    let t = trajectory(vec![step(
        2,
        &[("a", "shell", "ls"), ("b", "shell", "pwd")],
        &[("a", "listing")],
    )]);
    let outputs = outputs_by_call(&t.steps[0]);
    assert_eq!(outputs, ["listing", ""]);
}

#[test]
fn results_that_do_not_name_their_calls_are_paired_by_position_when_the_counts_agree() {
    let t = trajectory(vec![step(
        2,
        &[("a", "shell", "ls"), ("b", "shell", "pwd")],
        &[("unknown-1", "first"), ("unknown-2", "second")],
    )]);
    assert_eq!(outputs_by_call(&t.steps[0]), ["first", "second"]);
}

#[test]
fn only_the_agents_steps_make_calls() {
    let mut other = step(1, &[("z", "shell", "rm")], &[("z", "no")]);
    other.source = StepOrigin::User;
    let t = trajectory(vec![
        other,
        step(2, &[("a", "shell", "ls")], &[("a", "ok")]),
    ]);
    let calls = tool_calls(&t);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].step, 2);
}
