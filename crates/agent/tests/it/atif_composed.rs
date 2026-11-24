// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent pipelines whose every trajectory is
// held to the format's own validator, for its clients. If your team needs
// expertise in agent trace formats then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec, against an independent reference: every trajectory splinter-agent
//! composes (a teacher's correction continuing a student's attempt, a policy
//! continuing one, a continuation led by a user's guidance) is ATIF by
//! Harbor's own validator, not only by Splinter's reading of it. Run by
//! `make test/worker`, which has Harbor installed.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use atif::{
    AgentProfile, ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep,
    Trajectory,
};
use serde_json::json;
use splinter_agent::correction::{compose, compose_as};
use splinter_agent::fork_at::POLICY;

fn call(step: u64, id: &str, path: &str, result: &str) -> TraceStep {
    let mut s = TraceStep::new(step, StepOrigin::Agent, "");
    s.tool_calls = Some(vec![
        ToolInvocation::new(id, "write_file").with_arguments(json!({"path": path}))
    ]);
    s.observation = Some(StepObservation::single(ObservationEntry::for_call(
        id, result,
    )));
    s
}

fn trajectory(agent: &str, steps: Vec<TraceStep>) -> Trajectory {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new(agent, "1"));
    t.steps = steps;
    t
}

/// Every composition the crate makes, by name.
fn composed() -> Vec<(&'static str, Trajectory)> {
    let student = trajectory(
        "scripted",
        vec![
            TraceStep::new(1, StepOrigin::User, "make it pass"),
            call(2, "s1", "a", "red"),
        ],
    );
    let continued = |extra: Vec<TraceStep>| {
        let mut steps = vec![
            TraceStep::new(1, StepOrigin::User, "make it pass"),
            call(2, "s1", "a", "red"),
        ];
        steps.extend(extra);
        trajectory("teacher", steps)
    };
    let corrected = continued(vec![
        call(3, "t1", "b", "green"),
        TraceStep::new(4, StepOrigin::Agent, "done"),
    ]);
    let guided = continued(vec![
        TraceStep::new(3, StepOrigin::User, "first make the parser accept dates"),
        call(4, "t1", "b", "green"),
        TraceStep::new(5, StepOrigin::Agent, "done"),
    ]);
    vec![
        (
            "a teacher's correction",
            compose(&student, &corrected, "student/1", "teacher/1").unwrap(),
        ),
        (
            "a policy's continuation",
            compose_as(&student, &corrected, "student/1", "student/1", POLICY).unwrap(),
        ),
        (
            "a guided continuation",
            compose(&student, &guided, "student/1", "teacher/1").unwrap(),
        ),
    ]
}

#[test]
#[ignore = "needs Harbor in workers/harbor/.venv (make worker/install); run by make test/worker"]
fn every_composed_trajectory_is_atif_by_harbors_validator() {
    let python = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../workers/harbor/.venv/bin/python"
    );
    assert!(
        std::path::Path::new(python).is_file(),
        "no Harbor environment: make worker/install"
    );
    let dir = tempfile::tempdir().unwrap();
    for (name, trajectory) in composed() {
        let path = dir.path().join("trajectory.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&trajectory).unwrap()).unwrap();
        let out = Command::new(python)
            .args([
                "-I",
                "-c",
                "import sys\nfrom harbor.utils.trajectory_validator import TrajectoryValidator\nv = TrajectoryValidator()\nok = v.validate(sys.argv[1])\nprint(v.get_errors())\nsys.exit(0 if ok else 1)",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{name} is not ATIF by Harbor's validator: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
