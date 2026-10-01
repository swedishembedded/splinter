// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Builders shared by the view specs: tasks, experiences with hand-built
//! ATIF trajectories, and the annotations that grade and relate them.

// Each spec uses a different subset of the builders.
#![allow(dead_code)]

use std::path::PathBuf;

use serde_json::json;
use splinter_lab::{WireMessage, SYSTEM_PROMPT};
use splinter_store::annotation::{
    Annotation, AnnotationBody, Label, Outcome, Producer, RelationKind, Strength,
};
use splinter_store::clock::FixedClock;
use splinter_store::experience::{
    Environment, Experience, ExperienceId, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use sven_sdk::atif::{
    AgentProfile, ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep,
    Trajectory,
};

/// Text that only the teacher ever saw; no record may contain it unless a
/// strip policy explicitly keeps it.
pub const SECRET: &str = "TEACHER-ONLY";

/// A closed-book task of `kind`.
pub fn task(kind: &str, instruction: &str, privileged: Vec<Privileged>) -> Task {
    task_with_evidence(kind, instruction, privileged, Vec::new())
}

/// A closed-book task of `kind` grounded in `evidence`.
pub fn task_with_evidence(
    kind: &str,
    instruction: &str,
    privileged: Vec<Privileged>,
    evidence: Vec<Span>,
) -> Task {
    Task::new(
        kind,
        evidence,
        Environment::closed_book(),
        instruction,
        privileged,
    )
    .unwrap()
}

/// A privileged item of `kind`.
pub fn privileged(kind: PrivilegedKind, content: &str) -> Privileged {
    Privileged {
        kind,
        content: content.into(),
        span: None,
    }
}

/// A trajectory of `steps`.
pub fn trajectory(steps: Vec<TraceStep>) -> Trajectory {
    let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    trajectory.steps = steps;
    trajectory
}

/// A user step.
pub fn user(step: u64, text: &str) -> TraceStep {
    TraceStep::new(step, StepOrigin::User, text)
}

/// An agent step answering with `text` and no tool call.
pub fn say(step: u64, text: &str) -> TraceStep {
    TraceStep::new(step, StepOrigin::Agent, text)
}

/// An agent step calling `name` with `arguments` as call `call`, which the
/// environment answered with `result`.
pub fn call(
    step: u64,
    call: &str,
    name: &str,
    arguments: serde_json::Value,
    result: &str,
) -> TraceStep {
    let mut s = TraceStep::new(step, StepOrigin::Agent, "");
    s.tool_calls = Some(vec![
        ToolInvocation::new(call, name).with_arguments(arguments)
    ]);
    s.observation = Some(StepObservation::single(ObservationEntry::for_call(
        call, result,
    )));
    s
}

/// An experience of `task`; `at` distinguishes solves of one task.
pub fn experience(
    task: &Task,
    trajectory: Trajectory,
    output: Option<&str>,
    at: &str,
) -> Experience {
    Experience::new(
        task.clone(),
        trajectory,
        output.map(str::to_string),
        Provenance::new("scripted/solver-1", &FixedClock::new(at)),
    )
    .unwrap()
}

/// An experience of `task` whose trajectory is the instruction and the
/// answer `output`.
pub fn answered(task: &Task, output: &str, at: &str) -> Experience {
    experience(
        task,
        trajectory(vec![user(1, &task.instruction), say(2, output)]),
        Some(output),
        at,
    )
}

fn producer() -> Producer {
    Producer {
        name: "grader".into(),
        version: "1".into(),
    }
}

/// A verdict on `exp`.
pub fn verdict(exp: &Experience, outcome: Outcome, strength: Strength) -> Annotation {
    Annotation {
        experience: exp.id().unwrap(),
        producer: producer(),
        body: AnnotationBody::Verdict {
            outcome,
            strength,
            evidence: json!({}),
        },
    }
}

/// A label on step `step` of `exp`.
pub fn label(exp: &Experience, step: u64, label: Label) -> Annotation {
    Annotation {
        experience: exp.id().unwrap(),
        producer: producer(),
        body: AnnotationBody::StepLabel {
            step,
            label,
            evidence: json!({}),
        },
    }
}

/// "`exp` `kind` `other`".
pub fn relation(exp: &Experience, kind: RelationKind, other: &Experience) -> Annotation {
    Annotation {
        experience: exp.id().unwrap(),
        producer: producer(),
        body: AnnotationBody::Relation {
            kind,
            other: other.id().unwrap(),
        },
    }
}

/// The id of `exp`.
pub fn id(exp: &Experience) -> ExperienceId {
    exp.id().unwrap()
}

/// A fresh scratch directory, removed when dropped.
pub struct Scratch(pub PathBuf);

impl Scratch {
    /// A new, empty scratch directory for `name`.
    pub fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("splinter-views-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `messages` after their first turn, which must be the system turn every
/// solve runs under, unsupervised.
pub fn after_system(messages: &[WireMessage]) -> &[WireMessage] {
    let [system, rest @ ..] = messages else {
        panic!("a conversation, got none");
    };
    assert_eq!(
        (system.role.as_str(), system.content.as_str(), system.train),
        ("system", SYSTEM_PROMPT, false),
        "every conversation starts with the solver's system turn"
    );
    rest
}
