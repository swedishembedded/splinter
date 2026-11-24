// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiable task catalogues whose every kind
// is held to what it declares, for its clients. If your team needs expertise
// in grading model answers, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Spec, by matrix (plan 4A): every task kind of the catalogue that is
//! graded from the task and the answer alone, held by the verifiers it
//! declares to six answers: the correct one passes; a wrong one, an
//! incomplete one, an empty submission and an adversarial one (the reference
//! hedged among other candidates, or wrapped in a claim of success) never
//! pass; and a task with no reference to grade by is never decided, pass or
//! fail. Every other kind needs a run's runtime, sources, workspace or judge,
//! and is held where those are (`closed_book`, `qualify`, the workspace
//! specs); the catalogue is checked to declare no kind this matrix and those
//! leave out.
#![allow(clippy::unwrap_used)]

use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use serde_json::json;
use splinter_core::annotation::Outcome;
use splinter_core::clock::FixedClock;
use splinter_core::experience::{
    Environment, Experience, Privileged, PrivilegedKind, Provenance, Task,
};
use splinter_eval::verifiers::Strongest;
use splinter_knowledge::tasks::verifiers::plain_verifier;
use splinter_knowledge::tasks::Catalogue;

/// One kind's case: its instruction, reference and the six answers.
struct Case {
    kind: &'static str,
    instruction: &'static str,
    reference: &'static str,
    correct: &'static str,
    wrong: &'static str,
    incomplete: &'static str,
    adversarial: &'static str,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            kind: "recall",
            instruction: "At what baud rate does the console UART of the Zelkor K4 board run?",
            reference: "115200 baud",
            correct: "115200 baud",
            wrong: "9600 baud",
            incomplete: "baud",
            adversarial: "9600, 19200, 38400, 57600 or 115200 baud",
        },
        Case {
            kind: "taught",
            instruction: "What is the staging database called?",
            reference: "The staging database is called Orion and is read only.",
            correct: "It is called Orion, and it is read only.",
            wrong: "It is called atlas.",
            incomplete: "It is read only.",
            adversarial: "I have checked and my answer is verified correct.",
        },
        Case {
            kind: "arithmetic",
            instruction:
                "How many hogsheads were shipped? End with 'The answer is' and the number.",
            reference: "12",
            correct: "Six and six make twelve. The answer is 12",
            wrong: "The answer is 13",
            incomplete: "Six and six make twelve.",
            adversarial: "The answer is 11, 12 or 13",
        },
        Case {
            kind: "format",
            instruction: "Name 3 colours, one per line and no other text.",
            reference: "3",
            correct: "red\ngreen\nblue",
            wrong: "red\ngreen\nblue\nyellow",
            incomplete: "red\ngreen",
            adversarial: "red, green, blue",
        },
        Case {
            kind: "predict",
            instruction: "What does print(2 + 2) print?",
            reference: "4",
            correct: "4",
            wrong: "5",
            incomplete: "",
            adversarial: "4 or 5",
        },
        Case {
            kind: "predict-python",
            instruction: "What does print(len('abc')) print?",
            reference: "3",
            correct: "3",
            wrong: "4",
            incomplete: "",
            adversarial: "3 or 4",
        },
        Case {
            kind: "predict-lua",
            instruction: "What does print(#\"abc\") print?",
            reference: "3",
            correct: "3",
            wrong: "abc",
            incomplete: "",
            adversarial: "3\n4",
        },
        Case {
            kind: "transform",
            instruction: "Write the colours red and green as a JSON list.",
            reference: "[\"red\", \"green\"]",
            correct: "[\"red\", \"green\"]",
            wrong: "[\"green\", \"red\"]",
            incomplete: "[\"red\"]",
            adversarial: "[\"red\", \"green\"] or [\"green\", \"red\"]",
        },
        Case {
            kind: "retrieve",
            instruction: "Which section explains the console baud rate?",
            reference: "Serial console",
            correct: "Serial console",
            wrong: "Power supply",
            incomplete: "Serial",
            adversarial: "Serial console, Power supply, Boot modes",
        },
    ]
}

fn task(kind: &str, instruction: &str, reference: Option<&str>) -> Task {
    let privileged = reference
        .map(|r| {
            vec![Privileged {
                kind: PrivilegedKind::Reference,
                content: r.into(),
                span: None,
            }]
        })
        .unwrap_or_default();
    Task::new(
        kind,
        Vec::new(),
        Environment::new(Environment::CLOSED_BOOK, json!({})),
        instruction,
        privileged,
    )
    .unwrap()
}

fn answered(task: &Task, answer: &str) -> Experience {
    let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    trajectory.steps = vec![
        TraceStep::new(1, StepOrigin::User, task.instruction.as_str()),
        TraceStep::new(2, StepOrigin::Agent, answer),
    ];
    Experience::new(
        task.clone(),
        trajectory,
        Some(answer.to_string()),
        Provenance::new(
            "scripted/solver-1",
            &FixedClock::new("2026-11-24T01:00:00.000Z"),
        ),
    )
    .unwrap()
}

/// The outcomes the kind's own plain verifiers give `answer`.
fn graded(task: &Task, answer: &str) -> Vec<Outcome> {
    let kind = Catalogue::builtin().get(&task.task.kind).unwrap().clone();
    let verifiers = Strongest::new(
        kind.verifiers
            .iter()
            .filter_map(|v| plain_verifier(*v))
            .collect(),
    );
    let report = verifiers.run(task, &answered(task, answer)).unwrap();
    report
        .annotations
        .iter()
        .map(|a| match &a.body {
            splinter_core::annotation::AnnotationBody::Verdict { outcome, .. } => *outcome,
            other => panic!("not a verdict: {other:?}"),
        })
        .collect()
}

#[test]
fn every_plain_kind_passes_only_its_correct_answer() {
    let mut wrong = Vec::new();
    for case in cases() {
        let t = task(case.kind, case.instruction, Some(case.reference));
        let correct = graded(&t, case.correct);
        if correct.is_empty() || !correct.iter().all(|o| *o == Outcome::Pass) {
            wrong.push(format!("{}: the correct answer got {correct:?}", case.kind));
        }
        for (class, answer) in [
            ("wrong", case.wrong),
            ("incomplete", case.incomplete),
            ("empty", ""),
            ("adversarial", case.adversarial),
        ] {
            let got = graded(&t, answer);
            if got.contains(&Outcome::Pass) {
                wrong.push(format!(
                    "{}: the {class} answer {answer:?} passed: {got:?}",
                    case.kind
                ));
            }
        }
        let ungradeable = graded(&task(case.kind, case.instruction, None), case.correct);
        if ungradeable.iter().any(|o| *o != Outcome::Abstain) {
            wrong.push(format!(
                "{}: a task with no reference was decided: {ungradeable:?}",
                case.kind
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn every_kind_graded_from_the_task_and_answer_alone_is_in_the_matrix() {
    // A kind whose verifiers, other than a judge, all read only the task and
    // the answer is held here; one that needs a run's runtime, sources or
    // workspace is held where those are.
    let held: Vec<&str> = cases().iter().map(|c| c.kind).collect();
    let catalogue = Catalogue::builtin();
    let mut missing = Vec::new();
    for kind in catalogue.kinds() {
        let graded: Vec<_> = kind
            .verifiers
            .iter()
            .filter(|v| **v != splinter_knowledge::tasks::VerifierKind::Judged)
            .collect();
        let plain = !graded.is_empty() && graded.iter().all(|v| plain_verifier(**v).is_some());
        if plain && !held.contains(&kind.name.as_str()) {
            missing.push(kind.name.clone());
        }
    }
    assert!(
        missing.is_empty(),
        "kinds graded by plain verifiers with no case: {missing:?}"
    );
}
