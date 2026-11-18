// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements dense per-token credit for agents that learn
// from their own failed attempts, for its clients. If your team needs
// expertise in on-policy self-distillation with privileged information then
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: where every attempt of a task failed, GRPO's advantage is zero and
//! says nothing; the self-distillation views turn such an attempt into the
//! student's own conversation plus a teacher context built from what the
//! verifier knows (the hidden checks that failed, the reference change). The
//! teacher context never reaches the student's messages, a group with a
//! spread of rewards keeps GRPO's scalar advantage, and an attempt with
//! nothing privileged to show is refused with a counted reason.

#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::{json, Value};
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::experience::{Environment, PrivilegedKind, Task};
use splinter_data::{
    write_dataset, Corpus, Distill, Exclusion, GroupKind, PolicyGradient, RecordBody, Unchecked,
    View,
};

const REFERENCE: &str = "--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-return a - b\n+return a + b\n";
const FAILING_TEST: &str = "test_secret_sum_of_twelve";

/// A repository task with its hidden check and reference change.
fn repository_task(with_privileged: bool) -> Task {
    let check = json!({
        "fail_to_pass": [{"argv": ["python3", "-m", "unittest", "tests.hidden"]}],
        "pass_to_pass": [{"argv": ["python3", "-m", "unittest", "tests.visible"]}],
    });
    let privileged = if with_privileged {
        vec![
            privileged(
                PrivilegedKind::Other("workspace-check".into()),
                &check.to_string(),
            ),
            privileged(PrivilegedKind::Reference, REFERENCE),
        ]
    } else {
        Vec::new()
    };
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::new(
            Environment::BENCH,
            json!({"task": {"id": "repo/calc", "version": "1"}}),
        ),
        "Make add() add.",
        privileged,
    )
    .unwrap()
}

struct Attempt {
    name: &'static str,
    passed: bool,
    privileged: bool,
}

fn attempt(name: &'static str, passed: bool) -> Attempt {
    Attempt {
        name,
        passed,
        privileged: true,
    }
}

fn evidence(a: &Attempt) -> Value {
    let commands = if a.passed {
        json!([{"phase": "fail_to_pass", "index": 0, "exit_code": 0, "passed": true}])
    } else {
        json!([{"phase": "fail_to_pass", "index": 0, "exit_code": 1, "passed": false,
                "observed": {"failed": [FAILING_TEST], "missing": []}}])
    };
    json!({"trial": a.name, "task": {"id": "repo/calc", "version": "1"}, "model": "m1",
           "commands": commands})
}

fn corpus(attempts: &[Attempt]) -> Corpus {
    let mut corpus = Corpus::new();
    for (n, a) in attempts.iter().enumerate() {
        let mut call_step = call(3, "c0", "shell", json!({"command": "ls"}), "calc.py");
        call_step.reasoning_content = Some("look".into());
        let trajectory = trajectory(vec![
            atif::TraceStep::new(1, atif::StepOrigin::System, "You are a careful engineer."),
            user(2, "Make add() add."),
            call_step,
            say(4, "Done."),
        ]);
        let at = format!("2026-11-01T00:00:00.{n:03}Z");
        let experience = experience(
            &repository_task(a.privileged),
            trajectory,
            Some("Done."),
            &at,
        );
        let note = Annotation {
            experience: experience.id().unwrap(),
            producer: Producer {
                name: "workspace".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Verdict {
                outcome: if a.passed {
                    Outcome::Pass
                } else {
                    Outcome::Fail
                },
                strength: Strength::Executable,
                evidence: evidence(a),
            },
        };
        corpus.insert(experience, vec![note]).unwrap();
    }
    corpus
}

fn view(distill: Distill) -> PolicyGradient {
    PolicyGradient::new(Strength::Executable).distilling(distill)
}

fn chat(
    record: &splinter_data::Record,
) -> (
    &Vec<splinter_core::chat::WireMessage>,
    Option<f64>,
    &Vec<splinter_data::TeacherMessage>,
) {
    match &record.body {
        RecordBody::Chat {
            messages,
            outcome,
            teacher_context,
            ..
        } => (
            messages,
            outcome.as_ref().map(|o| o.reward),
            teacher_context,
        ),
        other => panic!("a chat record, got {other:?}"),
    }
}

fn teacher_text(record: &splinter_data::Record) -> String {
    chat(record)
        .2
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn an_all_fail_group_is_distilled_and_a_graded_group_keeps_its_scalar_advantage() {
    // Task one-version group: all three failed. A second policy's group has a pass and a fail.
    let mut corpus = corpus(&[
        attempt("a1", false),
        attempt("a2", false),
        attempt("a3", false),
    ]);
    let mixed = self::corpus(&[attempt("b1", true), attempt("b2", false)]);
    // The second group is another model's attempts at the same task.
    for entry in mixed.entries() {
        let mut experience = entry.experience.clone();
        experience.provenance.created_at = format!("2026-11-02T00:00:00.{}Z", entry.id);
        let mut notes = entry.notes.clone();
        for note in &mut notes {
            if let AnnotationBody::Verdict { evidence, .. } = &mut note.body {
                evidence["model"] = json!("m2");
            }
            note.experience = experience.id().unwrap();
        }
        corpus.insert(experience, notes).unwrap();
    }
    let projection = view(Distill::Cliff).project(&corpus).unwrap();

    let kinds: Vec<(GroupKind, bool)> = projection
        .groups
        .iter()
        .map(|g| (g.kind, g.distilled))
        .collect();
    assert!(kinds.contains(&(GroupKind::AllFail, true)), "{kinds:?}");
    assert!(kinds.contains(&(GroupKind::Graded, false)), "{kinds:?}");
    // Two actions per attempt: three distilled attempts, two with a scalar advantage.
    let (teacher, scalar): (Vec<_>, Vec<_>) = projection
        .records
        .iter()
        .partition(|r| !chat(r).2.is_empty());
    assert_eq!((teacher.len(), scalar.len()), (6, 4));
    assert!(
        teacher.iter().all(|r| chat(r).1.is_none()),
        "a distilled record carries no scalar advantage"
    );
    assert!(scalar.iter().all(|r| chat(r).1.is_some()));
}

#[test]
fn the_teacher_is_shown_what_failed_and_the_reference_and_the_student_never_is() {
    let projection = view(Distill::Cliff)
        .project(&corpus(&[attempt("a1", false), attempt("a2", false)]))
        .unwrap();
    assert!(!projection.records.is_empty());
    for record in &projection.records {
        let text = teacher_text(record);
        assert!(
            text.contains(FAILING_TEST) && text.contains("python3 -m unittest tests.hidden"),
            "{text}"
        );
        assert!(
            text.contains("return a + b"),
            "the reference change is shown to the teacher: {text}"
        );
        let (messages, _, _) = chat(record);
        for m in messages {
            assert!(
                !m.content.contains(FAILING_TEST)
                    && !m.content.contains("return a + b")
                    && !m.content.contains("hidden"),
                "privileged text leaked into the student's turn {m:?}"
            );
        }
    }
}

#[test]
fn without_self_distillation_an_all_fail_group_still_teaches_nothing() {
    let projection = view(Distill::Off)
        .project(&corpus(&[attempt("a1", false), attempt("a2", false)]))
        .unwrap();
    assert!(projection.records.is_empty());
    assert_eq!(projection.count(Exclusion::NoGroupSignal), 2);
    assert_eq!(projection.groups[0].kind, GroupKind::AllFail);
    assert!(!projection.groups[0].distilled);
}

#[test]
fn the_self_distill_view_distils_passed_and_failed_attempts_alike() {
    let view = view(Distill::Every);
    assert_eq!(view.name(), "self-distill");
    let projection = view
        .project(&corpus(&[attempt("a1", true), attempt("a2", false)]))
        .unwrap();
    assert_eq!(projection.records.len(), 4);
    assert!(projection
        .records
        .iter()
        .all(|r| chat(r).1.is_none() && !chat(r).2.is_empty()));
    assert!(projection.groups.iter().all(|g| g.distilled));
    // A passed attempt's teacher knows it passed.
    assert!(projection
        .records
        .iter()
        .any(|r| teacher_text(r).contains("passed every hidden check")));
}

#[test]
fn an_attempt_with_nothing_privileged_to_show_is_refused_with_a_counted_reason() {
    let bare = Attempt {
        name: "a1",
        passed: true,
        privileged: false,
    };
    // A passed attempt whose evidence names no failed command and whose task carries no check or reference.
    let projection = view(Distill::Every)
        .project(&corpus(&[bare, attempt("a2", false)]))
        .unwrap();
    assert_eq!(projection.count(Exclusion::NoPrivilegedContext), 1);
    assert_eq!(
        projection.records.len(),
        2,
        "the other attempt is unaffected"
    );
}

#[test]
fn the_dataset_line_places_the_teacher_context_after_the_first_user_turn_for_brain() {
    let projection = view(Distill::Cliff)
        .project(&corpus(&[attempt("a1", false), attempt("a2", false)]))
        .unwrap();
    let path = std::env::temp_dir().join(format!("self-distill-spec-{}.jsonl", std::process::id()));
    write_dataset(&path, &projection, &Unchecked).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(splinter_data::manifest_path(&path));
    for line in text.lines() {
        let line: Value = serde_json::from_str(line).unwrap();
        let messages = line["messages"].as_array().unwrap();
        let first_user = messages.iter().position(|m| m["role"] == "user").unwrap();
        assert_eq!(line["teacher_context"]["at"], first_user + 1, "{line}");
        assert_eq!(line["teacher_context"]["messages"][0]["role"], "user");
        assert!(line.get("advantage").is_none() && line.get("reward").is_none());
    }
}
