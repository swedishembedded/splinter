// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the attempts of a pass@k solve are a group without a benchmark
//! trial to name them. A workspace verifier's verdict names neither a trial
//! nor a task, so the policy-gradient view places an attempt by what the
//! experience itself records: the task's address, the policy that solved it
//! and the attempt's number. An experience that records no attempt number
//! cannot be placed, and two policies' attempts at one task are two groups.

#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::{json, Value};
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, PrivilegedKind, Provenance, Task};
use splinter_data::{Corpus, Distill, Exclusion, GroupKind, PolicyGradient, RecordBody, View};

const FAILING_TEST: &str = "test_total_of_twelve";

fn repository_task() -> Task {
    let check = json!({
        "fail_to_pass": [{"argv": ["python3", "-m", "unittest", "tests.hidden"]}],
        "pass_to_pass": [],
    });
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::new(Environment::BENCH, json!({"snapshot": "s"})),
        "Make total() add.",
        vec![
            privileged(
                PrivilegedKind::Other("workspace-check".into()),
                &check.to_string(),
            ),
            privileged(PrivilegedKind::Reference, "--- a/t.py\n+++ b/t.py\n"),
        ],
    )
    .unwrap()
}

/// The evidence a workspace verifier records: commands, no trial, no task.
fn workspace_evidence(passed: bool) -> Value {
    let command = if passed {
        json!({"phase": "fail_to_pass", "index": 0, "exit_code": 0, "passed": true})
    } else {
        json!({"phase": "fail_to_pass", "index": 0, "exit_code": 1, "passed": false,
               "observed": {"failed": [FAILING_TEST], "missing": []}})
    };
    json!({ "commands": [command] })
}

/// One graded attempt, as `practice` stores it: the policy label and the
/// attempt number are in the provenance.
fn attempt(
    task: &Task,
    policy: &str,
    number: Option<u32>,
    passed: bool,
) -> (Experience, Annotation) {
    let trajectory = trajectory(vec![
        user(1, &task.instruction),
        call(2, "c0", "shell", json!({"command": "ls"}), "t.py"),
        say(3, "Done."),
    ]);
    let provenance = Provenance {
        policy: Some(policy.into()),
        attempt: number,
        ..Provenance::new(
            "scripted/solver-1",
            &FixedClock::new("2026-11-06T00:00:00.000Z"),
        )
    };
    let experience =
        Experience::new(task.clone(), trajectory, Some("Done.".into()), provenance).unwrap();
    let note = Annotation {
        experience: experience.id().unwrap(),
        producer: Producer {
            name: "workspace".into(),
            version: "1".into(),
        },
        body: AnnotationBody::Verdict {
            outcome: if passed { Outcome::Pass } else { Outcome::Fail },
            strength: Strength::Executable,
            evidence: workspace_evidence(passed),
        },
    };
    (experience, note)
}

fn corpus(attempts: Vec<(Experience, Annotation)>) -> Corpus {
    let mut corpus = Corpus::new();
    for (experience, note) in attempts {
        corpus.insert(experience, vec![note]).unwrap();
    }
    corpus
}

fn rewards(projection: &splinter_data::Projection) -> Vec<f64> {
    projection
        .records
        .iter()
        .filter_map(|r| match &r.body {
            RecordBody::Chat { outcome, .. } => outcome.as_ref().map(|o| o.reward),
            _ => None,
        })
        .collect()
}

#[test]
fn attempts_of_one_task_by_one_policy_are_a_group_without_a_trial_name() {
    let task = repository_task();
    let projection = PolicyGradient::new(Strength::Executable)
        .project(&corpus(vec![
            attempt(&task, "release-1", Some(0), true),
            attempt(&task, "release-1", Some(1), false),
        ]))
        .unwrap();
    assert_eq!(projection.groups.len(), 1, "{:?}", projection.groups);
    assert_eq!(projection.groups[0].kind, GroupKind::Graded);
    assert_eq!(projection.groups[0].task, task.task.id.to_string());
    assert_eq!(projection.groups[0].trials.len(), 2);
    let mut seen = rewards(&projection);
    seen.sort_by(f64::total_cmp);
    // Two actions of each attempt: the call and the answer.
    assert_eq!(seen, [0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn two_policies_attempts_at_one_task_are_two_groups() {
    let task = repository_task();
    let projection = PolicyGradient::new(Strength::Executable)
        .project(&corpus(vec![
            attempt(&task, "release-1", Some(0), true),
            attempt(&task, "release-1", Some(1), false),
            attempt(&task, "release-2", Some(0), false),
            attempt(&task, "release-2", Some(1), false),
        ]))
        .unwrap();
    let kinds: Vec<GroupKind> = projection.groups.iter().map(|g| g.kind).collect();
    assert_eq!(projection.groups.len(), 2, "{kinds:?}");
    assert!(kinds.contains(&GroupKind::Graded) && kinds.contains(&GroupKind::AllFail));
}

#[test]
fn an_experience_that_records_no_attempt_number_cannot_be_placed() {
    let task = repository_task();
    let projection = PolicyGradient::new(Strength::Executable)
        .project(&corpus(vec![attempt(&task, "release-1", None, true)]))
        .unwrap();
    assert_eq!(projection.count(Exclusion::NoRollout), 1);
    assert!(projection.groups.is_empty());
}

#[test]
fn an_all_fail_group_is_distilled_with_what_the_workspace_verifier_knows() {
    let task = repository_task();
    let projection = PolicyGradient::new(Strength::Executable)
        .distilling(Distill::Cliff)
        .project(&corpus(vec![
            attempt(&task, "release-1", Some(0), false),
            attempt(&task, "release-1", Some(1), false),
        ]))
        .unwrap();
    assert_eq!(projection.groups[0].kind, GroupKind::AllFail);
    assert!(projection.groups[0].distilled);
    assert_eq!(projection.records.len(), 4);
    for record in &projection.records {
        let RecordBody::Chat {
            teacher_context: teacher,
            ..
        } = &record.body
        else {
            panic!("not a chat record");
        };
        let context = &teacher.first().expect("a teacher context").content;
        assert!(context.contains(FAILING_TEST), "{context}");
        assert!(context.contains("--- a/t.py"), "{context}");
    }
}

#[test]
fn an_all_fail_task_that_keeps_nothing_privileged_is_counted() {
    let bare = Task::new(
        "workspace-change",
        Vec::new(),
        Environment::new(Environment::BENCH, json!({"snapshot": "s"})),
        "Make total() add.",
        Vec::new(),
    )
    .unwrap();
    let failed = |n| {
        let (experience, mut note) = attempt(&bare, "release-1", Some(n), false);
        // Nothing the verifier recorded to show a teacher either.
        if let AnnotationBody::Verdict { evidence, .. } = &mut note.body {
            *evidence = json!({"refused": "no_patch"});
        }
        (experience, note)
    };
    let projection = PolicyGradient::new(Strength::Executable)
        .distilling(Distill::Cliff)
        .project(&corpus(vec![failed(0), failed(1)]))
        .unwrap();
    assert_eq!(projection.count(Exclusion::NoPrivilegedContext), 2);
    assert!(projection.records.is_empty());
}
