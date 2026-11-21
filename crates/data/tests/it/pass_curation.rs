// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curation of an agent's own verified
// trajectories into training data, for its clients. If your team needs
// expertise in self-training data pipelines, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: the sft-step view curates the passes of each task when asked, in
//! this order:
//!
//! 1. **final answer** - a pass whose last agent step calls a tool or says
//!    nothing is left out (`no_final_answer`);
//! 2. **distinct calls** - passes of one task whose sequences of tool calls
//!    (name and arguments, runs of whitespace as one space) are the same are
//!    one demonstration: the one with the least reasoning is kept, the others
//!    are `duplicate_pass`;
//! 3. **per-task cap** - at most N passes of a task are kept, the least
//!    reasoning per action first; the rest are `over_task_cap`.
//!
//! Failed attempts are not passes and are left out as before; the manifest
//! records the curation. Without the options every pass is kept.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::experience::Experience;
use splinter_data::curation::PassCuration;
use splinter_data::{Corpus, Exclusion, Projection, RecordBody, SftStep, View};

/// An attempt at `task`: its shell commands, its reasoning on the first
/// step, and whether it ends with a final answer.
fn attempt(
    task: &str,
    name: &str,
    commands: &[&str],
    reasoning: usize,
    answers: bool,
) -> Experience {
    let instruction = format!("Do {task}.");
    let task = common::task("workspace-change", &instruction, vec![]);
    let mut steps = vec![user(1, &instruction)];
    for (i, command) in commands.iter().enumerate() {
        let mut step = call(
            i as u64 + 2,
            &format!("c{i}"),
            "shell",
            json!({ "command": command }),
            "ok",
        );
        if i == 0 {
            step.reasoning_content = Some("r".repeat(reasoning));
        }
        steps.push(step);
    }
    if answers {
        steps.push(say(commands.len() as u64 + 2, &format!("answer {name}")));
    }
    experience(
        &task,
        trajectory(steps),
        Some(&format!("answer {name}")),
        &format!("2026-11-21T01:00:{:02}.000Z", name.len() + commands.len()),
    )
}

fn corpus() -> Corpus {
    let mut corpus = Corpus::new();
    let passes = [
        attempt("a", "p1", &["vim  -c   wq"], 10, true),
        // The same call as p1 once whitespace is normalised, with less reasoning.
        attempt("a", "p2", &["vim -c wq"], 5, true),
        attempt("a", "p3", &["ls", "vim -c x"], 40, true),
        attempt("a", "p4", &["vim -c y"], 30, true),
        // Ends on a tool call: no final answer.
        attempt("a", "p5", &["vim -c z"], 1, false),
        attempt("b", "q1", &["true"], 3, true),
    ];
    for exp in passes {
        let pass = verdict(&exp, Outcome::Pass, Strength::Executable);
        corpus.insert(exp, vec![pass]).unwrap();
    }
    let failed = attempt("a", "f1", &["false"], 0, true);
    let fail = verdict(&failed, Outcome::Fail, Strength::Executable);
    corpus.insert(failed, vec![fail]).unwrap();
    corpus
}

/// The passes whose records were written, by the name in their last
/// supervised message (the answer, or for p5 its last call).
fn kept(projection: &Projection) -> Vec<String> {
    let mut names: Vec<String> = projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Chat { messages, .. } => messages
                .iter()
                .rev()
                .find(|m| m.train)
                .map(|m| m.content.clone())
                .unwrap_or_default(),
            other => panic!("a chat record, got {other:?}"),
        })
        .collect();
    names.sort();
    names
}

fn build(curation: PassCuration) -> Projection {
    SftStep::new(Strength::Executable)
        .packed(true)
        .pass_curation(curation)
        .project(&corpus())
        .unwrap()
}

#[test]
fn without_curation_every_pass_is_kept() {
    let projection = build(PassCuration::default());
    assert_eq!(projection.records.len(), 6);
    assert_eq!(projection.count(Exclusion::Failed), 1);
    assert_eq!(
        projection.pass_curation, None,
        "nothing curated, nothing recorded"
    );
}

#[test]
fn final_answers_distinct_calls_and_a_cap_leave_the_least_reasoned_distinct_passes() {
    let curation = PassCuration {
        final_answer: true,
        distinct_calls: true,
        per_task: Some(2),
    };
    let projection = build(curation);
    assert_eq!(kept(&projection), ["answer p2", "answer p3", "answer q1"]);
    assert_eq!(projection.count(Exclusion::NoFinalAnswer), 1, "p5");
    assert_eq!(
        projection.count(Exclusion::DuplicatePass),
        1,
        "p1, p2's duplicate"
    );
    assert_eq!(
        projection.count(Exclusion::OverTaskCap),
        1,
        "p4: 30 per action against p3's 20"
    );
    assert_eq!(projection.count(Exclusion::Failed), 1);
    assert_eq!(projection.pass_curation, Some(curation));
}

#[test]
fn each_curation_applies_alone() {
    let only = |final_answer, distinct_calls, per_task| {
        let projection = build(PassCuration {
            final_answer,
            distinct_calls,
            per_task,
        });
        (projection.records.len(), kept(&projection))
    };
    let (n, _) = only(true, false, None);
    assert_eq!(n, 5, "p5 left out");
    let (n, names) = only(false, true, None);
    assert_eq!(n, 5);
    assert!(!names.contains(&"answer p1".to_string()), "{names:?}");
    // A cap of one keeps, per task, the pass with the least reasoning per
    // action: p5 (1 per action) in a, q1 in b.
    let (n, names) = only(false, false, Some(1));
    assert_eq!(n, 2);
    assert!(names.contains(&"answer q1".to_string()), "{names:?}");
}
