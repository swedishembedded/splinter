// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training small models on the
// verified work of larger ones for its clients. If your team needs expertise
// in negative controls for agent fine-tuning then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the two negative controls of the coding experiments are views of the
//! same experiences. The unverified control keeps trajectories whatever their
//! verdict, except those an exposure audit failed; the shuffled control keeps
//! the verified trajectories but gives each the instruction of another task.

#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Annotation, Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_core::experience::{Experience, PrivilegedKind};
use splinter_data::{Control, Corpus, Exclusion, Projection, RecordBody, SftStep, View};

fn solved(name: &str, instruction: &str) -> Experience {
    let task = task(
        name,
        instruction,
        vec![privileged(PrivilegedKind::Reference, "x")],
    );
    experience(
        &task,
        trajectory(vec![
            user(1, instruction),
            call(2, "c1", "run_code", json!({"code": name}), "ok"),
            say(3, "done"),
        ]),
        Some("done"),
        "2026-09-30T01:00:00.000Z",
    )
}

fn corpus_of(entries: Vec<(Experience, Vec<Annotation>)>) -> Corpus {
    let mut corpus = Corpus::new();
    for (experience, notes) in entries {
        corpus.insert(experience, notes).unwrap();
    }
    corpus
}

fn project(corpus: &Corpus, control: Control) -> Projection {
    SftStep::new(Strength::Executable)
        .packed(true)
        .control(control)
        .project(corpus)
        .unwrap()
}

fn first_user(record: &RecordBody) -> String {
    match record {
        RecordBody::Chat { messages, .. } => messages
            .iter()
            .find(|m: &&WireMessage| m.role == "user")
            .unwrap()
            .content
            .clone(),
        other => panic!("a chat record, got {other:?}"),
    }
}

#[test]
fn the_unverified_control_keeps_failed_trajectories_but_never_an_exposed_one() {
    let (passed, failed, exposed, unjudged) = (
        solved("a", "Fix the parser."),
        solved("b", "Fix the writer."),
        solved("c", "Fix the reader."),
        solved("d", "Fix the lookup."),
    );
    let corpus = corpus_of(vec![
        (
            passed.clone(),
            vec![verdict(&passed, Outcome::Pass, Strength::Executable)],
        ),
        (
            failed.clone(),
            vec![verdict(&failed, Outcome::Fail, Strength::Executable)],
        ),
        (
            exposed.clone(),
            vec![
                verdict(&exposed, Outcome::Pass, Strength::Executable),
                verdict(&exposed, Outcome::Fail, Strength::Constraint),
            ],
        ),
        (unjudged.clone(), vec![]),
    ]);

    let verified = project(&corpus, Control::Verified);
    assert_eq!(verified.records.len(), 1, "only the pass");
    let control = project(&corpus, Control::Unverified);
    assert_eq!(control.records.len(), 3, "pass, fail and no verdict");
    assert_eq!(
        control.count(Exclusion::Failed),
        1,
        "the exposed one stays out, as failed"
    );
}

#[test]
fn the_shuffled_control_gives_every_trajectory_another_tasks_instruction() {
    let all: Vec<_> = [
        ("a", "Fix the parser."),
        ("b", "Fix the writer."),
        ("c", "Fix the reader."),
    ]
    .into_iter()
    .map(|(n, i)| solved(n, i))
    .collect();
    let corpus = corpus_of(
        all.iter()
            .map(|e| {
                (
                    e.clone(),
                    vec![verdict(e, Outcome::Pass, Strength::Executable)],
                )
            })
            .collect(),
    );
    let verified = project(&corpus, Control::Verified);
    let seed = any_seed();
    let shuffled = project(&corpus, Control::ShuffledInstructions { seed });
    let again = project(&corpus, Control::ShuffledInstructions { seed });
    assert_eq!(shuffled.records.len(), verified.records.len());
    for (own, moved) in verified.records.iter().zip(&shuffled.records) {
        assert_ne!(
            first_user(&own.body),
            first_user(&moved.body),
            "no record keeps its own"
        );
        // The trajectory it supervises is unchanged.
        let tail = |r: &RecordBody| match r {
            RecordBody::Chat { messages, .. } => {
                messages.iter().skip(2).cloned().collect::<Vec<_>>()
            }
            _ => unreachable!(),
        };
        assert_eq!(tail(&own.body), tail(&moved.body));
    }
    let used: std::collections::BTreeSet<_> = shuffled
        .records
        .iter()
        .map(|r| first_user(&r.body))
        .collect();
    assert_eq!(used.len(), 3, "a permutation: each instruction used once");
    let texts = |p: &Projection| {
        p.records
            .iter()
            .map(|r| first_user(&r.body))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        texts(&shuffled),
        texts(&again),
        "the same seed gives the same pairing"
    );
}

#[test]
fn a_shuffle_of_one_instruction_is_refused_not_silently_the_identity() {
    let only = solved("a", "Fix the parser.");
    let corpus = corpus_of(vec![(
        only.clone(),
        vec![verdict(&only, Outcome::Pass, Strength::Executable)],
    )]);
    let result = SftStep::new(Strength::Executable)
        .packed(true)
        .control(Control::ShuffledInstructions { seed: any_seed() })
        .project(&corpus);
    assert!(result.is_err());
}
