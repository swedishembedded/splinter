// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training small models on the
// verified work of larger ones for its clients. If your team needs expertise
// in fine-tuning agents on their own trajectories then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a packed sft-step dataset holds one record per trajectory in which
//! every admissible action is supervised and everything else is context, so
//! that it supervises exactly what the unpacked view's records supervise
//! together - without repeating the context once per action.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Label, Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_core::experience::{Experience, PrivilegedKind, Task};
use splinter_data::{
    manifest_path, write_dataset, Corpus, Exclusion, Manifest, Projection, RecordBody, SftStep,
    Unchecked, View,
};

const INSTRUCTION: &str = "Compute one plus one and then two plus two, and report both.";

fn arithmetic() -> Task {
    task(
        "arithmetic",
        INSTRUCTION,
        vec![privileged(PrivilegedKind::Reference, "2 4")],
    )
}

/// Three actions: two tool calls (steps 2 and 3) and a final answer (step 4).
fn three_actions(task: &Task) -> Experience {
    experience(
        task,
        trajectory(vec![
            user(1, INSTRUCTION),
            call(2, "c1", "run_code", json!({"code": "1+1"}), "2"),
            call(3, "c2", "run_code", json!({"code": "2+2"}), "4"),
            say(4, "The answers are 2 and 4"),
        ]),
        Some("The answers are 2 and 4"),
        "2026-09-30T01:00:00.000Z",
    )
}

fn corpus_of(experience: &Experience, labels: &[(u64, Label)]) -> Corpus {
    let mut notes = vec![verdict(experience, Outcome::Pass, Strength::Formal)];
    notes.extend(labels.iter().map(|(step, l)| label(experience, *step, *l)));
    let mut corpus = Corpus::new();
    corpus.insert(experience.clone(), notes).unwrap();
    corpus
}

fn project(corpus: &Corpus, packed: bool) -> Projection {
    SftStep::new(Strength::Formal)
        .packed(packed)
        .project(corpus)
        .unwrap()
}

fn messages(projection: &Projection) -> Vec<&[WireMessage]> {
    projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Chat { messages, .. } => &messages[..],
            other => panic!("a chat record, got {other:?}"),
        })
        .collect()
}

/// The supervised messages of every record, in order.
fn targets(records: &[&[WireMessage]]) -> Vec<WireMessage> {
    records
        .iter()
        .flat_map(|m| m.iter().filter(|m| m.train).cloned())
        .collect()
}

#[test]
fn a_packed_record_supervises_the_union_of_the_unpacked_records() {
    let task = arithmetic();
    let experience = three_actions(&task);
    let corpus = corpus_of(&experience, &[]);

    let unpacked = project(&corpus, false);
    let packed = project(&corpus, true);
    let (unpacked_records, packed_records) = (messages(&unpacked), messages(&packed));
    assert_eq!(unpacked_records.len(), 3);
    let [record] = &packed_records[..] else {
        panic!("one record per trajectory, got {}", packed_records.len())
    };

    assert_eq!(targets(&packed_records), targets(&unpacked_records));
    assert_eq!(packed.records[0].metadata.experiences, [id(&experience)]);

    // Everything else is the context the longest unpacked record holds.
    let longest = unpacked_records.last().unwrap();
    let unsupervised = |m: &[WireMessage]| -> Vec<WireMessage> {
        m.iter()
            .cloned()
            .map(|m| WireMessage { train: false, ..m })
            .collect()
    };
    assert_eq!(unsupervised(record), unsupervised(longest));
}

#[test]
fn a_step_labelled_bad_stays_unsupervised_in_the_packed_record() {
    let task = arithmetic();
    let experience = three_actions(&task);
    let corpus = corpus_of(&experience, &[(3, Label::Bad)]);

    let unpacked = project(&corpus, false);
    let packed = project(&corpus, true);
    assert_eq!(messages(&unpacked).len(), 2);
    let packed_records = messages(&packed);
    let [record] = &packed_records[..] else {
        panic!("one record per trajectory, got {}", packed_records.len())
    };

    assert_eq!(targets(&packed_records), targets(&messages(&unpacked)));
    let bad = record
        .iter()
        .find(|m| m.tool_calls.iter().any(|c| c.id.as_deref() == Some("c2")))
        .unwrap();
    assert!(!bad.train, "the bad action is context, not a target");
    assert_eq!(packed.count(Exclusion::BadStep), 1);
    assert_eq!(unpacked.count(Exclusion::BadStep), 1);
}

#[test]
fn a_trajectory_with_no_admissible_action_yields_no_packed_record() {
    let task = arithmetic();
    let experience = three_actions(&task);
    let corpus = corpus_of(
        &experience,
        &[(2, Label::Bad), (3, Label::Bad), (4, Label::Bad)],
    );
    assert!(project(&corpus, true).records.is_empty());
}

/// Training keeps a packed record as one sequence and a dialogue's turns as
/// separate examples, so the manifest must say which a dataset is.
#[test]
fn the_manifest_records_that_a_dataset_is_packed_and_only_then() {
    let task = arithmetic();
    let experience = three_actions(&task);
    let corpus = corpus_of(&experience, &[]);
    let scratch = Scratch::new("sft-step-packed-manifest");
    let manifest_of = |name: &str, packed: bool| {
        let path = scratch.0.join(name);
        write_dataset(&path, &project(&corpus, packed), &Unchecked).unwrap();
        let bytes = std::fs::read(manifest_path(&path)).unwrap();
        (
            serde_json::from_slice::<Manifest>(&bytes).unwrap().packed,
            String::from_utf8(bytes).unwrap().contains("\"packed\""),
        )
    };
    assert_eq!(manifest_of("packed.jsonl", true), (true, true));
    assert_eq!(
        manifest_of("steps.jsonl", false),
        (false, false),
        "a manifest of an unpacked dataset is the one it always was"
    );
}
