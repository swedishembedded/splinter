// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a projection becomes a file only in a format its consumer reads
//! correctly, and never without a manifest that says what is in it.
//!
//! Objectives brain trains are written in brain's own format: SFT, and
//! classification rendered as SFT, as `generic-messages-v2`; preference
//! pairs as `generic-preference-v1`, each checked by brain's own parser
//! before it lands. Any other objective is refused unless the caller asks
//! for Splinter's export-only format, so no file brain would misread is
//! ever written. Beside every dataset lies its manifest:
//! the view, objective, strip policy, minimum strength, experiences, and
//! the counts of records and of exclusions by reason.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::json;
use std::collections::BTreeMap;

use splinter_core::annotation::{Outcome, Strength};
use splinter_core::experience::{Digest, PrivilegedKind};
use splinter_lab::SYSTEM_PROMPT;
use splinter_views::{
    manifest_path, write_dataset, Corpus, Format, Objective, Preference, Projection, Record,
    RecordBody, RecordMetadata, SftFinal, View, ViewError, WriteOptions, EXPORT_FORMAT,
};

fn graded_corpus() -> (Corpus, Vec<splinter_core::experience::ExperienceId>) {
    let sum = task("arithmetic", "What is two plus two?", Vec::new());
    let pass = answered(&sum, "4", "2026-09-30T01:00:00.000Z");
    let fail = answered(&sum, "5", "2026-09-30T02:00:00.000Z");
    let ungraded = answered(&sum, "6", "2026-09-30T03:00:00.000Z");
    let mut corpus = Corpus::new();
    corpus
        .insert(
            pass.clone(),
            vec![verdict(&pass, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    corpus
        .insert(
            fail.clone(),
            vec![verdict(&fail, Outcome::Fail, Strength::Formal)],
        )
        .unwrap();
    corpus.insert(ungraded, Vec::new()).unwrap();
    (corpus, vec![id(&pass), id(&fail)])
}

fn manifest(path: &std::path::Path) -> (serde_json::Value, Digest) {
    let bytes = std::fs::read(manifest_path(path)).unwrap();
    (serde_json::from_slice(&bytes).unwrap(), Digest::of(&bytes))
}

#[test]
fn every_dataset_has_a_content_addressed_manifest_with_its_counts() {
    let (corpus, ids) = graded_corpus();
    let scratch = Scratch::new("manifest");
    let path = scratch.0.join("sft.jsonl");
    let projection = SftFinal::new(Strength::Formal).project(&corpus).unwrap();
    let dataset = write_dataset(&path, &projection, WriteOptions::default()).unwrap();

    let (manifest, digest) = manifest(&path);
    assert_eq!(
        dataset.manifest, digest,
        "the manifest is named by its content"
    );
    assert_eq!(
        manifest,
        json!({
            "format": "generic-messages-v2",
            "dataset": dataset.digest,
            "view": "sft-final",
            "objective": "sft",
            "strip": "all",
            "min_strength": "formal",
            "experiences": [ids[0]],
            "tasks": [],
            "sources": [],
            "counts": {
                "records": 1,
                "excluded": {"failed": 1, "undecided": 1},
            },
        })
    );
}

#[test]
fn a_preference_dataset_is_generic_preference_v1_which_brain_parses() {
    // The task carries a hint only the teacher saw; the student prompt of
    // every pair is stripped of it.
    let sum = task(
        "arithmetic",
        "What is two plus two?",
        vec![privileged(PrivilegedKind::Hint, SECRET)],
    );
    let pass = answered(&sum, "4", "2026-09-30T01:00:00.000Z");
    let fail = answered(&sum, "5", "2026-09-30T02:00:00.000Z");
    let mut corpus = Corpus::new();
    corpus
        .insert(
            pass.clone(),
            vec![verdict(&pass, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    corpus
        .insert(
            fail.clone(),
            vec![verdict(&fail, Outcome::Fail, Strength::Formal)],
        )
        .unwrap();
    let projection = Preference::new(Strength::Formal).project(&corpus).unwrap();
    assert_eq!(projection.objective, Objective::Dpo);

    let scratch = Scratch::new("preference");
    let path = scratch.0.join("dpo.jsonl");
    let dataset = write_dataset(&path, &projection, WriteOptions::default()).unwrap();
    assert_eq!(dataset.format, Format::GenericPreferenceV1);
    assert_eq!(dataset.records, 1);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(!text.contains(SECRET), "{text}");
    let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
    assert_eq!(
        line,
        json!({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What is two plus two?"}
            ],
            "chosen": {"role": "assistant", "content": "4"},
            "rejected": {"role": "assistant", "content": "5"},
            "metadata": {
                "experiences": [id(&pass), id(&fail)],
                "view": "preference",
                "objective": "dpo",
            },
        })
    );
    let summary = splinter_policy::train::validate_preference_dataset(&path).unwrap();
    assert_eq!(summary.pairs, 1, "brain's own parser reads the pair");
    let (manifest, _) = manifest(&path);
    assert_eq!(manifest["format"], "generic-preference-v1");
    assert_eq!(manifest["experiences"], json!([id(&pass), id(&fail)]));

    // Export is only for what brain cannot train: a preference dataset
    // keeps brain's format when export is allowed.
    let exported = scratch.0.join("exported.jsonl");
    let dataset =
        write_dataset(&exported, &projection, WriteOptions { export_only: true }).unwrap();
    assert_eq!(dataset.format, Format::GenericPreferenceV1);
}

#[test]
fn a_preference_candidate_carries_the_tool_calls_of_its_final_turn() {
    let sum = task("arithmetic", "What is two plus two?", Vec::new());
    // The final turn answers and calls a tool in the same step.
    let mut last = call(2, "call-1", "run_code", json!({"code": "print(2+2)"}), "4");
    last.message = sven_sdk::atif::MessageBody::Text("4".into());
    let pass = experience(
        &sum,
        trajectory(vec![user(1, &sum.instruction), last]),
        Some("4"),
        "2026-09-30T01:00:00.000Z",
    );
    let fail = answered(&sum, "5", "2026-09-30T02:00:00.000Z");
    let mut corpus = Corpus::new();
    for (exp, outcome) in [(&pass, Outcome::Pass), (&fail, Outcome::Fail)] {
        corpus
            .insert(exp.clone(), vec![verdict(exp, outcome, Strength::Formal)])
            .unwrap();
    }
    let projection = Preference::new(Strength::Formal).project(&corpus).unwrap();
    let scratch = Scratch::new("preference-tools");
    let path = scratch.0.join("dpo.jsonl");
    write_dataset(&path, &projection, WriteOptions::default()).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
    assert_eq!(
        line["chosen"],
        json!({
            "role": "assistant",
            "content": "4",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "run_code", "arguments": "{\"code\":\"print(2+2)\"}"},
            }],
        })
    );
    assert_eq!(
        line["rejected"],
        json!({"role": "assistant", "content": "5"})
    );
    splinter_policy::train::validate_preference_dataset(&path).unwrap();
}

#[test]
fn objectives_brain_cannot_train_are_refused_unless_exported() {
    let metadata = |objective| RecordMetadata {
        group: None,
        experiences: Vec::new(),
        task: None,
        sources: Vec::new(),
        view: "hand-built".into(),
        objective,
    };
    let bodies = [
        (
            Objective::Contrastive,
            RecordBody::Contrastive {
                query: "q".into(),
                positive: "p".into(),
                negatives: vec!["n".into()],
            },
        ),
        (
            Objective::Reward,
            RecordBody::Rewarded {
                messages: Vec::new(),
                reward: 1.0,
            },
        ),
        (Objective::Cpt, RecordBody::Text { text: "t".into() }),
    ];
    let scratch = Scratch::new("export");
    for (objective, body) in bodies {
        let projection = Projection {
            view: "hand-built".into(),
            objective,
            strip: None,
            min_strength: None,
            records: vec![Record {
                body: body.clone(),
                metadata: metadata(objective),
            }],
            excluded: BTreeMap::new(),
        };
        assert!(!objective.trainable_by_brain(), "{objective:?}");
        let path = scratch.0.join(format!("{objective:?}.jsonl"));
        let refused = write_dataset(&path, &projection, WriteOptions::default());
        assert!(
            matches!(refused, Err(ViewError::ObjectiveNotTrainable { objective: o }) if o == objective),
            "{objective:?}: {refused:?}"
        );
        assert!(!path.exists() && !manifest_path(&path).exists());

        let dataset =
            write_dataset(&path, &projection, WriteOptions { export_only: true }).unwrap();
        assert_eq!(dataset.format, Format::SplinterExportV1);
        assert_eq!(dataset.trained_messages, None, "not a chat dataset");
        let text = std::fs::read_to_string(&path).unwrap();
        let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
        assert_eq!(line["format"], EXPORT_FORMAT);
        assert_eq!(line["shape"], body.shape());
        let (manifest, _) = manifest(&path);
        assert_eq!(manifest["format"], EXPORT_FORMAT);
    }

    // A trainable objective keeps brain's format even when export is
    // allowed: one objective, one file shape.
    let (corpus, _) = graded_corpus();
    let sft = SftFinal::new(Strength::Formal).project(&corpus).unwrap();
    let dataset = write_dataset(
        &scratch.0.join("sft.jsonl"),
        &sft,
        WriteOptions { export_only: true },
    )
    .unwrap();
    assert_eq!(dataset.format, Format::GenericMessagesV2);
}
