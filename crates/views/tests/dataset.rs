// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a projection becomes a file only in a format its consumer reads
//! correctly, and never without a manifest that says what is in it.
//!
//! Objectives brain trains (SFT, and classification rendered as SFT) are
//! written as `generic-messages-v2`. Any other objective is refused unless
//! the caller asks for Splinter's export-only format, so no file brain
//! would misread is ever written. Beside every dataset lies its manifest:
//! the view, objective, strip policy, minimum strength, experiences, and
//! the counts of records and of exclusions by reason.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::json;
use splinter_store::annotation::{Outcome, Strength};
use splinter_store::experience::Digest;
use splinter_views::{
    manifest_path, write_dataset, Corpus, Format, Objective, Preference, SftFinal, View, ViewError,
    WriteOptions, EXPORT_FORMAT,
};

fn graded_corpus() -> (Corpus, Vec<splinter_store::experience::ExperienceId>) {
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
fn an_objective_brain_cannot_train_is_refused_unless_exported() {
    let (corpus, ids) = graded_corpus();
    let scratch = Scratch::new("export");
    let path = scratch.0.join("dpo.jsonl");
    let projection = Preference::new(Strength::Formal).project(&corpus).unwrap();
    assert_eq!(projection.objective, Objective::Dpo);
    assert_eq!(projection.records.len(), 1);

    assert!(matches!(
        write_dataset(&path, &projection, WriteOptions::default()),
        Err(ViewError::ObjectiveNotTrainable {
            objective: Objective::Dpo
        })
    ));
    assert!(!path.exists() && !manifest_path(&path).exists());

    let dataset = write_dataset(&path, &projection, WriteOptions { export_only: true }).unwrap();
    assert_eq!(dataset.format, Format::SplinterExportV1);
    assert_eq!(dataset.trained_messages, None, "not a chat dataset");
    let text = std::fs::read_to_string(&path).unwrap();
    let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
    assert_eq!(line["format"], EXPORT_FORMAT);
    assert_eq!(line["shape"], "preference");
    assert_eq!(line["chosen"]["content"], "4");
    assert_eq!(line["rejected"]["content"], "5");
    assert_eq!(line["metadata"]["experiences"], json!(ids));
    assert_eq!(line["metadata"]["objective"], "dpo");
    let (manifest, _) = manifest(&path);
    assert_eq!(manifest["format"], EXPORT_FORMAT);
    assert_eq!(manifest["experiences"], json!(ids));

    // A trainable objective keeps brain's format even when export is
    // allowed: one objective, one file shape.
    let sft = SftFinal::new(Strength::Formal).project(&corpus).unwrap();
    let dataset = write_dataset(
        &scratch.0.join("sft.jsonl"),
        &sft,
        WriteOptions { export_only: true },
    )
    .unwrap();
    assert_eq!(dataset.format, Format::GenericMessagesV2);
}
