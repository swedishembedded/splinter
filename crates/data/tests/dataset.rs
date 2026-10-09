// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a projection becomes a file only in a format its consumer reads
//! correctly, and never without a manifest that says what is in it.
//!
//! Objectives with a chat or preference shape are written as chat lines
//! (`generic-messages-v2`: SFT, and classification rendered as SFT) and
//! preference pairs (`generic-preference-v1`); any other objective is written
//! in Splinter's own export format. A caller's check can refuse a file before
//! it lands. Beside every dataset lies its manifest:
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
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_data::{
    manifest_path, write_dataset, Corpus, Format, Objective, Preference, Projection, Record,
    RecordBody, RecordMetadata, SftFinal, Unchecked, View, ViewError, EXPORT_FORMAT,
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
    let dataset = write_dataset(&path, &projection, &Unchecked).unwrap();

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
fn a_preference_dataset_is_written_as_generic_preference_v1() {
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
    let dataset = write_dataset(&path, &projection, &Unchecked).unwrap();
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
    let (manifest, _) = manifest(&path);
    assert_eq!(manifest["format"], "generic-preference-v1");
    assert_eq!(manifest["experiences"], json!([id(&pass), id(&fail)]));
}

#[test]
fn a_preference_candidate_carries_the_tool_calls_of_its_final_turn() {
    let sum = task("arithmetic", "What is two plus two?", Vec::new());
    // The final turn answers and calls a tool in the same step.
    let mut last = call(2, "call-1", "run_code", json!({"code": "print(2+2)"}), "4");
    last.message = atif::MessageBody::Text("4".into());
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
    write_dataset(&path, &projection, &Unchecked).unwrap();
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
}

#[test]
fn objectives_without_a_chat_or_preference_shape_are_exported() {
    let metadata = |objective| RecordMetadata {
        group: None,
        split: None,
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
            system_prompt: None,
            terms: None,
        };
        assert_eq!(objective.line_format(), Format::SplinterExportV1);
        let path = scratch.0.join(format!("{objective:?}.jsonl"));
        let dataset = write_dataset(&path, &projection, &Unchecked).unwrap();
        assert_eq!(dataset.format, Format::SplinterExportV1);
        assert_eq!(dataset.trained_messages, None, "not a chat dataset");
        let text = std::fs::read_to_string(&path).unwrap();
        let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
        assert_eq!(line["format"], EXPORT_FORMAT);
        assert_eq!(line["shape"], body.shape());
        let (manifest, _) = manifest(&path);
        assert_eq!(manifest["format"], EXPORT_FORMAT);
    }

    // One objective, one file shape.
    let (corpus, _) = graded_corpus();
    let sft = SftFinal::new(Strength::Formal).project(&corpus).unwrap();
    let dataset = write_dataset(&scratch.0.join("sft.jsonl"), &sft, &Unchecked).unwrap();
    assert_eq!(dataset.format, Format::GenericMessagesV2);
}

#[test]
fn a_check_that_refuses_a_file_leaves_nothing_at_the_path() {
    struct Refuses;
    impl splinter_data::DatasetCheck for Refuses {
        fn check(&self, _: Format, pending: &std::path::Path, _: usize) -> Result<(), String> {
            assert!(pending.exists(), "the check reads the file before it lands");
            Err("not a file this backend reads".into())
        }
    }
    let (corpus, _) = graded_corpus();
    let scratch = Scratch::new("refused");
    let path = scratch.0.join("sft.jsonl");
    let projection = SftFinal::new(Strength::Formal).project(&corpus).unwrap();
    let refused = write_dataset(&path, &projection, &Refuses);
    assert!(
        matches!(&refused, Err(ViewError::Invalid { reason, .. }) if reason.contains("not a file")),
        "{refused:?}"
    );
    assert!(!path.exists() && !manifest_path(&path).exists());
    assert!(!path.with_extension("pending").exists());
}
