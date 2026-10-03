// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: what the views write is what brain's trainers accept, and the
//! backend - not the views - decides which objectives it trains.
//!
//! The views write `generic-messages-v2` and `generic-preference-v1` lines
//! and rules about those formats are enforced by brain's parser, on the other
//! side of a repository boundary. A unit test asserting "the JSON looks
//! right" asserts the writer's belief about the format, so these specs run the
//! real parser over what the writer produced.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use splinter_core::chat::WireMessage;
use splinter_data::{
    manifest_path, write_dataset, Format, Objective, Projection, Record, RecordBody,
    RecordMetadata, ViewError,
};
use splinter_policy::{BrainDatasetCheck, PolicyError, TrainingCapabilities};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("policy-views-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn metadata(objective: Objective) -> RecordMetadata {
    RecordMetadata {
        group: None,
        experiences: Vec::new(),
        task: None,
        sources: Vec::new(),
        view: "hand-built".into(),
        objective,
    }
}

fn message(role: &str, content: &str, train: bool) -> WireMessage {
    WireMessage {
        role: role.into(),
        content: content.into(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        train,
    }
}

fn projection(objective: Objective, body: RecordBody) -> Projection {
    Projection {
        view: "hand-built".into(),
        objective,
        strip: None,
        min_strength: None,
        records: vec![Record {
            body,
            metadata: metadata(objective),
        }],
        excluded: BTreeMap::new(),
    }
}

fn chat(train: bool) -> Projection {
    projection(
        Objective::Sft,
        RecordBody::Chat {
            messages: vec![
                message("system", "You are helpful.", false),
                message("user", "What is two plus two?", false),
                message("assistant", "4", train),
            ],
        },
    )
}

#[test]
fn a_chat_dataset_is_read_whole_and_its_supervision_is_counted_alike() {
    let dir = scratch("chat");
    let path = dir.join("sft.jsonl");
    let written = write_dataset(&path, &chat(true), &BrainDatasetCheck).expect("brain accepts it");

    let summary = splinter_policy::train::validate_dataset(&path).expect("the real parser");
    assert_eq!(summary.records, written.records);
    assert_eq!(
        Some(summary.trained_messages),
        written.trained_messages,
        "the writer counts the supervised messages as the trainer's parser does"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_chat_dataset_that_supervises_nothing_is_refused_and_leaves_nothing() {
    let dir = scratch("unsupervised");
    let path = dir.join("sft.jsonl");
    let refused = write_dataset(&path, &chat(false), &BrainDatasetCheck);
    assert!(
        matches!(&refused, Err(ViewError::Invalid { reason, .. }) if reason.contains("no message is supervised")),
        "{refused:?}"
    );
    assert!(!path.exists() && !manifest_path(&path).exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_preference_dataset_is_read_pair_for_pair() {
    let dir = scratch("preference");
    let path = dir.join("dpo.jsonl");
    let pairs = projection(
        Objective::Dpo,
        RecordBody::Preference {
            prompt: vec![message("user", "What is two plus two?", false)],
            chosen: message("assistant", "4", false),
            rejected: message("assistant", "5", false),
        },
    );
    let written = write_dataset(&path, &pairs, &BrainDatasetCheck).expect("brain accepts it");
    assert_eq!(written.format, Format::GenericPreferenceV1);

    let summary = splinter_policy::train::validate_preference_dataset(&path).expect("real parser");
    assert_eq!(summary.pairs, written.records);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn only_the_objectives_brain_trains_from_a_file_are_trainable() {
    let brain = TrainingCapabilities::BRAIN;
    for objective in [Objective::Sft, Objective::Classification, Objective::Dpo] {
        assert!(brain.require(objective).is_ok(), "{objective:?}");
    }
    for objective in [Objective::Contrastive, Objective::Reward, Objective::Cpt] {
        assert!(
            matches!(
                brain.require(objective),
                Err(PolicyError::ObjectiveNotTrainable { objective: o }) if o == objective
            ),
            "{objective:?}"
        );
        assert!(!brain.supports(objective.line_format()), "{objective:?}");
    }
}
