// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements versioned data contracts across the systems
// it composes, for its clients. If your team needs expertise in keeping
// producer and consumer schemas in step, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The bytes an external trainer writes, held to the schemas Splinter reads.
//!
//! Both fixtures are written by the trainer, not by Splinter: `job-plan.json`
//! is the plan a distributed training run is launched under and
//! `checkpoint-commit.json` is the manifest it publishes at a commit. A field
//! added, renamed or removed on either side is a change to the pair of files
//! here and to the schemas in `splinter_core::checkpoint`; parsing with
//! `deny_unknown_fields` is what makes a one-sided change fail.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use splinter_core::checkpoint::{CheckpointManifest, JobPlan};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    std::fs::read(&path).unwrap()
}

#[test]
fn the_job_plan_the_trainer_reads_has_the_fields_splinter_links_against() {
    let plan: JobPlan = serde_json::from_slice(&fixture("job-plan.json")).unwrap();
    assert_eq!(plan.job, "qwen-coding-v1");
    assert_eq!(plan.world_size, 2);
    assert_eq!((plan.tp, plan.pp, plan.dp), (1, 1, 2));
    assert_eq!(plan.coordinator, "127.0.0.1:7000");
    assert_eq!(plan.handshake_timeout_ms, 120_000);
    assert_eq!(plan.op_timeout_ms, 600_000);
}

#[test]
fn the_checkpoint_manifest_the_trainer_writes_names_each_artifact() {
    let manifest: CheckpointManifest =
        serde_json::from_slice(&fixture("checkpoint-commit.json")).unwrap();
    assert_eq!(manifest.job, "qwen-coding-v1");
    assert_eq!(manifest.step, 500);
    assert_eq!(manifest.world_size, 2);
    assert_eq!(manifest.files.len(), 2);

    let state = &manifest.files[0];
    assert_eq!(state.role, "state");
    assert_eq!(state.path.as_str(), "state/train.state");
    assert_eq!(state.len, 2048);
    assert_eq!(
        state.sha256.digest().as_str(),
        "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    );

    let weights = &manifest.files[1];
    assert_eq!(weights.role, "weights");
    assert_eq!(weights.path.as_str(), "weights.safetensors");
    assert_eq!(weights.len, 4096);
    assert_eq!(
        weights.sha256.digest().as_str(),
        "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    );

    // The plan is carried as the exact text it was submitted with, so the
    // manifest and the plan fixture name the same run.
    let plan: JobPlan = serde_json::from_str(&manifest.plan).unwrap();
    let submitted: JobPlan = serde_json::from_slice(&fixture("job-plan.json")).unwrap();
    assert_eq!(plan, submitted);
}
