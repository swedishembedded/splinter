// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose training runs as a
// job on a shared model service, for its clients. If your team needs
// expertise in testing training clients against a scripted service, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What the served-training specs share: a checkpoint directory brain's store
//! would name `Qwen/Toy`, an artifact store, chat records, the endpoints of a
//! scripted service, and a fine-tune request over them.

// Helpers unwrap: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use splinter_model::endpoint::BrainEndpoints;
use splinter_model::scripted_endpoint::{ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::served_train::ArtifactStore;
use splinter_model::train::{BasePrecision, FineTune};

pub struct Rig {
    pub dir: tempfile::TempDir,
    pub model_dir: PathBuf,
    pub store: PathBuf,
}

/// A checkpoint directory brain's store would name `Qwen/Toy`, an artifact
/// store, and a training set of three conversations.
pub fn rig() -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("models/Qwen/Toy");
    std::fs::create_dir_all(&model_dir).unwrap();
    let store = dir.path().join("artifacts");
    std::fs::create_dir_all(&store).unwrap();
    Rig {
        dir,
        model_dir,
        store,
    }
}

pub fn records(path: &Path, rollouts: bool) {
    let lines: Vec<String> = ["cab", "fed", "ace"]
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let mut record = serde_json::json!({
                "messages": [
                    {"role": "user", "content": prompt, "train": false},
                    {"role": "assistant", "content": format!("answer {i}"), "train": true}
                ],
                "tools": []
            });
            if rollouts {
                record["reward"] = serde_json::json!(if i == 0 { 1.0 } else { 0.0 });
                record["group"] = serde_json::json!("g");
            }
            record.to_string()
        })
        .collect();
    std::fs::write(path, lines.join("\n")).unwrap();
}

pub fn endpoints(service: &ScriptedEndpoint, rig: &Rig) -> BrainEndpoints {
    BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY)
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&rig.store).unwrap())
        .with_refused_wait(Duration::from_millis(50))
}

pub fn fine_tune_request<'a>(
    rig: &'a Rig,
    attempt: &'a Path,
    train: &'a Path,
    held_out: &'a Path,
) -> FineTune<'a> {
    FineTune {
        model_dir: &rig.model_dir,
        train,
        held_out,
        held_out_text: None,
        monitor: None,
        eval_every: 0,
        patience: 0,
        attempt_dir: attempt,
        steps: 3,
        rank: 4,
        alpha: 8.0,
        replay: &[],
        replay_share: None,
        weighted_replay: &[],
        grad_accum: 2,
        continue_from: None,
        cancel: None,
        base: BasePrecision::Bf16,
        learning_rate: Some(2e-4),
        weight_decay: 0.01,
        // seed: the job's request carries it, and the spec reads it back.
        seed: 7,
        max_block: None,
        thinking: false,
        pack_answers: false,
        on_step: None,
        keep_evaluations: None,
        resume_key: None,
    }
}

// Shared by test binaries that do not all use it.
#[allow(dead_code)]
pub fn data_of(params: &serde_json::Value) -> serde_json::Value {
    serde_json::from_str(params["data"].as_str().unwrap()).unwrap()
}
