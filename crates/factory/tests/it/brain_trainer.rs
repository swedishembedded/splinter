// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The brain trainer, against a scripted brain service: a variant is one
//! training job whose datasets go by digest and whose adapter comes back to
//! the candidate's directory; asked again under the same resume key, brain
//! answers from its record and trains nothing; the candidate is counted
//! only once the endpoint lists it; and parameters brain does not take are
//! refused before anything is sent.

use std::time::Duration;

use serde_json::json;
use splinter_factory::recipe::ModelRef;
use splinter_factory::training::{BrainTrainer, TrainRequest, Trainer};
use splinter_model::endpoint::BrainEndpoints;
use splinter_model::scripted_endpoint::{Card, ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::ArtifactStore;

fn records(path: &std::path::Path) {
    let line = json!({"messages": [
        {"role": "system", "content": "You are Sven", "train": false},
        {"role": "user", "content": "edit", "train": false},
        {"role": "assistant", "content": "done", "train": true}
    ], "tools": []});
    std::fs::write(path, format!("{line}\n{line}\n")).unwrap();
}

#[test]
fn a_variant_is_a_resumable_brain_job_whose_adapter_is_served_before_it_counts() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("artifacts");
    let adapters = dir.path().join("adapters");
    std::fs::create_dir_all(&store).unwrap();
    std::fs::create_dir_all(&adapters).unwrap();
    let service = ScriptedEndpoint::start(vec![Card::base("Qwen/Toy")])
        .with_training(JobScript::over(store.clone()))
        .serving_adapters(adapters.clone());
    let endpoints = BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY)
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&store).unwrap())
        .with_adapter_directory(adapters)
        .with_refused_wait(Duration::from_millis(50));
    let trainer = BrainTrainer::new(endpoints, dir.path().join("models"));
    let base: ModelRef = "brain:Qwen/Toy".parse().unwrap();
    let (train, validation) = (
        dir.path().join("train.jsonl"),
        dir.path().join("validation.jsonl"),
    );
    records(&train);
    std::fs::write(&validation, "").unwrap();
    let out = dir.path().join("lora-r4-abc");
    std::fs::create_dir_all(&out).unwrap();
    let params = json!({"rank": 4, "steps": 3});
    let request = TrainRequest {
        base: &base,
        method: "lora",
        params: &params,
        train: &train,
        validation: &validation,
        seed: 7,
        resume_key: "lora-r4-abc",
        out: &out,
    };
    let trained = trainer.train(&request).unwrap();
    let jobs = service.jobs().created();
    assert_eq!(jobs.len(), 1);
    assert_eq!(
        (
            jobs[0]["base"].as_str(),
            jobs[0]["rank"].as_u64(),
            jobs[0]["seed"].as_u64()
        ),
        (Some("Qwen/Toy"), Some(4), Some(7))
    );
    assert_eq!(
        trained.report["held_out"], "train",
        "an empty validation split scores on the training records"
    );
    let adapter = trained.adapter.clone().unwrap();
    assert!(adapter.starts_with(&out) && adapter.is_file());

    // The process that asked died; asked again under its key, nothing trains again.
    std::fs::remove_dir_all(&out).unwrap();
    std::fs::create_dir_all(&out).unwrap();
    trainer.train(&request).unwrap();
    assert_eq!(service.jobs().trained(), 1);

    let served = trainer
        .serve(
            &base,
            Some(&trainer.train(&request).unwrap().adapter.unwrap()),
        )
        .unwrap();
    assert!(
        !served.is_empty() && served != "Qwen/Toy",
        "the adapter, not the base: {served}"
    );

    let wrong = json!({"rank": 4, "steps": 3, "rnak": 8});
    let refused = trainer
        .train(&TrainRequest {
            params: &wrong,
            ..request.clone()
        })
        .unwrap_err();
    assert!(refused.to_string().contains("rnak"), "{refused}");
    assert_eq!(
        service.jobs().created().len(),
        jobs.len() + 2,
        "the refused request reached no job"
    );
}
