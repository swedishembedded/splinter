// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The brain trainer, against a scripted brain service: a variant is one
//! training job whose datasets go by digest and whose adapter comes back to
//! the candidate's directory; asked again under the same resume key, brain
//! answers from its record and trains nothing; the candidate is counted
//! only once the endpoint lists it; parameters brain does not take are
//! refused before anything is sent; and a dataset's records are measured by
//! brain's own check, each record given its own length.

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
        objective: "sft",
        train: &train,
        validation: &validation,
        // seed: the test checks the seed it was given is the one brain is asked for.
        seed: 7,
        thinking: false,
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

    let bf16 = json!({"rank": 4, "steps": 3, "base_precision": "bf16"});
    let out_bf16 = dir.path().join("lora-r4-bf16");
    std::fs::create_dir_all(&out_bf16).unwrap();
    trainer
        .train(&TrainRequest {
            params: &bf16,
            out: &out_bf16,
            resume_key: "lora-r4-bf16",
            ..request.clone()
        })
        .unwrap();
    assert_eq!(
        service.jobs().created().last().unwrap()["base_format"],
        "bf16"
    );
    let monitored = json!({"rank": 4, "steps": 6, "eval_every": 2, "patience": 3});
    let out_monitored = dir.path().join("lora-r4-monitored");
    std::fs::create_dir_all(&out_monitored).unwrap();
    std::fs::write(&validation, std::fs::read(&train).unwrap()).unwrap();
    trainer
        .train(&TrainRequest {
            params: &monitored,
            out: &out_monitored,
            resume_key: "lora-r4-monitored",
            ..request.clone()
        })
        .unwrap();
    let created = service.jobs().created();
    let job = created.last().unwrap();
    assert_eq!(
        (job["eval_every"].as_u64(), job["patience"].as_u64()),
        (Some(2), Some(3)),
        "the validation records are monitored and the best step is kept"
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
        jobs.len() + 4,
        "the refused request reached no job"
    );

    // A preference dataset is a preference job.
    let pairs = dir.path().join("pairs.jsonl");
    let pair = json!({
        "prompt": [{"role": "user", "content": "edit"}],
        "chosen": {"role": "assistant", "content": "done"},
        "rejected": {"role": "assistant", "content": "gave up"},
        "metadata": {"view": "preference", "objective": "preference"},
    });
    std::fs::write(&pairs, format!("{pair}\n{pair}\n")).unwrap();
    let no_pairs = dir.path().join("no-pairs.jsonl");
    std::fs::write(&no_pairs, "").unwrap();
    let out_pairs = dir.path().join("lora-r4-pairs");
    std::fs::create_dir_all(&out_pairs).unwrap();
    trainer
        .train(&TrainRequest {
            objective: "preference",
            train: &pairs,
            validation: &no_pairs,
            out: &out_pairs,
            resume_key: "lora-r4-pairs",
            ..request.clone()
        })
        .unwrap();
    assert_eq!(
        service.jobs().created().last().unwrap()["objective"],
        "preference"
    );
}

#[test]
fn a_variants_parameters_are_resolved_with_the_trainers_defaults_and_refused_early() {
    let endpoints =
        BrainEndpoints::new(vec!["http://127.0.0.1:9/v1".into()], SCRIPTED_KEY).unwrap();
    let trainer = BrainTrainer::new(endpoints, std::path::PathBuf::from("models"));
    let resolved = trainer
        .resolve("lora", &json!({"rank": 8, "steps": 10}))
        .unwrap();
    assert_eq!(resolved["alpha"], 16.0, "alpha defaults to twice the rank");
    assert_eq!(
        resolved["base_precision"], "fp32",
        "brain's default tier, named"
    );
    assert_eq!(resolved["grad_accum"], 1);
    assert!(
        trainer.resolve("lora", &json!({"rank": 8})).is_err(),
        "steps is required"
    );
    assert!(
        trainer
            .resolve("dpo", &json!({"rank": 8, "steps": 1}))
            .is_err(),
        "an unsupported method"
    );
}

#[test]
fn every_record_of_a_batch_has_a_length_the_first_included() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("artifacts");
    std::fs::create_dir_all(&store).unwrap();
    let service = ScriptedEndpoint::start(vec![Card::base("Qwen/Toy")])
        .with_training(JobScript::over(store.clone()));
    let endpoints = BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY)
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&store).unwrap());
    let trainer = BrainTrainer::new(endpoints, dir.path().join("models"));
    let base: ModelRef = "brain:Qwen/Toy".parse().unwrap();
    let line = json!({"messages": [
        {"role": "user", "content": "edit", "train": false},
        {"role": "assistant", "content": "done", "train": true}
    ], "tools": []})
    .to_string();
    let lengths = trainer
        .measure(&base, &[line.clone(), line.clone(), line])
        .unwrap();
    assert_eq!(lengths.len(), 3);
    assert!(lengths.iter().all(|tokens| *tokens > 0), "{lengths:?}");
}

#[test]
fn each_record_is_measured_as_its_own_length_by_brains_check() {
    // The scripted check counts the characters of a record's messages as its
    // tokens, so each length says which record it is of; brain numbers the
    // lines from 1.
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("artifacts");
    std::fs::create_dir_all(&store).unwrap();
    let service = ScriptedEndpoint::start(vec![Card::base("Qwen/Toy")])
        .with_training(JobScript::over(store.clone()));
    let endpoints = BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY)
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&store).unwrap());
    let trainer = BrainTrainer::new(endpoints, dir.path().join("models"));
    let answers = [
        "short".to_owned(),
        "long ".repeat(40),
        "mid mid mid".to_owned(),
    ];
    let lines: Vec<String> = answers
        .iter()
        .map(|answer| {
            json!({"messages": [
                {"role": "user", "content": "edit", "train": false},
                {"role": "assistant", "content": answer, "train": true}
            ], "tools": []})
            .to_string()
        })
        .collect();
    let lengths = trainer
        .measure(&"brain:Qwen/Toy".parse().unwrap(), &lines)
        .unwrap();
    let own: Vec<u64> = answers
        .iter()
        .map(|a| ("edit".len() + a.len()) as u64)
        .collect();
    assert_eq!(lengths, own);
}
