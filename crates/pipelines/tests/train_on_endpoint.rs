// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose training runs as a
// job on a shared model service, for its clients. If your team needs
// expertise in moving fine-tuning off the application and onto a serving
// cluster, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: every training regime is a job of the brain service, merging an
//! adapter included; its checkpoint is installed in the run's model store and
//! read from there. With no service to ask (no endpoint, no `brain` binary)
//! a run is refused by name: nothing trains in this process.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::path::Path;

use common::{config, Scratch};
use splinter_agent::CancelToken;
use splinter_core::training::Regime;
use splinter_model::scripted_endpoint::{ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::ArtifactStore;
use splinter_model::BrainEndpoints;
use splinter_orchestrator::Context;
use splinter_pipelines::train::{BrainTrainer, TrainPlan, Trainer, Tuning};

fn plan(dir: &Path, regime: Regime) -> TrainPlan {
    TrainPlan {
        candidate: "candidate-1".into(),
        regime,
        dir: dir.to_path_buf(),
        datasets: Vec::new(),
        base: dir.join("Qwen/Toy"),
        continue_from: None,
        parent: None,
        replay_file: None,
        replay_share: None,
        rehearsal: None,
        steps: 1,
        rank: 2,
        beta: 0.1,
        tuning: Tuning::default(),
        train_precision: None,
        seed: 1, // seed: the scripted endpoint answers the same whatever the seed; any value
        eval_every: 0,
        patience: 0,
        monitor_share: 0.0,
        resume_key: None,
    }
}

#[test]
fn a_merge_is_a_job_whose_checkpoint_is_installed_in_the_model_store() {
    let scratch = Scratch::new("train-on-endpoint");
    let store = scratch.0.join("artifacts");
    std::fs::create_dir_all(&store).unwrap();
    let mut settings = config(&scratch);
    let server = ScriptedEndpoint::start(vec![]).with_training(JobScript {
        models: Some(settings.model_store.clone()),
        ..JobScript::over(store.clone())
    });
    settings.brain = Some(
        BrainEndpoints::new(vec![server.url()], SCRIPTED_KEY)
            .unwrap()
            .with_artifact_store(ArtifactStore::open(&store).unwrap()),
    );
    let models = settings.model_store.clone();
    let ctx = Context::new(settings, false).unwrap();
    let adapter = scratch.0.join("adapter.safetensors");
    std::fs::write(&adapter, b"lora").unwrap();
    let work = scratch.0.join("merge-1");

    let checkpoint = BrainTrainer
        .merge(
            &ctx,
            &plan(&scratch.0, Regime::Sft).base,
            &adapter,
            &work.join("checkpoint"),
            &CancelToken::new(),
        )
        .unwrap();

    assert_eq!(server.jobs().actions(), vec!["merge"]);
    assert_eq!(checkpoint.dir, models.join("local/splinter-merge-1"));
    assert!(checkpoint.weights.is_file());
}

#[test]
fn with_no_service_to_ask_a_run_is_refused_by_name() {
    let scratch = Scratch::new("train-without-endpoint");
    let mut settings = config(&scratch);
    settings.brain_binary = None;
    let ctx = Context::new(settings, false).unwrap();
    let error = BrainTrainer
        .merge(
            &ctx,
            &scratch.0,
            &scratch.0,
            &scratch.0.join("merge-1/checkpoint"),
            &CancelToken::new(),
        )
        .map(|_| ())
        .unwrap_err();
    assert!(
        error.to_string().contains("--brain-endpoint"),
        "the run names how to reach a server: {error}"
    );
}
