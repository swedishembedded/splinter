// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose training runs as a
// job on a shared model service, for its clients. If your team needs
// expertise in moving fine-tuning off the application and onto a serving
// cluster, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: with brain endpoints, a training regime that has no job on the
//! service (preference optimisation, a full-parameter fine-tune, merging an
//! adapter) is refused by name instead of loading the model in this process
//! on cards the service is using. Without endpoints they are not refused for
//! that reason: the in-process path stays.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::path::Path;

use common::{config, Scratch};
use splinter_agent::CancelToken;
use splinter_core::training::Regime;
use splinter_model::scripted_endpoint::{ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::BrainEndpoints;
use splinter_orchestrator::{Context, OrchestratorError};
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
    }
}

fn refusal(error: OrchestratorError) -> String {
    match error {
        OrchestratorError::Refused(why) => why,
        other => panic!("expected a refusal, got {other}"),
    }
}

#[test]
fn a_regime_without_a_job_is_refused_by_name_when_the_run_has_endpoints() {
    let scratch = Scratch::new("train-on-endpoint");
    let server = ScriptedEndpoint::start(vec![]);
    let mut settings = config(&scratch);
    settings.brain = Some(BrainEndpoints::new(vec![server.url()], SCRIPTED_KEY).unwrap());
    let ctx = Context::new(settings, false).unwrap();
    let cancel = CancelToken::new();

    let dpo = BrainTrainer
        .train_preference(&ctx, &plan(&scratch.0, Regime::Dpo), &cancel)
        .map(|_| ())
        .unwrap_err();
    assert!(refusal(dpo).contains("a preference (DPO) run has no job on the brain service"));
    let full = BrainTrainer
        .train_full(&ctx, &plan(&scratch.0, Regime::Sft), &cancel)
        .map(|_| ())
        .unwrap_err();
    assert!(refusal(full).contains("a full-parameter run has no job"));
    let merge = BrainTrainer
        .merge(
            &ctx,
            &scratch.0,
            &scratch.0,
            &scratch.0.join("merged"),
            &cancel,
        )
        .map(|_| ())
        .unwrap_err();
    assert!(refusal(merge).contains("merging an adapter into a base has no job"));
}

#[test]
fn without_endpoints_those_regimes_are_not_refused_for_that_reason() {
    let scratch = Scratch::new("train-without-endpoint");
    let ctx = Context::new(config(&scratch), false).unwrap();
    let error = BrainTrainer
        .train_preference(&ctx, &plan(&scratch.0, Regime::Dpo), &CancelToken::new())
        .map(|_| ())
        .unwrap_err();
    assert!(
        !error.to_string().contains("brain service"),
        "the in-process path ran and failed on its own terms: {error}"
    );
}
