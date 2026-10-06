// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements preference fine-tuning of agent policies
// on their own verified answers for its clients. If your team needs
// expertise in preference optimisation and evaluation-gated releases, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: `train` trains the regime its datasets' objective names.
//!
//! * Chat datasets are trained by supervised fine-tuning, preference pair
//!   datasets by preference (DPO) fine-tuning; one run trains one regime,
//!   and `beta` applies to preferences only.
//! * A preference candidate continues the champion like any other, replays
//!   nothing (brain's preference trainer trains on its pairs alone), records
//!   its regime, and is released through the same gate.
//! * A later supervised candidate replays only the chat records of its
//!   lineage.
//!
//! Training is the fixtures' test double; brain's preference trainer itself
//! is exercised by brain's own tests.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{
    anchor_file, dataset, decide, gate_context, policy, preference_dataset, released, Brain,
    FakeTrainer, ANCHOR, FACTS,
};
use splinter_agent::CancelToken;
use splinter_core::dataset::DatasetId;
use splinter_core::training::Regime;
use splinter_orchestrator::Context;
use splinter_pipelines::release::anchor;
use splinter_pipelines::train::{
    train, TrainRequest, Tuning, DEFAULT_DPO_BETA, DEFAULT_LEARNING_RATE, DEFAULT_REPLAY_FRACTION,
    DEFAULT_WEIGHT_DECAY,
};

fn request(datasets: &[&DatasetId], beta: Option<f32>) -> TrainRequest {
    TrainRequest {
        datasets: datasets.iter().map(ToString::to_string).collect(),
        rehearsal: None,
        from: policy(),
        replay_fraction: DEFAULT_REPLAY_FRACTION,
        steps: Some(1),
        rank: 4,
        beta,
        tuning: Tuning::default(),
    }
}

fn refusal(ctx: &Context, request: &TrainRequest, trainer: &FakeTrainer) -> String {
    let refused = train(ctx, request, trainer, &CancelToken::new()).unwrap_err();
    assert!(refused.is_refusal(), "{refused}");
    refused.to_string()
}

#[test]
fn a_preference_dataset_trains_by_dpo_from_the_champion_and_is_gated_like_any_other() {
    let (scratch, ctx) = gate_context("train-dpo", Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    let champion = released(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let champion_adapter = ctx.releases().get(&champion).unwrap().artifact;

    let pairs = preference_dataset(&ctx, "beta", FACTS);
    let trainer = FakeTrainer::knowing(&[ANCHOR, "alpha", "beta"]);
    let candidate = train(
        &ctx,
        &request(&[&pairs], Some(0.2)),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(*trainer.called.lock().unwrap(), [Regime::Dpo]);
    let plan = trainer.plans.lock().unwrap()[0].clone();
    assert_eq!((plan.regime, plan.beta), (Regime::Dpo, 0.2));
    assert_eq!(plan.continue_from, Some(champion_adapter));
    assert_eq!(plan.replay_file, None);

    assert_eq!(candidate.regime, Regime::Dpo);
    assert_eq!(candidate.parent.as_ref(), Some(&champion));
    assert_eq!(candidate.replay, None, "a preference run replays nothing");
    assert_eq!((candidate.base_score, candidate.tuned_score), (None, None));
    let preference = candidate.preference.as_ref().unwrap();
    assert_eq!(preference.beta, 0.2);
    assert_eq!(
        preference.held_out_score, None,
        "the double measures nothing"
    );

    let decided = decide(&ctx, &candidate);
    assert!(decided.gate.passed, "{:#?}", decided.gate);
    let release = ctx
        .releases()
        .get(decided.release.as_ref().unwrap())
        .unwrap();
    assert_eq!(release.manifest.training.regime, Regime::Dpo);
    assert_eq!(release.manifest.parent, Some(champion.clone()));

    // The next supervised candidate replays its lineage's chat records:
    // the first release's, not the preference release's pairs.
    let chat = dataset(&ctx, "gamma", FACTS);
    let trainer = FakeTrainer::knowing(&[ANCHOR, "alpha", "beta", "gamma"]);
    let next = train(
        &ctx,
        &request(&[&chat], None),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(*trainer.called.lock().unwrap(), [Regime::Sft]);
    let replay = next.replay.unwrap();
    let sources: Vec<_> = replay.sources.iter().map(|s| &s.release).collect();
    assert_eq!(sources, [&champion]);
    assert!(replay.records > 0);
}

#[test]
fn a_run_trains_one_regime_and_beta_only_for_preferences() {
    let (_scratch, ctx) = gate_context("train-regimes", Brain::Missing);
    let chat = dataset(&ctx, "alpha", FACTS);
    let pairs = preference_dataset(&ctx, "beta", FACTS);
    let trainer = FakeTrainer::knowing(&[]);

    let mixed = refusal(&ctx, &request(&[&chat, &pairs], None), &trainer);
    assert!(mixed.contains("one run trains one regime"), "{mixed}");
    let beta = refusal(&ctx, &request(&[&chat], Some(0.2)), &trainer);
    assert!(beta.contains("preference datasets only"), "{beta}");
    assert!(trainer.called.lock().unwrap().is_empty());

    let sft = train(
        &ctx,
        &request(&[&chat], None),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(sft.regime, Regime::Sft);
    assert!(sft.preference.is_none());
    let dpo = train(
        &ctx,
        &request(&[&pairs], None),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(dpo.regime, Regime::Dpo);
    assert_eq!(dpo.preference.unwrap().beta, DEFAULT_DPO_BETA);
    assert_eq!(*trainer.called.lock().unwrap(), [Regime::Sft, Regime::Dpo]);
}

#[test]
fn the_tuning_of_a_run_reaches_the_trainer_and_every_default_is_resolved_into_the_plan() {
    let (_scratch, ctx) = gate_context("train-tuning", Brain::Missing);
    let chat = dataset(&ctx, "alpha", FACTS);
    let trainer = FakeTrainer::knowing(&[ANCHOR, "alpha"]);
    train(
        &ctx,
        &request(&[&chat], None),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    // The plan resolves what the request left open and keeps what it named.
    let plan = trainer.plans.lock().unwrap()[0].clone();
    // `train` and `learn` share one learning rate, a scale of two whatever
    // the rank, and no weight decay, each recorded in the plan.
    assert!(!plan.tuning.bf16_base);
    assert_eq!(plan.tuning.learning_rate, Some(DEFAULT_LEARNING_RATE));
    assert_eq!(plan.tuning.alpha, Some(2.0 * plan.rank as f32));
    assert_eq!(plan.tuning.weight_decay, Some(DEFAULT_WEIGHT_DECAY));
    assert!(!Tuning::default().bf16_base && Tuning::default().learning_rate.is_none());

    let tuned = TrainRequest {
        tuning: Tuning {
            bf16_base: true,
            learning_rate: Some(1e-4),
            alpha: Some(64.0),
            weight_decay: Some(0.01),
            records_per_step: Some(4),
            ..Tuning::default()
        },
        rank: 32,
        ..request(&[&chat], None)
    };
    let trainer = FakeTrainer::knowing(&[ANCHOR, "alpha"]);
    train(&ctx, &tuned, &trainer, &CancelToken::new()).unwrap();
    let plan = trainer.plans.lock().unwrap()[0].clone();
    assert!(plan.tuning.bf16_base);
    assert_eq!(plan.tuning.learning_rate, Some(1e-4));
    assert_eq!((plan.rank, plan.tuning.alpha), (32, Some(64.0)));
    assert_eq!(plan.tuning.weight_decay, Some(0.01));
}

#[test]
fn a_machine_configured_to_hold_bases_at_bf16_trains_at_bf16_whichever_command_trains() {
    // A base too large for the card at fp32 is a fact about the machine: a
    // training that names no tuning still holds it at bf16, or it does not fit.
    let scratch = common::Scratch::new("train-machine-bf16");
    let mut settings = common::config(&scratch);
    settings.bf16_base = true;
    let ctx = Context::new(settings, false).unwrap();
    let chat = dataset(&ctx, "alpha", FACTS);
    let trainer = FakeTrainer::knowing(&[ANCHOR, "alpha"]);
    train(
        &ctx,
        &request(&[&chat], None),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    assert!(trainer.plans.lock().unwrap()[0].tuning.bf16_base);
}
