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
use splinter_campaign::release::{anchor, ReleaseStore};
use splinter_campaign::train::{
    train, Regime, TrainRequest, DEFAULT_DPO_BETA, DEFAULT_REPLAY_FRACTION,
};
use splinter_campaign::Context;
use splinter_views::DatasetId;
use sven_sdk::CancelToken;

fn request(datasets: &[&DatasetId], beta: Option<f32>) -> TrainRequest {
    TrainRequest {
        datasets: datasets.iter().map(ToString::to_string).collect(),
        from: policy(),
        replay_fraction: DEFAULT_REPLAY_FRACTION,
        steps: 1,
        rank: 4,
        beta,
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
    anchor::freeze(&ctx, &anchor_file(&scratch.0, 4)).unwrap();
    let champion = released(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let champion_adapter = ReleaseStore::open(ctx.root())
        .get(&champion)
        .unwrap()
        .adapter;

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
    let release = ReleaseStore::open(ctx.root())
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
