// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training runs that stop where their
// held-out loss turns and carry the best checkpoint, for its clients. If your
// team needs expertise in overfit control for small-data fine-tunes, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a supervised run is sized as a ceiling and watched as it trains.
//!
//! * A request that names no steps gets a budget of passes over the
//!   datasets' examples, a step that averages several records, a monitoring
//!   cadence over that budget, the default patience and a monitoring share;
//!   what it names is kept, and a cadence of zero monitors nothing.
//! * The candidate records the curve the trainer reports, the step it
//!   carries, and what the curve warns of; the release gate repeats the
//!   warnings beside its verdict instead of passing silently.
//! * A monitoring share outside (0, 1/2] is refused before anything trains.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{
    anchor_file, dataset, decide, gate_context, policy, Brain, FakeTrainer, ANCHOR,
};
use splinter_agent::CancelToken;
use splinter_core::training::{CurvePoint, Selection, TrainingCurve};
use splinter_pipelines::datasets::examples_in;
use splinter_pipelines::release::anchor;
use splinter_pipelines::train::{
    auto_records_per_step, eval_every_for, steps_for, train, TrainRequest, Tuning,
    DEFAULT_MONITOR_SHARE, DEFAULT_PATIENCE, DEFAULT_REPLAY_FRACTION,
};

fn request(data: &str, steps: Option<u32>, tuning: Tuning) -> TrainRequest {
    TrainRequest {
        datasets: vec![data.to_string()],
        from: policy(),
        replay_fraction: DEFAULT_REPLAY_FRACTION,
        steps,
        rank: 4,
        beta: None,
        tuning,
    }
}

#[test]
fn a_run_without_named_steps_gets_a_budget_a_cadence_and_a_monitor_and_keeps_what_it_names() {
    let (_scratch, ctx) = gate_context("monitoring-plan", Brain::Honest);
    let data = dataset(&ctx, "alpha", 40);
    let trainer = FakeTrainer::knowing(&["alpha"]);
    let candidate = train(
        &ctx,
        &request(&data.to_string(), None, Tuning::default()),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    let plan = trainer.plans.lock().unwrap()[0].clone();
    let examples = examples_in(&plan.datasets[0].path).unwrap();
    let records_per_step = auto_records_per_step(examples);
    assert_eq!(plan.tuning.records_per_step, Some(records_per_step));
    assert_eq!(
        plan.steps,
        steps_for(examples, records_per_step),
        "a ceiling of passes"
    );
    assert_eq!(plan.eval_every, eval_every_for(plan.steps));
    assert_eq!(plan.patience, DEFAULT_PATIENCE);
    assert_eq!(plan.monitor_share, DEFAULT_MONITOR_SHARE);
    assert_eq!(
        (
            plan.tuning.eval_every,
            plan.tuning.patience,
            plan.tuning.monitor_share
        ),
        (
            Some(plan.eval_every),
            Some(plan.patience),
            Some(plan.monitor_share)
        ),
        "the plan's tuning is resolved"
    );
    assert_eq!(candidate.steps, plan.steps);
    let curve = candidate
        .curve
        .as_ref()
        .expect("a supervised candidate records its curve");
    assert_eq!(
        (curve.selected_step, curve.selection),
        (plan.steps, Selection::LastStep)
    );
    assert!(candidate.warnings().is_empty());

    let named = Tuning {
        eval_every: Some(0),
        patience: Some(0),
        monitor_share: Some(0.3),
        records_per_step: Some(2),
        ..Tuning::default()
    };
    let trainer = FakeTrainer::knowing(&["alpha"]);
    train(
        &ctx,
        &request(&data.to_string(), Some(7), named),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    let plan = trainer.plans.lock().unwrap()[0].clone();
    assert_eq!(
        (
            plan.steps,
            plan.eval_every,
            plan.patience,
            plan.monitor_share,
            plan.tuning.records_per_step
        ),
        (7, 0, 0, 0.3, Some(2))
    );
}

#[test]
fn the_gate_repeats_what_the_curve_warns_of() {
    let (scratch, ctx) = gate_context("monitoring-warnings", Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    let data = dataset(&ctx, "alpha", 40);
    let trainer = FakeTrainer::knowing(&[ANCHOR, "alpha"]);
    // A run that carried its last step although the monitoring loss had
    // turned at step 20, with a gap of more than a quarter at the end.
    let point = |step, train_loss, monitor_loss| CurvePoint {
        step,
        train_loss,
        monitor_loss,
    };
    *trainer.curve.lock().unwrap() = Some(TrainingCurve {
        steps: 40,
        steps_completed: 40,
        eval_every: 10,
        patience: 0,
        monitor_records: 4,
        points: vec![
            point(10, 2.0, 1.8),
            point(20, 1.4, 1.3),
            point(30, 1.0, 1.35),
            point(40, 0.7, 1.4),
        ],
        selected_step: 40,
        selection: Selection::LastStep,
        stopped_early: false,
    });
    let candidate = train(
        &ctx,
        &request(&data.to_string(), Some(40), Tuning::default()),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    let warnings = candidate.warnings();
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(
        warnings[0].contains("generalisation gap"),
        "{}",
        warnings[0]
    );
    assert!(
        warnings[1].contains("rose before the end"),
        "{}",
        warnings[1]
    );

    let released = decide(&ctx, &candidate);
    assert!(released.gate.passed, "{:?}", released.gate);
    assert_eq!(released.warnings, warnings, "a pass repeats the warnings");
    let manifest = ctx
        .releases()
        .get(released.release.as_ref().unwrap())
        .unwrap()
        .manifest;
    assert_eq!(
        manifest.training.curve.as_ref().map(|c| c.selected_step),
        Some(40)
    );
}

#[test]
fn a_monitor_share_outside_its_range_is_refused() {
    let (_scratch, ctx) = gate_context("monitoring-share", Brain::Honest);
    let data = dataset(&ctx, "alpha", 8);
    for share in [0.0, 0.6] {
        let tuning = Tuning {
            monitor_share: Some(share),
            ..Tuning::default()
        };
        let trainer = FakeTrainer::knowing(&["alpha"]);
        let refused = train(
            &ctx,
            &request(&data.to_string(), None, tuning),
            &trainer,
            &CancelToken::new(),
        )
        .unwrap_err();
        assert!(
            refused.is_refusal() && refused.to_string().contains("monitor share"),
            "{refused}"
        );
        assert!(
            trainer.plans.lock().unwrap().is_empty(),
            "refused before anything trains"
        );
    }
}
