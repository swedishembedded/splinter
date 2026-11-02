// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the trials of a benchmark, ingested, become a policy-gradient dataset.
//! Attempts at one task form a group whose rewards come from the benchmark's
//! grade (a pass, or the metrics of a failure under the configured shape);
//! every action of every attempt is a record that carries its attempt's
//! advantage. Held-out trials never reach it, and building it again gives the
//! same dataset.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::bench::{context, export, sft_request, Trial, NO_OPTIONS};
use serde_json::{json, Value};
use splinter_data::Exclusion;
use splinter_orchestrator::Context;
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::datasets::{
    build, build_dropping_leaks, AdvantageKind, BuildRequest, RlBuild, ViewName,
};
use splinter_store::experiences::{ExperienceSet, SetId};

fn request(set: SetId, rl: RlBuild) -> BuildRequest {
    BuildRequest {
        view: ViewName::PolicyGradient,
        rl,
        ..sft_request(set)
    }
}

fn shaped() -> RlBuild {
    RlBuild {
        reward_metrics: vec![("iou".into(), 0.75), ("feature_match".into(), 0.25)],
        fail_ceiling: Some(0.5),
        advantage: AdvantageKind::Standardised,
    }
}

fn measured(iou: f64, feature_match: f64) -> Value {
    json!({"iou": iou, "feature_match": feature_match})
}

/// Three train attempts at one task and a held-out one; returns the train set
/// and the set of both.
fn ingest_attempts(ctx: &Context, dir: &std::path::Path) -> (SetId, SetId) {
    let trials = [
        Trial::new("a-pass", "org/cad", Some("train"), true).metrics(measured(1.0, 1.0)),
        Trial::new("b-near", "org/cad", Some("train"), false).metrics(measured(0.8, 0.0)),
        Trial::new("c-far", "org/cad", Some("train"), false).metrics(measured(0.0, 0.0)),
        Trial::new("x-exam", "org/exam", Some("eval"), true).metrics(measured(1.0, 1.0)),
    ];
    let done = ingest_bench_trials(ctx, &[export(dir, &trials)], NO_OPTIONS).unwrap();
    assert!(done.refused.is_empty(), "{:?}", done.refused);
    let store = ctx.experiences();
    let train = done.train_set.unwrap();
    let mut members = store.get_set(&train).unwrap().members;
    members.extend(store.get_set(&done.eval_set.unwrap()).unwrap().members);
    let mixed = store
        .put_set(&ExperienceSet {
            name: "mixed".into(),
            members,
        })
        .unwrap();
    (train, mixed)
}

#[test]
fn attempts_are_rewarded_by_their_grade_and_every_action_carries_its_advantage() {
    let (scratch, ctx) = context("pg-build");
    let (train, _) = ingest_attempts(&ctx, &scratch.0.join("export"));
    let built = build(&ctx, &request(train, shaped())).unwrap();
    assert_eq!(built.view, "policy-gradient");
    // Two actions in each of three attempts.
    assert_eq!(built.records, 6);
    assert_eq!(built.groups.len(), 1);
    let group = &built.groups[0];
    assert_eq!(
        (group.task.as_str(), group.version.as_str()),
        ("org/cad", "2")
    );
    assert_eq!(group.policy, "qwen-27b");
    let by_trial = |name: &str| group.trials.iter().find(|t| t.trial == name).unwrap();
    assert!((by_trial("a-pass").reward - 1.0).abs() < 1e-12);
    // 0.5 * (0.75 * 0.8 + 0.25 * 0.0)
    assert!((by_trial("b-near").reward - 0.3).abs() < 1e-12);
    assert!(by_trial("c-far").reward.abs() < 1e-12);
    assert!(by_trial("a-pass").advantage > 0.0 && by_trial("c-far").advantage < 0.0);
    assert!(
        by_trial("a-pass").advantage > by_trial("b-near").advantage
            && by_trial("b-near").advantage > by_trial("c-far").advantage
    );

    let text = std::fs::read_to_string(&built.path).unwrap();
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 6);
    assert!(lines.iter().all(|l| l["advantage"].is_f64()));
}

#[test]
fn building_again_gives_the_same_dataset() {
    let (scratch, ctx) = context("pg-deterministic");
    let (train, _) = ingest_attempts(&ctx, &scratch.0.join("export"));
    let first = build(&ctx, &request(train.clone(), shaped())).unwrap();
    let again = build(&ctx, &request(train, shaped())).unwrap();
    assert_eq!(first.dataset, again.dataset);
}

#[test]
fn a_held_out_trial_is_refused_or_dropped_never_trained_on() {
    let (scratch, ctx) = context("pg-leak");
    let (_, mixed) = ingest_attempts(&ctx, &scratch.0.join("export"));
    let refused = build(&ctx, &request(mixed.clone(), shaped())).unwrap_err();
    assert!(refused.to_string().contains("leakage"), "{refused}");
    let built = build_dropping_leaks(&ctx, &request(mixed, shaped())).unwrap();
    assert_eq!(built.excluded.get(&Exclusion::ExamLeak), Some(&1));
    assert_eq!(built.records, 6);
    assert_eq!(built.groups.len(), 1);
}

#[test]
fn a_failed_attempt_without_a_configured_metric_stops_the_build_naming_it() {
    let (scratch, ctx) = context("pg-missing-metric");
    let (train, _) = ingest_attempts(&ctx, &scratch.0.join("export"));
    let rl = RlBuild {
        reward_metrics: vec![("chamfer".into(), 1.0)],
        ..shaped()
    };
    let error = build(&ctx, &request(train, rl)).unwrap_err().to_string();
    assert!(
        error.contains("chamfer") && error.contains("b-near"),
        "{error}"
    );
}

#[test]
fn reward_options_belong_to_the_policy_gradient_view_alone() {
    let (scratch, ctx) = context("pg-options");
    let (train, _) = ingest_attempts(&ctx, &scratch.0.join("export"));
    let wrong_view = BuildRequest {
        rl: shaped(),
        ..sft_request(train.clone())
    };
    let error = build(&ctx, &wrong_view).unwrap_err().to_string();
    assert!(error.contains("--reward-metric"), "{error}");
    let ceiling_alone = RlBuild {
        fail_ceiling: Some(0.5),
        ..RlBuild::default()
    };
    let error = build(&ctx, &request(train, ceiling_alone))
        .unwrap_err()
        .to_string();
    assert!(error.contains("--fail-ceiling"), "{error}");
}
