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

use crate::common;

use common::bench::{context, export, sft_request, Trial, NO_OPTIONS};
use serde_json::{json, Value};
use splinter_data::Exclusion;
use splinter_orchestrator::Context;
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::datasets::{
    build, build_dropping_leaks, AdvantageKind, BuildRequest, RewardPreset, RlBuild, ViewName,
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
        ..RlBuild::default()
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
    assert!(
        group.trials.iter().all(|t| t.written),
        "every attempt of a graded group is written"
    );

    let text = std::fs::read_to_string(&built.path).unwrap();
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 6);
    assert!(lines
        .iter()
        .all(|l| l["reward"].is_f64() && l["group"].is_string() && l["rollout"].is_string()));
}

#[test]
fn packed_every_attempt_is_one_record_with_its_reward_group_and_rollout() {
    let (scratch, ctx) = context("pg-packed");
    let (train, _) = ingest_attempts(&ctx, &scratch.0.join("export"));
    let built = build(
        &ctx,
        &BuildRequest {
            pack: true,
            ..request(train, shaped())
        },
    )
    .unwrap();
    // Three attempts, one record each, both actions of each supervised.
    assert_eq!(built.records, 3);
    let lines = lines(&built.path);
    assert!(lines
        .iter()
        .all(|l| l["reward"].is_f64() && l["group"].is_string() && l["rollout"].is_string()));
    for line in &lines {
        let trained = line["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["train"] == true)
            .count();
        assert_eq!(trained, 2, "{line}");
    }
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
        split_seed: None,
        history: Default::default(),
        base: None,
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

const WHY: &str = "a check failed: python3 -m unittest tests.hidden (exit 1)";

/// An all-fail group (the cliff) and a graded one, both train; an eval attempt.
fn ingest_cliff(ctx: &Context, dir: &std::path::Path) -> (SetId, SetId) {
    let trials = [
        Trial::new("a1", "org/stuck", Some("train"), false).reason(WHY),
        Trial::new("a2", "org/stuck", Some("train"), false).reason(WHY),
        Trial::new("b-pass", "org/cad", Some("train"), true),
        Trial::new("b-fail", "org/cad", Some("train"), false).reason(WHY),
        Trial::new("x-exam", "org/exam", Some("eval"), false).reason(WHY),
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

fn lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn the_cliff_group_is_distilled_with_the_verifiers_reason_and_the_graded_group_keeps_its_advantage()
{
    let (scratch, ctx) = context("pg-cliff");
    let (train, _) = ingest_cliff(&ctx, &scratch.0.join("export"));
    let rl = RlBuild {
        cliff_distill: true,
        ..RlBuild::default()
    };
    let built = build(&ctx, &request(train, rl)).unwrap();
    assert_eq!(built.view, "policy-gradient");
    let kinds: Vec<(String, &str, bool)> = built
        .groups
        .iter()
        .map(|g| (g.task.clone(), g.kind.name(), g.distilled))
        .collect();
    assert!(
        kinds.contains(&("org/stuck".into(), "all-fail", true)),
        "{kinds:?}"
    );
    assert!(
        kinds.contains(&("org/cad".into(), "graded", false)),
        "{kinds:?}"
    );

    let all = lines(&built.path);
    let (taught, scored): (Vec<&Value>, Vec<&Value>) =
        all.iter().partition(|l| l.get("teacher_context").is_some());
    // Two actions in each of the two cliff attempts; two in each of the graded two.
    assert_eq!((taught.len(), scored.len()), (4, 4));
    assert!(scored
        .iter()
        .all(|l| l["reward"].is_f64() && l["group"].is_string()));
    for line in taught {
        assert!(line.get("reward").is_none());
        let text = line["teacher_context"]["messages"][0]["content"]
            .as_str()
            .unwrap();
        assert!(text.contains(WHY), "{text}");
        // The student's turns never hold the verifier's words.
        assert!(!line["messages"].to_string().contains("tests.hidden"));
    }
}

#[test]
fn the_self_distill_view_takes_every_attempt_and_keeps_the_exam_sealed() {
    let (scratch, ctx) = context("pg-self-distill");
    let (train, mixed) = ingest_cliff(&ctx, &scratch.0.join("export"));
    // The passed attempt carries no reason and no reference: it has nothing to show a teacher.
    let request = |set| BuildRequest {
        view: ViewName::SelfDistill,
        ..request(set, RlBuild::default())
    };
    let built = build(&ctx, &request(train)).unwrap();
    assert_eq!(built.view, "self-distill");
    assert_eq!(
        built.excluded.get(&Exclusion::NoPrivilegedContext),
        Some(&1)
    );
    assert_eq!(
        built.records, 6,
        "three failed attempts of two actions each"
    );
    assert!(lines(&built.path)
        .iter()
        .all(|l| l.get("teacher_context").is_some() && l.get("reward").is_none()));

    let refused = build(&ctx, &request(mixed.clone())).unwrap_err();
    assert!(refused.to_string().contains("leakage"), "{refused}");
    let dropped = build_dropping_leaks(&ctx, &request(mixed)).unwrap();
    assert_eq!(dropped.excluded.get(&Exclusion::ExamLeak), Some(&1));
    assert_eq!(dropped.records, 6);
}

#[test]
fn cliff_distillation_belongs_to_the_policy_gradient_view_alone() {
    let (scratch, ctx) = context("pg-cliff-refused");
    let (train, _) = ingest_cliff(&ctx, &scratch.0.join("export"));
    let rl = RlBuild {
        cliff_distill: true,
        ..RlBuild::default()
    };
    let wrong = BuildRequest {
        view: ViewName::SelfDistill,
        ..request(train.clone(), rl.clone())
    };
    assert!(build(&ctx, &wrong)
        .unwrap_err()
        .to_string()
        .contains("--cliff-distill"));
    let sft = BuildRequest {
        view: ViewName::SftStep,
        ..request(train, rl)
    };
    assert!(build(&ctx, &sft)
        .unwrap_err()
        .to_string()
        .contains("--cliff-distill"));
}

/// Practice material staging the fixture's task (content digest `d1`, as
/// every fixture trial records it) with the naive floor `floor` and the
/// lineage `breach/late`.
fn practice_with_floor(dir: &std::path::Path, floor: f64) -> std::path::PathBuf {
    let task = dir.join("tasks/cad-1");
    std::fs::create_dir_all(&task).unwrap();
    std::fs::write(
        task.join("task.toml"),
        "[metadata]\nbench_defect_lineage = \"breach/late\"\n",
    )
    .unwrap();
    let practice = json!({
        "schema": "bench.practice/2", "collection": "tasks/cad", "set": "train",
        "environment": "userns",
        "tasks": [{"name": "cad-1", "digest": "s1", "source_digest": "d1", "naive_floor": floor}],
    });
    std::fs::write(dir.join("practice.json"), practice.to_string()).unwrap();
    dir.to_path_buf()
}

/// Spec (`--reward shaped-partial`): a failure earns `c * max(0, (f -
/// f_naive) / (1 - f_naive))` of its `functional_correctness` above its
/// task's naive floor from the practice material; the manifest records the
/// floors and where they came from; without material every floor is zero and
/// the manifest says so.
#[test]
fn shaped_partial_credits_a_failure_above_its_tasks_naive_floor() {
    let (scratch, ctx) = context("pg-shaped-partial");
    let f = |v: f64| json!({"functional_correctness": v});
    let trials = [
        Trial::new("a-pass", "org/cad", Some("train"), true).metrics(f(1.0)),
        Trial::new("b-most", "org/cad", Some("train"), false).metrics(f(0.75)),
        Trial::new("c-naive", "org/cad", Some("train"), false).metrics(f(0.2)),
    ];
    let done = ingest_bench_trials(
        &ctx,
        &[export(&scratch.0.join("export"), &trials)],
        NO_OPTIONS,
    )
    .unwrap();
    let train = done.train_set.unwrap();
    let rl = RlBuild {
        reward: RewardPreset::ShapedPartial,
        ..RlBuild::default()
    };
    let floored = BuildRequest {
        practice: vec![practice_with_floor(&scratch.0.join("practice"), 0.5)],
        ..request(train.clone(), rl.clone())
    };
    let built = build(&ctx, &floored).unwrap();
    let group = &built.groups[0];
    let reward = |name: &str| {
        group
            .trials
            .iter()
            .find(|t| t.trial == name)
            .unwrap()
            .reward
    };
    assert!((reward("a-pass") - 1.0).abs() < 1e-12);
    // 0.5 * (0.75 - 0.5) / (1 - 0.5)
    assert!((reward("b-most") - 0.25).abs() < 1e-12);
    assert!(
        reward("c-naive").abs() < 1e-12,
        "at the floor earns nothing"
    );
    assert_eq!(group.naive_floor, Some(0.5));
    // The build reports what the group can teach per token, by its lineage.
    let signal = built.signal.as_ref().expect("a signal report");
    assert_eq!(signal.families[0].family, "breach/late");
    assert_eq!(signal.uniform.informative_share, Some(1.0));
    assert!(signal.tasks[0].learning_signal.is_some_and(|ls| ls > 0.0));
    let manifest = splinter_pipelines::datasets::resolve_dataset(&ctx, &built.dataset.to_string())
        .unwrap()
        .manifest;
    let credit = manifest.partial_credit.expect("the manifest records it");
    assert_eq!(credit.metric, "functional_correctness");
    assert_eq!(credit.sources.len(), 1);
    assert_eq!(credit.groups_without_floor, 0);

    let unfloored = build(&ctx, &request(train.clone(), rl)).unwrap();
    let group = &unfloored.groups[0];
    let reward = |name: &str| {
        group
            .trials
            .iter()
            .find(|t| t.trial == name)
            .unwrap()
            .reward
    };
    assert!((reward("b-most") - 0.375).abs() < 1e-12);
    let manifest =
        splinter_pipelines::datasets::resolve_dataset(&ctx, &unfloored.dataset.to_string())
            .unwrap()
            .manifest;
    let credit = manifest.partial_credit.unwrap();
    assert!(credit.sources.is_empty());
    assert_eq!(credit.groups_without_floor, 1);

    let mixed = RlBuild {
        reward: RewardPreset::ShapedPartial,
        reward_metrics: vec![("iou".into(), 1.0)],
        ..RlBuild::default()
    };
    let error = build(&ctx, &request(train, mixed)).unwrap_err().to_string();
    assert!(error.contains("shaped-partial"), "{error}");
}
