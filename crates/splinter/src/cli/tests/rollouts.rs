// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for reinforcement learning from rollouts: building a
//! `policy-gradient` dataset and training one.

use splinter_sdk::model::policy_gradient::{LossAggregation, Reference};

use super::{command, parse, Command, DatasetCommand};

#[test]
fn a_policy_gradient_dataset_is_built_with_a_shaped_reward_and_a_chosen_advantage() {
    let Command::Dataset(DatasetCommand::Build(args)) = command(&[
        "dataset",
        "build",
        "ab12",
        "--view",
        "policy-gradient",
        "--reward-metric",
        "iou=0.5",
        "--reward-metric",
        "feature_match=0.25",
        "--reward-metric",
        "deviation_score=0.25",
        "--fail-ceiling",
        "0.4",
        "--advantage",
        "mean-centred",
        "--roles",
        "designer",
        "--images",
    ]) else {
        panic!("dataset build");
    };
    assert_eq!(args.view.as_str(), "policy-gradient");
    assert_eq!(
        args.reward_metrics,
        [
            ("iou".to_string(), 0.5),
            ("feature_match".to_string(), 0.25),
            ("deviation_score".to_string(), 0.25)
        ]
    );
    assert_eq!(args.fail_ceiling, Some(0.4));
    assert_eq!(
        args.advantage,
        Some(splinter_sdk::datasets::AdvantageKind::MeanCentred)
    );
    for refused in [
        &["--reward-metric", "iou"][..],
        &["--reward", "partial"],
        &["--reward-metric", "iou=2"],
        &["--advantage", "raw"],
    ] {
        let mut argv = vec!["dataset", "build", "ab12", "--view", "policy-gradient"];
        argv.extend_from_slice(refused);
        assert!(parse(&argv).is_err(), "{refused:?}");
    }
}

#[test]
fn the_shaped_partial_reward_is_named_with_its_credit_and_the_practice_its_floors_come_from() {
    let Command::Dataset(DatasetCommand::Build(args)) = command(&[
        "dataset",
        "build",
        "ab12",
        "--view",
        "policy-gradient",
        "--reward",
        "shaped-partial",
        "--fail-ceiling",
        "0.5",
        "--practice",
        "staged/legal",
        "--practice",
        "staged/support",
    ]) else {
        panic!("dataset build");
    };
    assert_eq!(
        args.reward,
        Some(splinter_sdk::datasets::RewardPreset::ShapedPartial)
    );
    assert_eq!(args.fail_ceiling, Some(0.5));
    assert_eq!(args.practice.len(), 2);
}

#[test]
fn a_run_on_rollouts_is_set_with_brains_policy_gradient_settings() {
    let Command::Train(train) = command(&[
        "train",
        "ab12",
        "--from",
        "local:./ckpt+round-3.safetensors",
        "--clip-eps",
        "0.2",
        "--kl-beta",
        "0.02",
        "--reference",
        "start",
        "--loss-agg",
        "token-sum",
        "--temperature",
        "0.7",
    ]) else {
        panic!("train");
    };
    let options = train.policy_gradient.options(train.temperature);
    assert_eq!(
        (options.clip_eps, options.kl_beta, options.temperature),
        (Some(0.2), Some(0.02), Some(0.7))
    );
    assert_eq!(options.reference, Some(Reference::StartOfRun));
    assert_eq!(options.loss_agg, Some(LossAggregation::TokenSum));
    // Left out, they are brain's defaults.
    let Command::Train(plain) = command(&["train", "ab12"]) else {
        panic!("train");
    };
    assert!(plain
        .policy_gradient
        .options(plain.temperature)
        .named()
        .is_empty());
    for refused in [
        &["--clip-eps", "1.5"][..],
        &["--clip-eps", "0"],
        &["--kl-beta", "-1"],
        &["--reference", "adapter"],
        &["--loss-agg", "mean"],
        &["--temperature", "0"],
    ] {
        let mut argv = vec!["train", "ab12"];
        argv.extend_from_slice(refused);
        assert!(parse(&argv).is_err(), "{refused:?}");
    }
}

#[test]
fn self_distillation_is_built_from_the_cliff_groups_and_trained_with_named_guards() {
    let Command::Dataset(DatasetCommand::Build(args)) = command(&[
        "dataset",
        "build",
        "ab12",
        "--view",
        "policy-gradient",
        "--cliff-distill",
    ]) else {
        panic!("dataset build");
    };
    assert!(args.cliff_distill);
    // The view that distils every attempt needs no flag.
    let Command::Dataset(DatasetCommand::Build(every)) =
        command(&["dataset", "build", "ab12", "--view", "self-distill"])
    else {
        panic!("dataset build");
    };
    assert_eq!(every.view.as_str(), "self-distill");

    let Command::Train(train) = command(&[
        "train",
        "ab12",
        "--teacher-policy",
        "start",
        "--advantage-bound",
        "5",
        "--sequence-advantage-cap",
        "40",
    ]) else {
        panic!("train");
    };
    let options = train.policy_gradient.options(train.temperature);
    assert_eq!(options.teacher, Some(Reference::StartOfRun));
    assert_eq!(
        (options.advantage_bound, options.sequence_advantage_cap),
        (Some(5.0), Some(40.0))
    );
    for refused in [
        &["--teacher-policy", "adapter"][..],
        &["--advantage-bound", "0"],
        &["--advantage-bound", "-1"],
        &["--sequence-advantage-cap", "0"],
    ] {
        let mut argv = vec!["train", "ab12"];
        argv.extend_from_slice(refused);
        assert!(parse(&argv).is_err(), "{refused:?}");
    }
}

#[test]
fn a_run_on_rollouts_skips_the_steps_its_guards_name() {
    // A step whose mean KL or gradient norm is over its bound is skipped by
    // brain and recorded, so one outlier batch cannot throw the policy off.
    let Command::Train(train) = command(&[
        "train",
        "ab12",
        "--max-step-kl",
        "0.1",
        "--max-step-grad-norm",
        "5",
    ]) else {
        panic!("train");
    };
    let options = train.policy_gradient.options(train.temperature);
    assert_eq!(
        (options.max_step_kl, options.max_step_grad_norm),
        (Some(0.1), Some(5.0))
    );
    for refused in [
        &["--max-step-kl", "0"][..],
        &["--max-step-kl", "-1"],
        &["--max-step-grad-norm", "0"],
    ] {
        let mut argv = vec!["train", "ab12"];
        argv.extend_from_slice(refused);
        assert!(parse(&argv).is_err(), "{refused:?}");
    }
}

#[test]
fn a_policy_gradient_dataset_can_train_only_the_tokens_the_policy_sampled() {
    let Command::Dataset(DatasetCommand::Build(args)) = command(&[
        "dataset",
        "build",
        "ab12",
        "--view",
        "policy-gradient",
        "--sampled-only",
    ]) else {
        panic!("dataset build");
    };
    assert!(args.sampled_only);
    let Command::Dataset(DatasetCommand::Build(plain)) =
        command(&["dataset", "build", "ab12", "--view", "policy-gradient"])
    else {
        panic!("dataset build");
    };
    assert!(!plain.sampled_only);
}

#[test]
fn a_policy_gradient_dataset_takes_dapos_over_length_penalty() {
    let Command::Dataset(DatasetCommand::Build(args)) = command(&[
        "dataset",
        "build",
        "ab12",
        "--view",
        "policy-gradient",
        "--overlong-max-tokens",
        "16384",
        "--overlong-buffer",
        "4096",
    ]) else {
        panic!("dataset build");
    };
    assert_eq!(
        (args.overlong_max_tokens, args.overlong_buffer),
        (Some(16384), Some(4096))
    );
    // A buffer is a stretch before a budget; it needs one, and neither is zero.
    assert!(parse(&[
        "dataset",
        "build",
        "ab12",
        "--view",
        "policy-gradient",
        "--overlong-buffer",
        "8"
    ])
    .is_err());
    assert!(parse(&[
        "dataset",
        "build",
        "ab12",
        "--view",
        "policy-gradient",
        "--overlong-max-tokens",
        "0"
    ])
    .is_err());
}

#[test]
fn a_run_on_rollouts_names_its_recipe() {
    let Command::Train(train) = command(&["train", "ab12", "--recipe", "on-policy-rl"]) else {
        panic!("train");
    };
    assert_eq!(train.recipe, Some(splinter_sdk::train::Recipe::OnPolicyRl));
    assert!(parse(&["train", "ab12", "--recipe", "grpo"]).is_err());
}
