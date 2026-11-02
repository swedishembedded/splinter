// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for reinforcement learning from rollouts: building a
//! `policy-gradient` dataset.

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
        // A ceiling bounds the metrics' reward; without metrics it means nothing.
        &["--fail-ceiling", "0.4"][..],
        &["--reward-metric", "iou"],
        &["--reward-metric", "iou=2"],
        &["--advantage", "raw"],
    ] {
        let mut argv = vec!["dataset", "build", "ab12", "--view", "policy-gradient"];
        argv.extend_from_slice(refused);
        assert!(parse(&argv).is_err(), "{refused:?}");
    }
}
