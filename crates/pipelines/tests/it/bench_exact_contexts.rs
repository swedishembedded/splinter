// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements on-policy training from the exact prompts an
// agent's model was served, for its clients. If your team needs expertise in
// importance-weighted training on agent rollouts, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Spec: a single agent's benchmark trial whose request log holds every
//! request its model was sent becomes records read from those requests, the
//! harness's own user-turn lines (the time line) included, so every traced
//! action of a policy-gradient dataset carries the ids the policy was given
//! and sampled and the build counts it exact. Without the log divided among
//! its agent steps the context is a reconstruction, and counted as one.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::bench::{context, export, sft_request, Trial, NO_OPTIONS, TIME_LINE};
use serde_json::Value;
use splinter_data::ProxyReason;
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::datasets::{build, BuildRequest, RlBuild, ViewName};

fn policy_gradient(
    ctx: &splinter_orchestrator::Context,
    trials: &[Trial],
    dir: &std::path::Path,
) -> splinter_pipelines::datasets::Built {
    let done = ingest_bench_trials(ctx, &[export(dir, trials)], NO_OPTIONS).unwrap();
    assert!(done.refused.is_empty(), "{:?}", done.refused);
    let request = BuildRequest {
        view: ViewName::PolicyGradient,
        rl: RlBuild::default(),
        ..sft_request(done.train_set.unwrap())
    };
    build(ctx, &request).unwrap()
}

#[test]
fn a_trial_whose_requests_were_recorded_trains_every_traced_action_exactly() {
    let (scratch, ctx) = context("bench-exact-contexts");
    let trials = [
        Trial::new("a-pass", "org/fw", Some("train"), true).traced(),
        Trial::new("b-fail", "org/fw", Some("train"), false).traced(),
    ];
    let built = policy_gradient(&ctx, &trials, &scratch.0.join("export"));
    assert_eq!(built.records, 4, "two actions in each of two attempts");
    assert_eq!(built.tokens.exact, 4, "{:?}", built.tokens);
    assert!(built.tokens.proxy.is_empty(), "{:?}", built.tokens);

    let text = std::fs::read_to_string(&built.path).unwrap();
    for line in text
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
    {
        let messages = line["messages"].as_array().unwrap();
        let user = messages.iter().find(|m| m["role"] == "user").unwrap();
        assert!(
            user["content"].as_str().unwrap().ends_with(TIME_LINE),
            "the instruction as sent: {user}"
        );
        let trained: Vec<&Value> = messages.iter().filter(|m| m["train"] == true).collect();
        assert_eq!(trained.len(), 1);
        assert!(
            trained[0]["prompt_token_ids"].is_array() && trained[0]["token_ids"].is_array(),
            "{line}"
        );
        assert_eq!(line["tokenizer_digest"], "sha256:aa");
    }
}

#[test]
fn a_trial_without_its_requests_divided_among_its_steps_is_counted_as_reconstructed() {
    let (scratch, ctx) = context("bench-reconstructed-contexts");
    let trials = [
        Trial::new("a-pass", "org/fw", Some("train"), true),
        Trial::new("b-fail", "org/fw", Some("train"), false),
    ];
    let built = policy_gradient(&ctx, &trials, &scratch.0.join("export"));
    assert_eq!(built.records, 4);
    assert_eq!(built.tokens.exact, 0);
    assert_eq!(
        built.tokens.proxy.get(&ProxyReason::ContextReconstructed),
        Some(&4),
        "{:?}",
        built.tokens
    );
}
