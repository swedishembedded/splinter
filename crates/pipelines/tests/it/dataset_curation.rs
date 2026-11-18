// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curation of agent demonstrations so that a
// student model does not learn its teacher's unsafe exploration, for its
// clients. If your team needs expertise in safe distillation of coding agents,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a dataset built from benchmark trials does not teach reading the
//! repository's history, and says so.
//!
//! * An sft-step build drops, by default, an action that reads the history;
//!   the report and the manifest count it and name the policy.
//! * `truncate` ends the demonstration there; `keep` learns it.
//! * A view that imitates no demonstration refuses the option.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::bench::{context, export, sft_request, Trial, NO_OPTIONS};
use splinter_data::Exclusion;
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::datasets::{
    build, resolve_dataset, BuildRequest, HistoryCuration, ViewName,
};

#[test]
fn a_history_read_in_a_trial_is_not_learned_and_the_dataset_says_so() {
    let (scratch, ctx) = context("dataset-curation");
    let dir = export(
        &scratch.0.join("export"),
        &[
            Trial::new("probe", "org/a", Some("train"), true)
                .command("git log --all --oneline -20 --graph"),
            Trial::new("plain", "org/b", Some("train"), true),
        ],
    );
    let train = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS)
        .unwrap()
        .train_set
        .unwrap();

    // Each trial is a tool call and an answer: the probe's call is dropped.
    let built = build(&ctx, &sft_request(train.clone())).unwrap();
    assert_eq!(built.records, 3);
    assert_eq!(built.excluded.get(&Exclusion::RepositoryHistory), Some(&1));
    let curation = built.curation.clone().unwrap();
    assert_eq!(
        (curation.policy.as_str(), curation.mode),
        ("no-repository-history", HistoryCuration::DropStep)
    );
    let stored = resolve_dataset(&ctx, built.dataset.0.as_str()).unwrap();
    assert_eq!(stored.manifest.curation, built.curation);
    assert!(!std::fs::read_to_string(&built.path)
        .unwrap()
        .contains("\"train\":true,\"tool_calls\":[{\"id\":\"s2-c0\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"{\\\"command\\\":\\\"git log"));

    let with = |history| BuildRequest {
        history,
        ..sft_request(train.clone())
    };
    let truncated = build(&ctx, &with(HistoryCuration::Truncate)).unwrap();
    assert_eq!(truncated.records, 2);
    assert_eq!(
        truncated.excluded.get(&Exclusion::AfterRepositoryHistory),
        Some(&1)
    );
    let kept = build(&ctx, &with(HistoryCuration::Keep)).unwrap();
    assert_eq!((kept.records, kept.curation), (4, None));

    let refused = build(
        &ctx,
        &BuildRequest {
            view: ViewName::PolicyGradient,
            ..with(HistoryCuration::Keep)
        },
    )
    .unwrap_err()
    .to_string();
    assert!(refused.contains("--repository-history"), "{refused}");
}
