// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out evaluation that measures what a
// fine-tune learned across every domain it was trained on, for its clients.
// If your team needs expertise in leakage-free train/held-out splits for
// agent training data, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a dataset built from benchmark trials is split by seed and stratum.
//!
//! * Every record carries the seed of its dataset's split and its stratum, the
//!   domain its task declares; the manifest and the build report record the
//!   rule, the seed and the records of each stratum.
//! * Trials ingested one domain after another (the pilot's trap: the newest
//!   tenth of the file was all one domain) are held out at every domain's
//!   share, and so are the monitoring records carved from the rest.
//! * The seed is derived from the experiences unless given, so a rebuild splits
//!   the same; another seed splits otherwise. Two trials of one task are one
//!   group, never divided between training and scoring.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use common::bench::{context, export, sft_request, Trial, NO_OPTIONS};
use splinter_data::holdout::{monitor_split_file, split_dataset_file};
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::datasets::{build, resolve_dataset, BuildRequest};

const DOMAINS: [&str; 4] = ["support", "compiled", "legal", "accounting"];

/// Ten passing train trials per domain, one domain after another, as an
/// export lists them; the last task of each domain is tried twice.
fn trials() -> Vec<Trial> {
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    let mut trials = Vec::new();
    for domain in DOMAINS {
        for n in 0..10 {
            let task = leak(format!("org/{domain}-{n}"));
            let dimension = leak(format!("{domain}.{domain}-{n}"));
            trials.push(
                Trial::new(leak(format!("{domain}-{n}")), task, Some("train"), true)
                    .dimension(dimension),
            );
        }
        let task = leak(format!("org/{domain}-9"));
        trials.push(
            Trial::new(leak(format!("{domain}-9b")), task, Some("train"), true)
                .dimension(leak(format!("{domain}.{domain}-9"))),
        );
    }
    trials
}

/// The records of a file per stratum.
fn strata(path: &Path) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for line in fs::read_to_string(path).unwrap().lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        let stratum = record["metadata"]["split_draw"]["stratum"]
            .as_str()
            .unwrap()
            .to_string();
        *counts.entry(stratum).or_default() += 1;
    }
    counts
}

#[test]
fn trials_ingested_one_domain_after_another_are_held_out_and_monitored_across_every_domain() {
    let (scratch, ctx) = context("dataset-strata");
    let dir = export(&scratch.0.join("export"), &trials());
    let train = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS)
        .unwrap()
        .train_set
        .unwrap();
    let request = BuildRequest {
        pack: true,
        ..sft_request(train.clone())
    };
    let built = build(&ctx, &request).unwrap();
    assert_eq!(built.records, 44);
    let split = built.split.clone().unwrap();
    assert_eq!(split.rule, "seeded-stratified");
    let expected: BTreeMap<String, usize> = DOMAINS.iter().map(|d| (d.to_string(), 11)).collect();
    assert_eq!(split.strata, expected);
    let stored = resolve_dataset(&ctx, built.dataset.0.as_str()).unwrap();
    assert_eq!(
        stored.manifest.split, built.split,
        "the manifest records it"
    );

    // The held-out set and the monitoring set carved from the rest hold
    // every domain; the two trials of one task are on one side.
    let parts = scratch.0.join("parts");
    fs::create_dir_all(&parts).unwrap();
    let halves = split_dataset_file(&built.path, &parts).unwrap();
    let held = strata(&halves.held_out);
    // Eight tasks (the units a paired test needs), two of each domain; a
    // task tried twice brings both its records.
    assert_eq!(held.len(), 4, "{held:?}");
    assert!(held.values().all(|&n| n == 2 || n == 3), "{held:?}");
    let monitor = monitor_split_file(&halves.train, &parts, 0.2).unwrap();
    assert_eq!(strata(&monitor.monitor).len(), 4);
    let text = |p: &Path| fs::read_to_string(p).unwrap();
    for domain in DOMAINS {
        let group = format!("\"group\":\"bench:org/{domain}-9\"");
        assert!(
            !(text(&halves.held_out).contains(&group) && text(&halves.train).contains(&group)),
            "{domain}: one task on both sides"
        );
    }

    // A rebuild splits the same; another seed draws another split.
    assert_eq!(build(&ctx, &request).unwrap().split, built.split);
    let other_seed = BuildRequest {
        split_seed: Some(split.seed.wrapping_add(1)),
        ..request
    };
    let other = build(&ctx, &other_seed).unwrap();
    assert_eq!(other.split.unwrap().seed, split.seed.wrapping_add(1));
    let other_parts = scratch.0.join("other");
    fs::create_dir_all(&other_parts).unwrap();
    let other_halves = split_dataset_file(&other.path, &other_parts).unwrap();
    assert_ne!(text(&other_halves.held_out), text(&halves.held_out));
}
