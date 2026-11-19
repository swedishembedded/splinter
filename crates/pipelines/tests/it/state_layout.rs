// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable learning state that fits on
// quota-limited shared filesystems, for its clients. If your team needs
// expertise in migrating stores without changing a dataset, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec (golden): how the experience database lies on disk changes nothing a
//! dataset is built from. The same bench export ingested into a loose state
//! and into a packed one gives the same experience set and the same dataset
//! id, and so does the loose state after `state pack`.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::bench::{self, Trial};
use splinter_core::clock::FixedClock;
use splinter_expdb::backend::Layout;
use splinter_expdb::{Config as DbConfig, Database};
use splinter_orchestrator::Context;
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::datasets::build;
use splinter_store::experiences::SetId;
use splinter_store::workspace::Workspace;

fn trials() -> Vec<Trial> {
    vec![
        Trial::new("one", "org/a", Some("train"), true).scored(),
        Trial::new("two", "org/b", Some("train"), true).scored(),
        Trial::new("three", "org/c", Some("train"), false).scored(),
        Trial::new("four", "org/d", Some("eval"), true).scored(),
    ]
}

fn context(scratch: &common::Scratch) -> Context {
    Context::new(common::config(scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-28T09:00:00.000Z")))
}

/// Ingests the export into the scratch state and builds the sft-step
/// dataset of its train set: (set id, dataset id, records).
fn ingest_and_build(scratch: &common::Scratch) -> (SetId, String, usize) {
    let ctx = context(scratch);
    let export = bench::export(&scratch.0.join("export"), &trials());
    let ingested = ingest_bench_trials(&ctx, &[export], bench::NO_OPTIONS).unwrap();
    let set = ingested.train_set.unwrap();
    let built = build(&ctx, &bench::sft_request(set.clone())).unwrap();
    (set, built.dataset.to_string(), built.records)
}

fn rebuild(scratch: &common::Scratch, set: SetId) -> String {
    let ctx = context(scratch);
    build(&ctx, &bench::sft_request(set))
        .unwrap()
        .dataset
        .to_string()
}

#[test]
fn a_dataset_built_from_either_layout_and_after_packing_has_the_same_id() {
    let loose = common::Scratch::new("layout-loose");
    Database::open_as(loose.state().expdb(), Layout::Loose, DbConfig::default()).unwrap();
    let from_loose = ingest_and_build(&loose);
    assert!(from_loose.2 > 0, "the build has records: {from_loose:?}");

    let packed = common::Scratch::new("layout-packed");
    let from_packed = ingest_and_build(&packed);
    assert_eq!(
        Layout::of_root(&packed.state().expdb()).unwrap(),
        Some(Layout::Packed)
    );
    assert_eq!(
        from_packed, from_loose,
        "same set, same dataset, same records"
    );

    let report = Workspace::at(&loose.state()).pack_state().unwrap();
    assert_eq!(report.from, "loose");
    assert_eq!(
        rebuild(&loose, from_loose.0.clone()),
        from_loose.1,
        "the packed state builds the dataset the loose one did"
    );
}
