// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a trainer streams a plan in a seeded random order without reading
//! every record at random: samples are fetched a window at a time, nearby
//! blocks together, then handed out in the logical order.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::HashSet;

use common::{attempt, Scratch};
use splinter_expdb::model::Outcome;
use splinter_expdb::train::{LoaderConfig, Materialized, Recipe};
use splinter_expdb::{Config, WriterIdentity};

fn plan_over_many_blocks(
    scratch: &Scratch,
) -> (
    splinter_expdb::manifest::Snapshot,
    splinter_expdb::train::TrainingPlan,
) {
    let db = scratch.open_with(Config {
        block_records: 16,
        ..Config::default()
    });
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    for task in 0..20 {
        attempt(&mut c, task, "p", 4, Outcome::Pass);
    }
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();
    let plan = snapshot.compile(&Recipe::sft()).unwrap();
    (snapshot, plan)
}

fn contexts(items: impl Iterator<Item = splinter_expdb::Result<Materialized>>) -> Vec<String> {
    items
        .map(|m| match m.unwrap() {
            Materialized::Sft {
                context, target, ..
            } => format!("{context}=>{target}"),
            other => panic!("unexpected {other:?}"),
        })
        .collect()
}

#[test]
fn an_epoch_yields_every_sample_exactly_once() {
    let scratch = Scratch::new();
    let (snapshot, plan) = plan_over_many_blocks(&scratch);
    let seen = contexts(plan.loader(
        &snapshot,
        LoaderConfig {
            window: 16,
            seed: 3,
            epoch: 0,
        },
    ));
    assert_eq!(seen.len(), plan.samples.len());
    assert_eq!(
        seen.iter().collect::<HashSet<_>>().len(),
        seen.len(),
        "no sample is repeated"
    );
}

#[test]
fn the_order_is_fixed_by_the_seed_and_changes_with_the_seed_and_the_epoch() {
    let scratch = Scratch::new();
    let (snapshot, plan) = plan_over_many_blocks(&scratch);
    let order = |seed, epoch| {
        contexts(plan.loader(
            &snapshot,
            LoaderConfig {
                window: 16,
                seed,
                epoch,
            },
        ))
    };
    assert_eq!(order(1, 0), order(1, 0));
    assert_ne!(order(1, 0), order(2, 0));
    assert_ne!(order(1, 0), order(1, 1));
    let mut a = order(1, 0);
    let mut b = order(2, 0);
    a.sort();
    b.sort();
    assert_eq!(a, b, "only the order changes");
}

#[test]
fn a_window_that_covers_the_plan_reads_each_block_at_most_once() {
    let scratch = Scratch::new();
    let (snapshot, plan) = plan_over_many_blocks(&scratch);
    let total = snapshot
        .query(&splinter_expdb::query::Query::all())
        .unwrap()
        .blocks_total;
    let before = snapshot.stats().blocks_read;
    let items = contexts(plan.loader(
        &snapshot,
        LoaderConfig {
            window: plan.samples.len(),
            seed: 9,
            epoch: 0,
        },
    ));
    assert_eq!(items.len(), plan.samples.len());
    assert!(
        snapshot.stats().blocks_read - before <= total,
        "shuffled samples still read each block once"
    );
}

#[test]
fn a_zero_window_is_refused_rather_than_looping_forever() {
    let scratch = Scratch::new();
    let (snapshot, plan) = plan_over_many_blocks(&scratch);
    let mut loader = plan.loader(
        &snapshot,
        LoaderConfig {
            window: 0,
            seed: 1,
            epoch: 0,
        },
    );
    assert!(loader.next().unwrap().is_err());
}
