// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out splits of
// code-editing task sets, for its clients. If your team needs expertise in
// holding out whole task families before any training data is built, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: `tasks pick` splits the union of task sets into a picked set and
//! the rest. The split is a function of the seed and the tasks alone, not of
//! the order the sets were named in; it never divides a family when picking
//! a share of families; it caps each family when asked for a number per
//! family; and the two halves are disjoint and together hold every task once.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;

use common::gate::{gate_context, Brain};
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Task};
use splinter_core::rng::Seed;
use splinter_orchestrator::Context;
use splinter_pipelines::tasks::pick::{pick, Quota};
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

const FAMILIES: usize = 8;
const PER_FAMILY: usize = 3;

/// Task `n` of family `f`: the family is its first declared concept.
fn task(f: usize, n: usize) -> Task {
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::closed_book(),
        format!("Fix the fault {n} in file {f}."),
        Vec::new(),
    )
    .unwrap()
    .with_concepts([format!("file_{f}.py")])
    .unwrap()
}

fn set_of(ctx: &Context, name: &str, tasks: &[Task]) -> TaskSetId {
    let members = tasks
        .iter()
        .map(|t| TaskEntry {
            task: ctx.tasks().put(t).unwrap(),
            generator: None,
            prompt: None,
            variant_of: None,
            subject: None,
        })
        .collect();
    ctx.tasks()
        .put_set(&TaskSet {
            name: name.into(),
            members,
        })
        .unwrap()
}

/// Two sets over eight families that overlap in family 3.
fn two_sets(ctx: &Context) -> [TaskSetId; 2] {
    let all: Vec<Task> = (0..FAMILIES)
        .flat_map(|f| (0..PER_FAMILY).map(move |n| task(f, n)))
        .collect();
    let first: Vec<Task> = all.iter().take(4 * PER_FAMILY).cloned().collect();
    let second: Vec<Task> = all.iter().skip(3 * PER_FAMILY).cloned().collect();
    [set_of(ctx, "first", &first), set_of(ctx, "second", &second)]
}

fn tasks_of(ctx: &Context, set: &TaskSetId) -> Vec<Digest> {
    ctx.tasks()
        .get_set(set)
        .unwrap()
        .members
        .into_iter()
        .map(|m| m.task)
        .collect()
}

fn families_of(ctx: &Context, tasks: &[Digest]) -> BTreeSet<String> {
    tasks
        .iter()
        .map(|t| ctx.tasks().get(t).unwrap().concepts[0].clone())
        .collect()
}

#[test]
fn a_share_of_whole_families_is_picked_stably_and_the_halves_partition_the_union() {
    let (_scratch, ctx) = gate_context("task-pick-share", Brain::Honest);
    let [a, b] = two_sets(&ctx);
    // seed: the split is a pure function of the seed, so the property under test (the same seed gives the same split in any set order) needs one fixed seed.
    let seed = Seed::new(7);

    let picked = pick(
        &ctx,
        &[a.clone(), b.clone()],
        Quota::FamilyShare(0.25),
        seed,
    )
    .unwrap();
    let reordered = pick(
        &ctx,
        &[b.clone(), a.clone()],
        Quota::FamilyShare(0.25),
        seed,
    )
    .unwrap();
    assert_eq!(
        picked.picked, reordered.picked,
        "reordering the sets changed the pick"
    );
    assert_eq!(picked.rest, reordered.rest);

    let chosen = tasks_of(&ctx, &picked.picked);
    let rest = tasks_of(&ctx, &picked.rest);
    // A quarter of eight families, every task of each of them.
    assert_eq!(families_of(&ctx, &chosen).len(), 2);
    assert_eq!(chosen.len(), 2 * PER_FAMILY);
    assert_eq!(picked.picked_tasks, chosen.len());
    assert_eq!(picked.rest_tasks, rest.len());
    // Whole families: no family on both sides.
    assert!(families_of(&ctx, &chosen).is_disjoint(&families_of(&ctx, &rest)));
    // Disjoint and covering: every task of the union once (family 3 is in
    // both sets and counted once).
    let mut both: Vec<Digest> = chosen.iter().chain(&rest).cloned().collect();
    both.sort_by_key(ToString::to_string);
    let mut union: Vec<Digest> = tasks_of(&ctx, &a)
        .into_iter()
        .chain(tasks_of(&ctx, &b))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    union.sort_by_key(ToString::to_string);
    assert_eq!(both, union);
    assert_eq!(union.len(), FAMILIES * PER_FAMILY);

    // The seed decides which families.
    let picks: BTreeSet<BTreeSet<String>> = (0..16)
        .map(|s| {
            let p = pick(
                &ctx,
                &[a.clone(), b.clone()],
                Quota::FamilyShare(0.25),
                Seed::new(s),
            )
            .unwrap();
            families_of(&ctx, &tasks_of(&ctx, &p.picked))
        })
        .collect();
    assert!(picks.len() > 1, "the seed does not matter");
}

#[test]
fn a_cap_per_family_picks_that_many_of_every_family_and_leaves_the_rest() {
    let (_scratch, ctx) = gate_context("task-pick-cap", Brain::Honest);
    let [a, b] = two_sets(&ctx);

    let sample = pick(
        &ctx,
        &[a.clone(), b.clone()],
        Quota::PerFamily(1),
        // seed: the split is a pure function of the seed, so the property under test (the same seed gives the same split in any set order) needs one fixed seed.
        Seed::new(3),
    )
    .unwrap();
    let chosen = tasks_of(&ctx, &sample.picked);
    let rest = tasks_of(&ctx, &sample.rest);
    assert_eq!(chosen.len(), FAMILIES);
    assert_eq!(families_of(&ctx, &chosen).len(), FAMILIES);
    assert_eq!(rest.len(), FAMILIES * (PER_FAMILY - 1));
    assert!(chosen.iter().all(|t| !rest.contains(t)));

    // seed: the split is a pure function of the seed, so the property under test (the same seed gives the same split in any set order) needs one fixed seed.
    let again = pick(&ctx, &[b, a], Quota::PerFamily(1), Seed::new(3)).unwrap();
    assert_eq!(sample.picked, again.picked);
}

#[test]
fn a_quota_that_means_nothing_is_refused() {
    let (_scratch, ctx) = gate_context("task-pick-refused", Brain::Honest);
    let [a, _] = two_sets(&ctx);
    for quota in [
        Quota::FamilyShare(-0.1),
        Quota::FamilyShare(1.5),
        Quota::FamilyShare(f64::NAN),
        Quota::PerFamily(0),
    ] {
        assert!(
            // seed: a refused quota is refused whatever the seed; any fixed value serves.
            pick(&ctx, std::slice::from_ref(&a), quota, Seed::new(1)).is_err(),
            "{quota:?}"
        );
    }
}
