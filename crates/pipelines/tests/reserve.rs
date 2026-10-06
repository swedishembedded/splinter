// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out examinations of what a
// model learned from a person's writing, for its clients. If your team needs
// expertise in reserving an exam from overlapping sources before any training
// data is built, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the exam's families are reserved before anything is generated, by a
//! stable hash of the family's name. The text of a reserved family, every
//! edition of it, is in no source anything after the reservation reads; the
//! exam is written from the reserved text alone; and a request that cannot be
//! honoured is refused with the reason.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;

use common::{scratch_context, Scripted};
use splinter_core::source::SourceId;
use splinter_pipelines::reserve::{reserve, Reservation, ReserveRequest};
use splinter_pipelines::sources::{add, SourceTarget};

/// `n` words no other `seed` shares.
fn words(seed: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{seed}w{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn run(name: &str, letters: usize, families: usize) -> (common::Scratch, Vec<String>, Reservation) {
    let (scratch, ctx) = scratch_context(name, Scripted::new(|_| String::new()), false);
    let dir = scratch.0.join("letters");
    std::fs::create_dir_all(&dir).unwrap();
    for n in 0..letters {
        let body = words(&format!("l{n}"), 200);
        std::fs::write(dir.join(format!("letter-{n:02}.txt")), &body).unwrap();
        if n < 6 {
            std::fs::write(
                dir.join(format!("reprint-{n:02}.txt")),
                format!("Dear sir, {body} yours"),
            )
            .unwrap();
        }
    }
    let id = add(&ctx, &SourceTarget::Path { path: dir })
        .unwrap()
        .source
        .id;
    let reservation = reserve(
        &ctx,
        &ReserveRequest {
            sources: &[id],
            families,
            seed: 1,
            touched_by: &[],
        },
    )
    .unwrap();
    let names = |ids: &[SourceId]| -> Vec<String> {
        ids.iter()
            .flat_map(|id| ctx.sources().get_source(id).unwrap().parts)
            .map(|p| p.name)
            .collect()
    };
    let mut both = names(&reservation.training);
    both.push("|".into());
    both.extend(names(&reservation.exam));
    (scratch, both, reservation)
}

#[test]
fn a_reserved_family_and_every_edition_of_it_leaves_the_training_source_and_makes_the_exam() {
    let (_scratch, names, reservation) = run("reserve-split", 40, 8);
    assert_eq!(reservation.families.len(), 8);
    let at = names.iter().position(|n| n == "|").unwrap();
    let (training, exam) = (&names[..at], &names[at + 1..]);
    // Parts are partitioned: nothing is in both, nothing is lost.
    let training_set: BTreeSet<_> = training.iter().collect();
    assert!(exam.iter().all(|n| !training_set.contains(n)));
    assert_eq!(training.len() + exam.len(), 46);
    // An edition of a reserved letter goes with it.
    for n in 0..6 {
        let letter = format!("letter-{n:02}.txt");
        let reprint = format!("reprint-{n:02}.txt");
        assert_eq!(
            exam.contains(&letter),
            exam.contains(&reprint),
            "{letter} and its reprint are one family"
        );
        assert_eq!(training.contains(&letter), training.contains(&reprint));
    }
    assert_eq!(reservation.total_families, 40);
}

#[test]
fn the_same_sources_and_seed_reserve_the_same_families() {
    let (_a, _, first) = run("reserve-a", 40, 8);
    let (_b, _, second) = run("reserve-b", 40, 8);
    assert_eq!(first.families, second.families);
}

#[test]
fn too_few_examinable_families_or_too_little_left_to_learn_from_is_refused() {
    let (scratch, ctx) = scratch_context("reserve-refuse", Scripted::new(|_| String::new()), false);
    let dir = scratch.0.join("letters");
    std::fs::create_dir_all(&dir).unwrap();
    for n in 0..10 {
        std::fs::write(dir.join(format!("l{n}.txt")), words(&format!("l{n}"), 200)).unwrap();
    }
    // One too short to write a task from.
    std::fs::write(dir.join("short.txt"), "a few words").unwrap();
    let id = add(&ctx, &SourceTarget::Path { path: dir })
        .unwrap()
        .source
        .id;
    let ask = |families: usize| {
        reserve(
            &ctx,
            &ReserveRequest {
                sources: std::slice::from_ref(&id),
                families,
                seed: 1,
                touched_by: &[],
            },
        )
        .unwrap_err()
        .to_string()
    };
    let too_many = ask(6);
    assert!(too_many.contains("at most half"), "{too_many}");
    // Nothing long enough to write a task from: the shortfall is named.
    let (scratch, ctx) = scratch_context("reserve-short", Scripted::new(|_| String::new()), false);
    let dir = scratch.0.join("letters");
    std::fs::create_dir_all(&dir).unwrap();
    for n in 0..10 {
        std::fs::write(dir.join(format!("s{n}.txt")), words(&format!("s{n}"), 30)).unwrap();
    }
    let id = add(&ctx, &SourceTarget::Path { path: dir })
        .unwrap()
        .source
        .id;
    let short = reserve(
        &ctx,
        &ReserveRequest {
            sources: &[id],
            families: 2,
            seed: 1,
            touched_by: &[],
        },
    )
    .unwrap_err()
    .to_string();
    assert!(
        short.contains("--exam-families") && short.contains("hold 0"),
        "{short}"
    );
}
