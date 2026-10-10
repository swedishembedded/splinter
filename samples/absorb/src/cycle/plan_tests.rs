// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specifications of [`super::plan`].

use super::fixtures::{manifest, Shape};
use super::plan::*;
use crate::facts::Manifest;
use crate::session::Style;

fn world() -> Manifest {
    manifest(&Shape {
        per_day: 2,
        days: 4,
        dev: 3,
        controls: 1,
    })
}

fn test_fact(m: &Manifest, n: usize) -> String {
    m.facts
        .iter()
        .filter(|f| f.role == Some(crate::roles::Role::Test))
        .nth(n)
        .expect("a test fact")
        .id
        .clone()
}

fn request(m: &Manifest) -> PlanRequest<'_> {
    PlanRequest {
        manifest: m,
        role: CycleRole::Test,
        nights: 6,
        per_night: 2,
        noise_per_night: 1,
        styles: vec![Style::Plain, Style::Realistic],
        null_night: Some(3),
        sham_night: Some(5),
        canary: Some(CanaryNights {
            fact: test_fact(m, 0),
            night: 1,
            back: 4,
        }),
    }
}

#[test]
fn each_night_teaches_its_days_facts_and_the_null_sham_and_canary_nights_are_where_configured() {
    let m = world();
    let plan = generate(&request(&m)).expect("a plan");
    let kinds: Vec<NightKind> = plan.nights.iter().map(|n| n.kind).collect();
    assert_eq!(
        kinds,
        [
            NightKind::Teach,
            NightKind::Teach,
            NightKind::Null,
            NightKind::Teach,
            NightKind::Sham,
            NightKind::Teach
        ]
    );
    let of = |night: usize, kind: SessionKind| -> Vec<&SessionPlan> {
        plan.nights[night - 1]
            .sessions
            .iter()
            .filter(|s| s.kind == kind)
            .collect()
    };
    assert_eq!(
        of(3, SessionKind::Noise).len(),
        plan.nights[2].sessions.len()
    );
    assert_eq!(
        of(1, SessionKind::Canary).len(),
        1,
        "the canary is asserted"
    );
    assert_eq!(
        of(1, SessionKind::Fact).len(),
        1,
        "beside the day's other fact"
    );
    assert_eq!(
        of(4, SessionKind::Fact).len(),
        3,
        "day 3 and the canary corrected back"
    );
    assert_eq!(
        of(5, SessionKind::Sham).len(),
        2,
        "the next day's facts, never corrected"
    );
    assert_eq!(of(5, SessionKind::Fact).len(), 0);

    let schedule = plan.schedule();
    let canary = &schedule[&test_fact(&m, 0)];
    assert_eq!((canary.canary, canary.taught), (Some(1), Some(4)));
    let day4 = &schedule[&test_fact(&m, 6)];
    assert_eq!((day4.shammed.clone(), day4.taught), (vec![5], Some(6)));

    let topics: Vec<usize> = plan
        .nights
        .iter()
        .flat_map(|n| n.sessions.iter().filter_map(|s| s.topic))
        .collect();
    let distinct: std::collections::BTreeSet<_> = topics.iter().collect();
    assert_eq!(
        distinct.len(),
        topics.len(),
        "no night repeats a noise topic"
    );
}

#[test]
fn a_plan_that_teaches_early_twice_or_the_wrong_facts_is_refused_naming_the_night() {
    let m = world();
    let plan = generate(&request(&m)).expect("a plan");
    let refused = |edit: &dyn Fn(&mut Plan)| {
        let mut broken = plan.clone();
        edit(&mut broken);
        format!("{:#}", broken.check(&m).expect_err("refused"))
    };
    let early = test_fact(&m, 6);
    let why = refused(&|p| {
        p.nights[0].sessions.push(SessionPlan {
            kind: SessionKind::Fact,
            fact: Some(early.clone()),
            style: Style::Plain,
            topic: None,
        });
    });
    assert!(
        why.contains("night 1") && why.contains("not earlier"),
        "{why}"
    );
    let twice = test_fact(&m, 2);
    let why = refused(&|p| {
        p.nights[4].sessions[0] = SessionPlan {
            kind: SessionKind::Fact,
            fact: Some(twice.clone()),
            style: Style::Plain,
            topic: None,
        };
    });
    assert!(why.contains("corrected twice"), "{why}");
    let dev = m
        .facts
        .iter()
        .find(|f| f.role == Some(crate::roles::Role::Dev))
        .expect("dev")
        .id
        .clone();
    let why = refused(&|p| {
        p.nights[1].sessions[0].fact = Some(dev.clone());
    });
    assert!(why.contains("night 2") && why.contains("dev"), "{why}");
    let plain = test_fact(&m, 3);
    let why = refused(&|p| {
        p.nights[1].sessions[0] = SessionPlan {
            kind: SessionKind::Canary,
            fact: Some(plain.clone()),
            style: Style::Plain,
            topic: None,
        };
    });
    assert!(why.contains("no canary"), "{why}");
    let why = refused(&|p| p.nights[2].night = 9);
    assert!(why.contains("numbered from 1"), "{why}");
}

#[test]
fn a_development_cycle_takes_its_facts_a_few_a_night_and_cannot_outrun_them() {
    let m = world();
    let dev = |nights| PlanRequest {
        role: CycleRole::Dev,
        nights,
        null_night: None,
        sham_night: None,
        canary: None,
        ..request(&m)
    };
    let plan = generate(&dev(2)).expect("a plan");
    let taught: Vec<usize> = plan
        .nights
        .iter()
        .map(|n| {
            n.sessions
                .iter()
                .filter(|s| s.kind == SessionKind::Fact)
                .count()
        })
        .collect();
    assert_eq!(taught, [2, 1], "three development facts, two a night");
    let why = format!("{:#}", generate(&dev(3)).expect_err("too many nights"));
    assert!(why.contains("only 2 teaching nights"), "{why}");
    assert_eq!(plan.first(1).expect("a night").nights.len(), 1);
    assert!(plan.first(3).is_err());
}
