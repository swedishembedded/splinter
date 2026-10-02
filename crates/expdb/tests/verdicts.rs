// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a pass/fail verdict carries the rank of its evidence. What a target
//! resolves to is the strongest evidence that decides: abstentions say
//! nothing, equal-rank disagreement decides nothing, and an unjudged target
//! has no resolution rather than a zero one. Relations between experiences
//! are first-class edges.
#![allow(clippy::unwrap_used)]

mod common;

use common::Scratch;
use splinter_expdb::analyze::{resolve_verdicts, Resolution};
use splinter_expdb::model::{Entity, Evaluation, EvaluatorRef, Rel, Ruling, Target, Verdict};
use splinter_expdb::{ContentId, WriterIdentity};

fn verdict(target: Target, who: &str, ruling: Ruling, rank: u8) -> Evaluation {
    Evaluation::new(target, EvaluatorRef::new(who, "1"), "verdict", 0.0, 1.0)
        .with_verdict(Verdict::new(ruling, rank))
}

fn v(ruling: Ruling, rank: u8) -> Verdict {
    Verdict::new(ruling, rank)
}

#[test]
fn the_strongest_deciding_evidence_wins() {
    assert_eq!(resolve_verdicts(&[]), None, "no judgement is not a zero");
    assert_eq!(
        resolve_verdicts(&[v(Ruling::Abstain, 3)]),
        None,
        "an abstention alone decides nothing"
    );
    assert_eq!(
        resolve_verdicts(&[v(Ruling::Pass, 1)]),
        Some(Resolution {
            passed: true,
            rank: 1
        })
    );
    assert_eq!(
        resolve_verdicts(&[
            v(Ruling::Pass, 1),
            v(Ruling::Fail, 2),
            v(Ruling::Abstain, 3)
        ]),
        Some(Resolution {
            passed: false,
            rank: 2
        }),
        "a stronger fail beats a weaker pass, and an abstention does not count"
    );
    assert_eq!(
        resolve_verdicts(&[v(Ruling::Pass, 2), v(Ruling::Fail, 2)]),
        None,
        "equal-rank disagreement decides nothing"
    );
    assert_eq!(
        resolve_verdicts(&[v(Ruling::Pass, 3), v(Ruling::Fail, 2), v(Ruling::Pass, 3)]),
        Some(Resolution {
            passed: true,
            rank: 3
        })
    );
}

#[test]
fn a_target_resolves_from_the_verdicts_that_still_stand() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let thing = Target::Entity(
        c.put_entity(&Entity::new("experience", serde_json::json!(1)))
            .unwrap(),
    );
    let other = Target::Entity(ContentId::of(b"unjudged"));
    c.evaluate(verdict(thing, "judge", Ruling::Pass, 1))
        .unwrap();
    c.evaluate(verdict(thing, "tests", Ruling::Fail, 4))
        .unwrap();
    c.flush().unwrap();

    let snapshot = db.snapshot().unwrap();
    assert_eq!(
        snapshot.resolution(&thing, "verdict").unwrap(),
        Some(Resolution {
            passed: false,
            rank: 4
        })
    );
    assert_eq!(snapshot.resolution(&other, "verdict").unwrap(), None);
    assert_eq!(
        snapshot.resolution(&thing, "other_criterion").unwrap(),
        None
    );

    c.retract(EvaluatorRef::new("tests", "1"), "flaky").unwrap();
    c.flush().unwrap();
    assert_eq!(
        db.snapshot()
            .unwrap()
            .resolution(&thing, "verdict")
            .unwrap(),
        Some(Resolution {
            passed: true,
            rank: 1
        }),
        "withdrawing the stronger evaluator exposes the weaker verdict"
    );
}

#[test]
fn every_resolution_of_a_criterion_is_read_in_one_pass() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let a = Target::Entity(
        c.put_entity(&Entity::new("experience", serde_json::json!("a")))
            .unwrap(),
    );
    let b = Target::Entity(
        c.put_entity(&Entity::new("experience", serde_json::json!("b")))
            .unwrap(),
    );
    c.evaluate(verdict(a, "tests", Ruling::Pass, 4)).unwrap();
    c.evaluate(verdict(b, "judge", Ruling::Abstain, 1)).unwrap();
    c.flush().unwrap();

    let all = db.snapshot().unwrap().resolutions("verdict").unwrap();
    assert_eq!(
        all.get(&a),
        Some(&Resolution {
            passed: true,
            rank: 4
        })
    );
    assert!(
        !all.contains_key(&b),
        "an abstention leaves the target unresolved"
    );
}

#[test]
fn a_verdict_survives_the_round_trip_and_an_evaluation_without_one_is_unchanged() {
    let plain = Evaluation::new(
        Target::Entity(ContentId::of(b"x")),
        EvaluatorRef::new("e", "1"),
        "c",
        0.5,
        0.5,
    );
    let json = serde_json::to_string(&plain).unwrap();
    assert!(
        !json.contains("verdict"),
        "no verdict, no field: old records read back identically"
    );
    let back: Evaluation = serde_json::from_str(&json).unwrap();
    assert_eq!(back, plain);

    let ruled = plain.with_verdict(v(Ruling::Fail, 2));
    let back: Evaluation = serde_json::from_str(&serde_json::to_string(&ruled).unwrap()).unwrap();
    assert_eq!(back.verdict, Some(v(Ruling::Fail, 2)));
}

#[test]
fn relations_between_experiences_are_edges() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let ids: Vec<_> = (0..2)
        .map(|n| {
            c.record(splinter_expdb::model::Body::Entity(Entity::new(
                "experience",
                serde_json::json!(n),
            )))
            .unwrap()
        })
        .collect();
    for rel in [
        Rel::PreferredOver,
        Rel::RetryOf,
        Rel::CritiqueOf,
        Rel::RevisionOf,
        Rel::VariantOf,
    ] {
        c.link(ids[1], rel, ids[0]).unwrap();
    }
    c.flush().unwrap();
    let index = db.snapshot().unwrap().index().unwrap();
    for rel in [
        Rel::PreferredOver,
        Rel::RetryOf,
        Rel::CritiqueOf,
        Rel::RevisionOf,
        Rel::VariantOf,
    ] {
        assert_eq!(index.edges_from(ids[1], Some(rel)), vec![ids[0]], "{rel:?}");
        assert_eq!(index.edges_to(ids[0], Some(rel)), vec![ids[1]], "{rel:?}");
    }
}
