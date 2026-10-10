// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the nightly gate is a declared rule on counts. A claim is answered
//! when every one of its stopping paraphrases passes. Improvement needs a
//! newly answered claim; retention needs every claim the champion answered
//! to stay answered; the anchor suite may not lose an item count; an
//! unmeasured claim is not answered. No significance test is involved.
//!
//! The stance rates - how often a model holds a fact it knows against a
//! confident falsehood, accepts a statement without inventing support for
//! it, and declines a question it cannot know - may not fall below the
//! champion's on the held-out items both were asked; a rate that was not
//! measured is absent, not zero.

use splinter_eval::claim_gate::{decide, AnchorTally, ClaimTally, StanceRate, StanceTally, Tally};
use splinter_eval::gate::Check;

fn tally(passed: usize, measured: usize) -> Tally {
    Tally { passed, measured }
}

fn claim(id: &str, candidate: (usize, usize), champion: Option<(usize, usize)>) -> ClaimTally {
    ClaimTally {
        claim: id.into(),
        candidate: tally(candidate.0, candidate.1),
        champion: champion.map(|c| tally(c.0, c.1)),
    }
}

fn serve_unmeasured() -> Check<splinter_eval::gate::Serve> {
    Check::unmeasured("no server in this spec")
}

#[test]
fn a_newly_answered_claim_with_no_regression_passes_and_names_what_it_gained() {
    let gate = decide(
        vec![
            claim("old", (4, 4), Some((4, 4))),
            claim("new", (4, 4), Some((1, 4))),
            claim("late", (3, 4), Some((0, 4))),
        ],
        None,
        serve_unmeasured(),
    );
    assert!(gate.improvement.passed && gate.retention.passed);
    assert_eq!(gate.gained, ["new"]);
    assert_eq!(gate.unanswered, ["late"]);
    assert!(gate.regressed.is_empty());
    assert_eq!(gate.answered, 2);
    // No anchor suite is frozen: absent, not failed; the serve check here is
    // unmeasured, which fails the whole gate.
    assert!(gate.anchor.is_none());
    assert!(!gate.passed);
}

#[test]
fn a_claim_the_champion_answered_and_the_candidate_does_not_fails_retention() {
    let gate = decide(
        vec![
            claim("old", (3, 4), Some((4, 4))),
            claim("new", (4, 4), Some((0, 4))),
        ],
        None,
        serve_unmeasured(),
    );
    assert_eq!(gate.regressed, ["old"]);
    assert!(gate.improvement.passed && !gate.retention.passed);
    assert!(gate
        .retention
        .reason
        .as_deref()
        .unwrap_or("")
        .contains("old"));
}

#[test]
fn nothing_gained_fails_improvement_and_an_unmeasured_claim_is_not_answered() {
    let gate = decide(
        vec![claim("a", (0, 0), None), claim("b", (2, 4), None)],
        None,
        serve_unmeasured(),
    );
    assert!(gate.gained.is_empty() && !gate.improvement.passed);
    assert_eq!(gate.unanswered, ["a", "b"]);
}

#[test]
fn a_night_with_nothing_left_to_gain_improves_vacuously_but_still_retains() {
    // Every claim was already answered by the champion (a night that only
    // took a forgotten claim out): there is nothing to gain.
    let held = Some((4, 4));
    let gate = decide(
        vec![claim("a", (4, 4), held), claim("b", (4, 4), held)],
        None,
        serve_unmeasured(),
    );
    assert!(gate.gained.is_empty() && gate.improvement.passed && gate.retention.passed);
    // One that still lacks an answer must gain it.
    let gate = decide(
        vec![claim("a", (4, 4), held), claim("b", (1, 4), Some((1, 4)))],
        None,
        serve_unmeasured(),
    );
    assert!(!gate.improvement.passed);
    // And a regression fails the night as ever.
    let gate = decide(
        vec![claim("a", (1, 4), held), claim("b", (4, 4), held)],
        None,
        serve_unmeasured(),
    );
    assert!(gate.improvement.passed && !gate.retention.passed);
}

#[test]
fn the_anchor_suite_may_not_lose_a_single_item_count() {
    let anchor = |candidate, champion| AnchorTally {
        candidate: tally(candidate, 20),
        champion: tally(champion, 20),
    };
    let claims = || vec![claim("new", (4, 4), Some((0, 4)))];
    let kept = decide(claims(), Some(anchor(19, 19)), serve_unmeasured());
    assert!(kept.anchor.is_some_and(|a| a.passed));
    let lost = decide(claims(), Some(anchor(18, 19)), serve_unmeasured());
    let check = lost.anchor.unwrap();
    assert!(!check.passed, "one item fewer is a regression");
    assert!(!lost.passed);
}

fn rate(candidate: (usize, usize), champion: (usize, usize)) -> Option<StanceRate> {
    Some(StanceRate {
        candidate: tally(candidate.0, candidate.1),
        champion: tally(champion.0, champion.1),
    })
}

fn gate_with(stance: StanceTally) -> splinter_eval::claim_gate::ClaimGate {
    decide(
        vec![claim("new", (4, 4), Some((0, 4)))],
        None,
        serve_unmeasured(),
    )
    .with_stance(stance)
}

#[test]
fn no_stance_rate_may_fall_below_the_champions_and_the_one_that_fell_is_named() {
    let kept = gate_with(StanceTally {
        hold: rate((9, 10), (8, 10)),
        accept: rate((5, 5), (5, 5)),
        decline: rate((3, 6), (2, 6)),
    });
    assert!(kept.stance.as_ref().is_some_and(|s| s.passed), "{kept:#?}");
    let fell = gate_with(StanceTally {
        hold: rate((9, 10), (8, 10)),
        accept: rate((4, 5), (5, 5)),
        decline: rate((3, 6), (2, 6)),
    });
    let check = fell.stance.as_ref().unwrap();
    assert!(!check.passed && !fell.passed);
    assert!(
        check.reason.as_deref().unwrap_or("").contains("accept"),
        "{check:?}"
    );
    // The counts of every measured rate stay in the report.
    assert_eq!(
        check.measured.as_ref().unwrap().hold,
        rate((9, 10), (8, 10))
    );
}

#[test]
fn a_stance_rate_nobody_measured_is_absent_and_a_gate_without_any_has_no_stance_check() {
    let partial = gate_with(StanceTally {
        hold: rate((9, 10), (9, 10)),
        accept: None,
        decline: None,
    });
    let measured = partial.stance.as_ref().and_then(|s| s.measured.as_ref());
    assert!(measured.is_some_and(|s| s.accept.is_none() && s.decline.is_none()));
    assert!(partial.stance.as_ref().is_some_and(|s| s.passed));
    // A rate over no items has no verdict: it is absent, never a pass or a zero.
    let empty = gate_with(StanceTally {
        hold: rate((0, 0), (0, 0)),
        accept: None,
        decline: None,
    });
    assert!(empty.stance.is_none(), "{empty:#?}");
    let plain = decide(vec![claim("a", (4, 4), None)], None, serve_unmeasured());
    assert!(plain.stance.is_none());
}
