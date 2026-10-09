// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the nightly gate is a declared rule on counts. A claim is answered
//! when every one of its stopping paraphrases passes. Improvement needs a
//! newly answered claim; retention needs every claim the champion answered
//! to stay answered; the anchor suite may not lose an item count; an
//! unmeasured claim is not answered. No significance test is involved.

use splinter_eval::claim_gate::{decide, AnchorTally, ClaimTally, Tally};
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
