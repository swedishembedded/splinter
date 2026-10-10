// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the gate of a consolidation. A consolidated checkpoint takes a
//! generation into the weights and is meant to gain nothing: it passes when
//! no claim the release in use answered is lost - the closing generation's
//! claims and the ones the base already held alike - the anchor suite loses
//! no item and plain brain serves it. A claim neither answers is carried and
//! named, not a reason to refuse.

use splinter_eval::claim_gate::{decide_consolidation, AnchorTally, ClaimTally, Tally};
use splinter_eval::gate::{Check, Serve};

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

fn served() -> Check<Serve> {
    Check {
        passed: true,
        measured: Some(Serve {
            binary: "brain".into(),
            startup_line: String::new(),
            served_digest: "sha256:a".into(),
            expected_digest: "sha256:a".into(),
            sampled: 4,
            agreed: 4,
            disagreed: Vec::new(),
        }),
        reason: None,
    }
}

#[test]
fn a_checkpoint_that_loses_nothing_passes_without_gaining() {
    let gate = decide_consolidation(
        vec![
            claim("closing", (4, 4), Some((4, 4))),
            claim("held", (4, 4), Some((4, 4))),
            claim("never learned", (1, 4), Some((1, 4))),
        ],
        None,
        served(),
    );
    assert!(gate.passed, "{gate:#?}");
    assert!(gate.improvement.passed && gate.retention.passed);
    assert!(gate.gained.is_empty());
    assert_eq!(gate.unanswered, ["never learned"], "carried, and named");
}

#[test]
fn a_claim_the_release_in_use_answered_and_the_checkpoint_does_not_refuses_it() {
    let gate = decide_consolidation(
        vec![
            claim("closing", (3, 4), Some((4, 4))),
            claim("held", (4, 4), Some((4, 4))),
        ],
        None,
        served(),
    );
    assert!(!gate.passed);
    assert_eq!(gate.regressed, ["closing"]);
    assert!(!gate.retention.passed && gate.improvement.passed);
}

#[test]
fn the_anchor_and_the_serve_check_still_decide() {
    let lost = AnchorTally {
        candidate: tally(7, 10),
        champion: tally(8, 10),
    };
    let gate = decide_consolidation(vec![claim("a", (4, 4), Some((4, 4)))], Some(lost), served());
    assert!(!gate.passed && gate.anchor.as_ref().is_some_and(|a| !a.passed));

    let gate = decide_consolidation(
        vec![claim("a", (4, 4), Some((4, 4)))],
        None,
        Check::unmeasured("no server in this spec"),
    );
    assert!(!gate.passed, "an unmeasured serve check fails the gate");
}

#[test]
fn a_claim_newly_answered_is_reported_and_welcome() {
    let gate = decide_consolidation(vec![claim("late", (4, 4), Some((0, 4)))], None, served());
    assert!(gate.passed);
    assert_eq!(gate.gained, ["late"]);
}
