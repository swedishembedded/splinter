// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: what to spend the next unit of compute on is a score over measured
//! properties, and a property nobody measured is absent, never zero.
#![allow(clippy::unwrap_used)]

use splinter_expdb::analyze::{rank, Priority};

fn full() -> Priority {
    Priority {
        uncertainty: Some(0.5),
        novelty: Some(0.8),
        learning_progress: Some(0.4),
        verification: Some(0.9),
        transfer: Some(0.7),
        cost: Some(2.0),
    }
}

#[test]
fn priority_is_uncertainty_novelty_progress_verification_and_transfer_per_unit_cost() {
    let expected = 0.5 * 0.8 * 0.4 * 0.9 * 0.7 / 2.0;
    assert!((full().score().unwrap() - expected).abs() < 1e-12);
}

#[test]
fn a_priority_with_an_unmeasured_property_has_no_score() {
    for missing in 0..6 {
        let mut p = full();
        match missing {
            0 => p.uncertainty = None,
            1 => p.novelty = None,
            2 => p.learning_progress = None,
            3 => p.verification = None,
            4 => p.transfer = None,
            _ => p.cost = None,
        }
        assert_eq!(p.score(), None, "property {missing} unmeasured");
    }
}

#[test]
fn a_cost_that_is_not_positive_gives_no_score_instead_of_infinity() {
    assert_eq!(
        Priority {
            cost: Some(0.0),
            ..full()
        }
        .score(),
        None
    );
    assert_eq!(
        Priority {
            cost: Some(-1.0),
            ..full()
        }
        .score(),
        None
    );
}

#[test]
fn ranking_puts_the_most_valuable_first_and_the_unscored_last() {
    let weak = Priority {
        uncertainty: Some(0.1),
        ..full()
    };
    let ranked = rank(vec![
        ("unscored", Priority::default()),
        ("weak", weak),
        ("strong", full()),
    ]);
    let order: Vec<_> = ranked.iter().map(|(name, _)| *name).collect();
    assert_eq!(order, ["strong", "weak", "unscored"]);
    assert_eq!(ranked[2].1, None);
}
