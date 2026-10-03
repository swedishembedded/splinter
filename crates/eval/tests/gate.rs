// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that adopt a model update
// only on paired, held-out evidence, for its clients. If your team needs
// expertise in evaluation statistics for model releases, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the improvement check decides on the sign test it is handed, and on
//! nothing else, so the gate can be run and judged without a model backend.

use std::collections::BTreeMap;

use splinter_eval::gate::{improvement, SuiteSummary};
use splinter_eval::paired::PairedOutcome;
use splinter_eval::significance::{SignTest, Significance};

/// A sign test that reports whatever p-value it was built with.
struct Fixed(f64);

impl Significance for Fixed {
    fn sign_test(&self, pairs: &[(bool, bool)]) -> SignTest {
        SignTest {
            discordant: pairs.iter().filter(|(c, b)| c != b).count(),
            candidate_wins: pairs.iter().filter(|(c, b)| *c && !*b).count(),
            p_value: self.0,
        }
    }
}

fn suite() -> SuiteSummary {
    SuiteSummary {
        name: "new".into(),
        tasks: 3,
        excluded: BTreeMap::new(),
    }
}

fn outcomes() -> Vec<PairedOutcome> {
    (0..3)
        .map(|n| PairedOutcome {
            item: format!("t{n}"),
            candidate: Some(true),
            baseline: Some(false),
        })
        .collect()
}

#[test]
fn an_improvement_is_significant_exactly_when_the_handed_test_says_so() {
    let significant = improvement(suite(), None, &outcomes(), 0.05, &Fixed(0.01));
    assert!(significant.passed, "{significant:?}");

    let not_significant = improvement(suite(), None, &outcomes(), 0.05, &Fixed(0.2));
    assert!(!not_significant.passed);
    let reason = not_significant.reason.expect("it says why");
    assert!(reason.contains("no significant improvement"), "{reason}");
    assert!(reason.contains("won 3 of 3"), "{reason}");
}

#[test]
fn nothing_graded_for_both_models_is_unmeasured_not_a_pass() {
    let unpaired = vec![PairedOutcome {
        item: "t".into(),
        candidate: Some(true),
        baseline: None,
    }];
    let check = improvement(suite(), None, &unpaired, 0.05, &Fixed(0.0));
    assert!(!check.passed);
    assert!(check.measured.is_none());
}
