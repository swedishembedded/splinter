// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that adopt a model update
// only on paired, held-out evidence, for its clients. If your team needs
// expertise in evaluation statistics for model releases, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the improvement check decides on the sign test it is handed and,
//! over every assigned task, on the candidate not having solved fewer than
//! the champion, so the gate can be run and judged without a model backend.

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
            cluster: None,
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
        cluster: None,
    }];
    let check = improvement(suite(), None, &unpaired, 0.05, &Fixed(0.0));
    assert!(!check.passed);
    assert!(check.measured.is_none());
}

/// A sign test that needs five discordant units of evidence to be significant.
struct NeedsFive;

impl Significance for NeedsFive {
    fn sign_test(&self, pairs: &[(bool, bool)]) -> SignTest {
        let discordant = pairs.iter().filter(|(c, b)| c != b).count();
        SignTest {
            discordant,
            candidate_wins: pairs.iter().filter(|(c, b)| *c && !*b).count(),
            p_value: if discordant >= 5 { 0.01 } else { 0.5 },
        }
    }
}

#[test]
fn many_wins_inside_one_family_of_sources_are_one_unit_of_evidence() {
    let item = |n: usize, cluster: Option<&str>, candidate: bool, baseline: bool| PairedOutcome {
        item: format!("t{n}"),
        candidate: Some(candidate),
        baseline: Some(baseline),
        cluster: cluster.map(str::to_string),
    };
    // Eight wins, all about one letter: the same evidence eight times.
    let outcomes: Vec<PairedOutcome> = (0..8)
        .map(|n| item(n, Some("letter"), true, false))
        .collect();
    let alone = improvement(suite(), None, &outcomes, 0.05, &NeedsFive);
    assert!(!alone.passed, "one family is one unit: {alone:?}");
    assert!(alone.reason.unwrap().contains("won 1 of 1"));

    // The same eight wins over eight families are eight units.
    let spread: Vec<PairedOutcome> = (0..8)
        .map(|n| item(n, Some(&format!("letter-{n}")), true, false))
        .collect();
    assert!(improvement(suite(), None, &spread, 0.05, &NeedsFive).passed);
    // And items with no family count one by one, as before.
    let unclustered: Vec<PairedOutcome> = (0..8).map(|n| item(n, None, true, false)).collect();
    assert!(improvement(suite(), None, &unclustered, 0.05, &NeedsFive).passed);
}

#[test]
fn a_candidate_that_solved_fewer_tasks_is_refused_even_when_the_handed_test_is_significant() {
    // Four paired tasks: the candidate is right on one, the champion on three.
    // This is the shape a candidate produces when its own attempts are set
    // aside as exposed while the champion's are graded: the paired sign test
    // sees only the pairs both were graded on, so a loss can disappear from
    // its denominator. The gate must still count every assigned task.
    let outcomes: Vec<PairedOutcome> = [(true, false), (false, true), (false, true), (false, true)]
        .into_iter()
        .enumerate()
        .map(|(n, (candidate, baseline))| PairedOutcome {
            item: format!("t{n}"),
            candidate: Some(candidate),
            baseline: Some(baseline),
            cluster: None,
        })
        .collect();
    let mut four = suite();
    four.tasks = 4;
    let check = improvement(four, None, &outcomes, 0.05, &Fixed(0.01));
    assert!(!check.passed, "fewer solved tasks must refuse: {check:?}");
    let reason = check.reason.expect("it says why");
    assert!(reason.contains("fewer tasks"), "{reason}");
    assert!(reason.contains("1 against 3"), "{reason}");
}

#[test]
fn non_inferiority_passes_when_the_candidate_solved_at_least_as_many() {
    let outcomes: Vec<PairedOutcome> = [(true, false), (true, true), (false, false)]
        .into_iter()
        .enumerate()
        .map(|(n, (candidate, baseline))| PairedOutcome {
            item: format!("t{n}"),
            candidate: Some(candidate),
            baseline: Some(baseline),
            cluster: None,
        })
        .collect();
    let check = improvement(suite(), None, &outcomes, 0.05, &Fixed(0.01));
    assert!(check.passed, "{check:?}");
}
