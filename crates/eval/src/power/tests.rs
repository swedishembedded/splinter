// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specification of the sign-test power functions against hand-computed
//! values. The exact test is local to the tests: the arithmetic the gate
//! uses is handed in, and these pin what any correct one must give.

use super::*;
use crate::significance::{SignTest, Significance};

/// P(Binomial(n, 1/2) >= k), exactly.
struct Exact;

impl Significance for Exact {
    fn sign_test(&self, pairs: &[(bool, bool)]) -> SignTest {
        let wins = pairs.iter().filter(|(c, b)| *c && !*b).count();
        let n = wins + pairs.iter().filter(|(c, b)| !*c && *b).count();
        let mut tail = 0.0;
        let mut choose = 1.0f64;
        for k in 0..=n {
            if k > 0 {
                choose = choose * (n - k + 1) as f64 / k as f64;
            }
            if k >= wins {
                tail += choose;
            }
        }
        SignTest {
            discordant: n,
            candidate_wins: wins,
            p_value: if n == 0 {
                1.0
            } else {
                tail / 2f64.powi(n as i32)
            },
        }
    }
}

#[test]
fn few_independent_units_need_unanimity_and_four_cannot_reach_the_level() {
    // n=5 unanimous: p = 1/32 = 0.03125; four of five: 6/32 = 0.1875.
    assert_eq!(critical_wins(5, 0.05, &Exact), Some(5));
    // n=6 unanimous: p = 1/64; five of six: 7/64 = 0.109.
    assert_eq!(critical_wins(6, 0.05, &Exact), Some(6));
    // n=8 tolerates one negative: 7 of 8 is 9/256 = 0.035; 6 of 8 is 37/256.
    assert_eq!(critical_wins(8, 0.05, &Exact), Some(7));
    // n=4 unanimous is 1/16 = 0.0625 > 0.05: no outcome is significant.
    assert_eq!(critical_wins(4, 0.05, &Exact), None);
}

#[test]
fn power_is_the_exact_chance_the_test_rejects() {
    // Every unit discordant and the candidate wins with probability 0.9:
    // five units need all five: 0.9^5.
    let plan = SignPlan {
        units: 5,
        discordance: 1.0,
        effect: 0.8,
        alpha: 0.05,
    };
    assert!((power(&plan, &Exact) - 0.9f64.powi(5)).abs() < 1e-12);
    // Eight units, win probability 0.9: at least 7 of 8.
    let plan = SignPlan {
        units: 8,
        discordance: 1.0,
        effect: 0.8,
        alpha: 0.05,
    };
    let expected = 0.9f64.powi(8) + 8.0 * 0.9f64.powi(7) * 0.1;
    assert!((power(&plan, &Exact) - expected).abs() < 1e-12);
    // Four units can never reject.
    let plan = SignPlan {
        units: 4,
        discordance: 1.0,
        effect: 1.0,
        alpha: 0.05,
    };
    assert_eq!(power(&plan, &Exact), 0.0);
}

#[test]
fn no_effect_has_the_size_of_the_test_not_its_power() {
    // At zero effect the chance of rejecting is the test's size, at most alpha.
    let plan = SignPlan {
        units: 8,
        discordance: 1.0,
        effect: 0.0,
        alpha: 0.05,
    };
    let size = power(&plan, &Exact);
    assert!((size - 9.0 / 256.0).abs() < 1e-12, "{size}");
}

#[test]
fn the_minimum_detectable_effect_is_the_smallest_lead_with_the_wanted_power() {
    let at = |units, discordance| minimum_detectable_effect(units, discordance, 0.05, 0.8, &Exact);
    // Five units, all discordant: needs win probability p with p^5 >= 0.8,
    // p = 0.8^(1/5) = 0.9564, an effect of 2p - 1 = 0.9129.
    let mde = at(5, 1.0).expect("reachable");
    assert!((mde - (2.0 * 0.8f64.powf(0.2) - 1.0)).abs() < 1e-4, "{mde}");
    // Four units cannot reach the level at all.
    assert_eq!(at(4, 1.0), None);
    // Ten units of which half differ cannot show any lead with four chances
    // in five: too few of them are discordant to reach the level.
    assert_eq!(at(10, 0.5), None);
    // More units see a smaller effect.
    assert!(at(100, 0.5).unwrap() < at(40, 0.5).unwrap());
    // A lead cannot exceed the discordance: units both arms get right or
    // wrong carry none.
    assert!(at(200, 0.25).unwrap() <= 0.25);
}

#[test]
fn spent_alpha_never_adds_up_to_more_than_the_level() {
    let alpha = 0.05;
    let total: f64 = (1..=200).map(|g| spent_alpha(alpha, g)).sum();
    assert!(total < alpha, "{total}");
    assert!((spent_alpha(alpha, 1) - 0.025).abs() < 1e-15);
    assert!((spent_alpha(alpha, 2) - alpha / 6.0).abs() < 1e-15);
    // Ten tests at the plain level release a null candidate with probability
    // up to 1 - 0.95^10 = 0.40; the spent levels bound it by their sum.
    let plain = 1.0 - (1.0 - alpha).powi(10);
    let spent: f64 = (1..=10).map(|g| spent_alpha(alpha, g)).sum();
    assert!(plain > 0.40 && spent < 0.05, "{plain} {spent}");
}

fn test_of(wins: usize, losses: usize) -> SignTest {
    let pairs: Vec<(bool, bool)> = [(true, false)]
        .repeat(wins)
        .into_iter()
        .chain([(false, true)].repeat(losses))
        .collect();
    Exact.sign_test(&pairs)
}

#[test]
fn a_miss_is_inconclusive_when_the_exam_could_not_have_seen_the_effect() {
    let verdict =
        |units, wins, losses| Verdict::of(&test_of(wins, losses), units, 0.05, Some(0.3), &Exact);
    // Rejected: significant, whatever the power.
    assert_eq!(verdict(8, 8, 0), Verdict::Significant);
    // Six units, three discordant, a lead of 0.3 asked for: six units cannot
    // show it with four chances in five, so a miss says nothing.
    assert!(matches!(verdict(6, 2, 1), Verdict::Inconclusive { .. }));
    // Two hundred units: a miss at the same lead is a real failure.
    assert_eq!(verdict(200, 50, 50), Verdict::NotSignificant);
    // No target lead: a miss is a failure, as before.
    assert_eq!(
        Verdict::of(&test_of(2, 1), 6, 0.05, None, &Exact),
        Verdict::NotSignificant
    );
}

fn outcomes(wins: usize, losses: usize, ties: usize) -> Vec<PairedOutcome> {
    let one = |n: usize, c: bool, b: bool| PairedOutcome {
        item: format!("t{n}"),
        candidate: Some(c),
        baseline: Some(b),
        cluster: Some(format!("family-{n}")),
    };
    (0..wins)
        .map(|n| one(n, true, false))
        .chain((0..losses).map(|n| one(100 + n, false, true)))
        .chain((0..ties).map(|n| one(200 + n, true, true)))
        .collect()
}

fn suite() -> SuiteSummary {
    SuiteSummary {
        name: "held-out".into(),
        tasks: 0,
        excluded: Default::default(),
    }
}

fn judged(outcomes: &[PairedOutcome], rule: Rule) -> Check<Improvement> {
    judged_improvement(suite(), None, outcomes, rule, &Exact)
}

const PLAIN: Rule = Rule {
    alpha: 0.05,
    generation: None,
    target_effect: None,
};

#[test]
fn five_unanimous_families_pass_at_the_plain_level_and_not_at_a_later_generation() {
    // p = 1/32 = 0.031: below 0.05, above the 0.025 the first generation
    // of the sequence may spend.
    let five = outcomes(5, 0, 0);
    assert!(judged(&five, PLAIN).passed);
    let first = judged(
        &five,
        Rule {
            generation: Some(1),
            ..PLAIN
        },
    );
    assert!(!first.passed && !first.inconclusive, "{first:?}");
    let recorded = first.measured.unwrap();
    assert!(
        (recorded.alpha - 0.025).abs() < 1e-15,
        "the level held to is recorded"
    );
    // Six unanimous families (1/64 = 0.0156) pass the first generation and
    // fail the third (0.05 / 12 = 0.0042).
    let six = outcomes(6, 0, 0);
    assert!(
        judged(
            &six,
            Rule {
                generation: Some(1),
                ..PLAIN
            }
        )
        .passed
    );
    assert!(
        !judged(
            &six,
            Rule {
                generation: Some(3),
                ..PLAIN
            }
        )
        .passed
    );
}

#[test]
fn a_miss_on_a_small_exam_is_inconclusive_and_releases_nothing() {
    let rule = Rule {
        target_effect: Some(0.3),
        ..PLAIN
    };
    let small = judged(&outcomes(2, 1, 3), rule);
    assert!(small.inconclusive && !small.passed, "{small:?}");
    assert!(
        small.reason.as_deref().unwrap().starts_with("inconclusive"),
        "{small:?}"
    );
    // The same shares on a large exam are a failure, not an inconclusive.
    let large = judged(&outcomes(40, 40, 120), rule);
    assert!(!large.inconclusive && !large.passed, "{large:?}");
    // A rejection stays a pass.
    assert!(judged(&outcomes(8, 0, 0), rule).passed);
}
