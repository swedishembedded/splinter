// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements evaluation statistics whose numbers are
// checked against exact references, for its clients. If your team needs
// expertise in paired significance tests and bootstrap intervals for model
// releases, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the release statistics of `splinter_model::stats` agree with exact
//! references computed independently of them (`statistics.json`, written by
//! `generate_statistics.py` beside it from the definitions in exact rational
//! arithmetic).
//!
//! * The sign test is the exact one-sided test `P(Binomial(n, 1/2) >= wins)`
//!   over the `n` discordant pairs; ties are dropped.
//! * The percentile bootstrap interval of a paired mean resamples items with
//!   both verdicts together. Its bootstrap distribution is known exactly
//!   (multinomial counts), so each bound must sit where that distribution's
//!   CDF crosses its quantile, within the Monte Carlo error of the resamples.
//!
//! Each check is also run against deliberate defects written here (a two-sided
//! test, ties counted, a strict tail, arms resampled independently, a
//! one-sided bound): every one must be caught, or the check proves nothing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde::Deserialize;
use splinter_model::stats::{bootstrap_interval, one_sided_p, sign_test};

#[derive(Deserialize)]
struct Fixture {
    sign_test: Vec<SignCase>,
    paired_bootstrap: Vec<PairedCase>,
}

#[derive(Deserialize)]
struct SignCase {
    wins: usize,
    losses: usize,
    ties: usize,
    one_sided_p: f64,
}

#[derive(Deserialize)]
struct PairedCase {
    name: String,
    both_right: usize,
    candidate_only: usize,
    baseline_only: usize,
    both_wrong: usize,
    level: f64,
    mean: f64,
    step: f64,
    paired_cdf: Vec<(f64, f64)>,
    unpaired_detectable: bool,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/conformance/statistics.json")).unwrap()
}

/// `(candidate right, baseline right)` per item, the ties alternating
/// between both right and both wrong.
fn sign_pairs(case: &SignCase) -> Vec<(bool, bool)> {
    let mut pairs = vec![(true, false); case.wins];
    pairs.extend(vec![(false, true); case.losses]);
    pairs.extend((0..case.ties).map(|i| (i % 2 == 0, i % 2 == 0)));
    pairs
}

/// A sign test over `(candidate right, baseline right)` pairs: its p-value.
type PValueOf<'a> = &'a dyn Fn(&[(bool, bool)]) -> f64;

/// The first fixture case on which `p_of` disagrees with the exact one-sided
/// p-value, by relative error.
fn check_sign_test(p_of: PValueOf<'_>) -> Result<(), String> {
    for case in fixture().sign_test {
        let got = p_of(&sign_pairs(&case));
        let tolerance = 1e-9 * case.one_sided_p.max(1e-300);
        if (got - case.one_sided_p).abs() > tolerance {
            return Err(format!(
                "sign test diverges at wins {} losses {} ties {}: {got} against exact {}",
                case.wins, case.losses, case.ties, case.one_sided_p
            ));
        }
    }
    Ok(())
}

#[test]
fn the_sign_test_is_the_exact_one_sided_binomial_tail_over_discordant_pairs() {
    check_sign_test(&|pairs| {
        let test = sign_test(pairs);
        let discordant = pairs.iter().filter(|(c, b)| c != b).count();
        assert_eq!(test.discordant, discordant, "ties are not discordant");
        test.p_value
    })
    .unwrap();
    // The same p-value from the counts alone, as the powered exam and the
    // release gate's task-level evidence ask for it.
    check_sign_test(&|pairs| {
        let wins = pairs.iter().filter(|p| **p == (true, false)).count();
        let losses = pairs.iter().filter(|p| **p == (false, true)).count();
        one_sided_p(wins, losses)
    })
    .unwrap();
}

#[test]
fn the_sign_test_check_catches_a_two_sided_test_counted_ties_and_a_strict_tail() {
    let swapped = |pairs: &[(bool, bool)]| -> Vec<(bool, bool)> {
        pairs.iter().map(|&(c, b)| (b, c)).collect()
    };
    let mutants: [(&str, PValueOf<'_>); 3] = [
        ("two-sided", &|pairs| {
            let upper = sign_test(pairs).p_value;
            let lower = sign_test(&swapped(pairs)).p_value;
            (2.0 * upper.min(lower)).min(1.0)
        }),
        ("ties counted as losses", &|pairs| {
            let as_losses: Vec<(bool, bool)> = pairs
                .iter()
                .map(|&(c, b)| if c == b { (false, true) } else { (c, b) })
                .collect();
            sign_test(&as_losses).p_value
        }),
        ("strict tail P(X > k)", &|pairs| {
            // P(X > k) over n is P(X >= k + 1) over the same n: one loss
            // turned into a win; with no loss left it is zero.
            let test = sign_test(pairs);
            let losses = test.discordant - test.candidate_wins;
            if losses == 0 {
                return 0.0;
            }
            let mut shifted = vec![(true, false); test.candidate_wins + 1];
            shifted.extend(vec![(false, true); losses - 1]);
            sign_test(&shifted).p_value
        }),
    ];
    for (name, mutant) in mutants {
        assert!(
            check_sign_test(mutant).is_err(),
            "the sign-test check did not catch the {name} mutant"
        );
    }
}

/// A seed no test depends on, printed so a failure can be replayed.
fn drawn_seed() -> u64 {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    println!("seed {seed}");
    seed
}

/// Resamples per interval: enough that the empirical quantile sits within a
/// few thousandths of the exact one in CDF terms.
const RESAMPLES: usize = 20_000;

/// How far, in CDF terms, an empirical bound may sit from the exact
/// quantile: more than five standard errors of an empirical 2.5% quantile at
/// [`RESAMPLES`] (`sqrt(0.025 * 0.975 / 20000)` is 0.0011).
const CDF_TOLERANCE: f64 = 0.006;

/// Whether `bound` is a bootstrap quantile `q` of `cdf`: on its lattice, with
/// at least `q` of the mass at or below it and at most `q` strictly below.
fn is_quantile(cdf: &[(f64, f64)], step: f64, bound: f64, q: f64) -> Result<(), String> {
    let at = cdf
        .iter()
        .position(|(x, _)| (x - bound).abs() < step * 1e-6)
        .ok_or_else(|| format!("{bound} is not a mean of the resampled items"))?;
    let below = if at == 0 { 0.0 } else { cdf[at - 1].1 };
    if cdf[at].1 + CDF_TOLERANCE < q || below - CDF_TOLERANCE > q {
        return Err(format!(
            "{bound} is not the {q} quantile: CDF {below} below it and {} at it",
            cdf[at].1
        ));
    }
    Ok(())
}

/// An interval method over paired items: the per-item `(candidate right,
/// baseline right)` in, `(mean, low, high)` out.
type IntervalOf<'a> = &'a dyn Fn(&[(bool, bool)], f64, u64) -> (f64, f64, f64);

/// The first fixture case, and the first bound, on which `interval_of`
/// diverges from the exact paired bootstrap distribution. With
/// `detectable_only`, only the cases an unpaired resample is known to fail.
fn check_paired_interval(interval_of: IntervalOf<'_>, detectable_only: bool) -> Result<(), String> {
    let seed = drawn_seed();
    for case in fixture().paired_bootstrap {
        if detectable_only && !case.unpaired_detectable {
            continue;
        }
        let mut items = vec![(true, true); case.both_right];
        items.extend(vec![(true, false); case.candidate_only]);
        items.extend(vec![(false, true); case.baseline_only]);
        items.extend(vec![(false, false); case.both_wrong]);
        let (mean, low, high) = interval_of(&items, case.level, seed);
        let name = &case.name;
        if (mean - case.mean).abs() > 1e-12 {
            return Err(format!("{name}: mean {mean}, exact {}", case.mean));
        }
        let tail = (1.0 - case.level) / 2.0;
        is_quantile(&case.paired_cdf, case.step, low, tail)
            .map_err(|e| format!("{name}: low bound: {e}"))?;
        is_quantile(&case.paired_cdf, case.step, high, 1.0 - tail)
            .map_err(|e| format!("{name}: high bound: {e}"))?;
    }
    Ok(())
}

fn differences(items: &[(bool, bool)]) -> Vec<f64> {
    items
        .iter()
        .map(|&(c, b)| f64::from(u8::from(c)) - f64::from(u8::from(b)))
        .collect()
}

/// The interval as splinter's callers make it: the bootstrap of the per-item
/// differences.
fn splinter_interval(items: &[(bool, bool)], level: f64, seed: u64) -> (f64, f64, f64) {
    let i = bootstrap_interval(&differences(items), RESAMPLES, level, seed).unwrap();
    (i.mean, i.low, i.high)
}

#[test]
fn the_paired_bootstrap_bounds_are_the_exact_quantiles_of_the_resampled_mean() {
    check_paired_interval(&splinter_interval, false).unwrap();
}

#[test]
fn the_interval_check_catches_independent_arms_and_a_one_sided_bound() {
    use splinter_core::rng::SplitMix64;
    // The arms resampled independently: each arm's right share drawn on its
    // own, which ignores that the arms agree on most items.
    let unpaired = |items: &[(bool, bool)], level: f64, seed: u64| {
        let n = items.len();
        let mut rng = SplitMix64::new(seed);
        let mut means: Vec<f64> = (0..RESAMPLES)
            .map(|_| {
                let c = (0..n).filter(|_| items[rng.below(n)].0).count();
                let b = (0..n).filter(|_| items[rng.below(n)].1).count();
                (c as f64 - b as f64) / n as f64
            })
            .collect();
        means.sort_by(f64::total_cmp);
        let at = |q: f64| means[(q * (RESAMPLES - 1) as f64).round() as usize];
        let tail = (1.0 - level) / 2.0;
        let mean = differences(items).iter().sum::<f64>() / n as f64;
        (mean, at(tail), at(1.0 - tail))
    };
    assert!(
        check_paired_interval(&unpaired, true).is_err(),
        "the interval check did not catch arms resampled independently"
    );
    // A 95% one-sided bound passed off as the two-sided interval's.
    let one_sided = |items: &[(bool, bool)], level: f64, seed: u64| {
        let i =
            bootstrap_interval(&differences(items), RESAMPLES, 2.0 * level - 1.0, seed).unwrap();
        (i.mean, i.low, i.high)
    };
    assert!(
        check_paired_interval(&one_sided, false).is_err(),
        "the interval check did not catch a one-sided bound"
    );
}
