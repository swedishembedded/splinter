// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in sizing and
// analysing a paired evaluation with honest statistics, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The small exact statistics a replication study rests on: the binomial tail
//! that makes a sign test, the Holm step-down correction over a pre-registered
//! family, and the spread of a measure over training seeds.
//!
//! Everything is computed in log space, so a tail over a thousand trials is as
//! exact as one over eight.

use serde::Serialize;

/// `ln(n!)` for every `n` up to and including `max`.
fn ln_factorials(max: usize) -> Vec<f64> {
    let mut table = Vec::with_capacity(max + 1);
    table.push(0.0);
    for n in 1..=max {
        table.push(table[n - 1] + (n as f64).ln());
    }
    table
}

/// The probabilities `P(X = k)` for `k` in `0..=n` of a binomial with `n`
/// trials and success probability `p`.
#[must_use]
pub fn binomial_pmf(n: usize, p: f64) -> Vec<f64> {
    if p <= 0.0 {
        return (0..=n).map(|k| f64::from(u8::from(k == 0))).collect();
    }
    if p >= 1.0 {
        return (0..=n).map(|k| f64::from(u8::from(k == n))).collect();
    }
    let ln = ln_factorials(n);
    let (ln_p, ln_q) = (p.ln(), (1.0 - p).ln());
    (0..=n)
        .map(|k| (ln[n] - ln[k] - ln[n - k] + k as f64 * ln_p + (n - k) as f64 * ln_q).exp())
        .collect()
}

/// `P(X >= k)` for a binomial with `n` trials and success probability `p`.
#[must_use]
pub fn binomial_upper_tail(n: usize, k: usize, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    // Summing the small terms first keeps the tail accurate when it is tiny.
    let pmf = binomial_pmf(n, p);
    pmf[k..].iter().rev().sum::<f64>().min(1.0)
}

/// The one-sided exact sign test: the chance of `wins` or more of `n`
/// non-tied comparisons going the candidate's way if each were a coin toss.
/// Zero comparisons carry no evidence, so the p-value is 1.
#[must_use]
pub fn sign_test_p(wins: usize, n: usize) -> f64 {
    binomial_upper_tail(n, wins, 0.5)
}

/// The least number of wins at which the one-sided sign test over `n`
/// comparisons is significant at `alpha`, or `None` when even `n` of `n` is
/// not.
#[must_use]
pub fn critical_wins(n: usize, alpha: f64) -> Option<usize> {
    let pmf = binomial_pmf(n, 0.5);
    let mut tail = 0.0;
    let mut critical = None;
    for k in (0..=n).rev() {
        tail += pmf[k];
        if tail > alpha {
            break;
        }
        critical = Some(k);
    }
    critical
}

/// The Holm step-down adjusted p-values of a pre-registered family of
/// `family_size` comparisons, of which `p_values` holds the ones that were
/// measured (`None` for one that was not). A comparison that was not measured
/// has no adjusted value and still counts in the family: shrinking the family
/// after seeing which comparisons exist would make the correction lenient.
///
/// # Panics
/// `family_size` is smaller than the number of p-values given.
#[must_use]
pub fn holm(p_values: &[Option<f64>], family_size: usize) -> Vec<Option<f64>> {
    assert!(
        family_size >= p_values.len(),
        "the family has {family_size} comparisons and {} were given",
        p_values.len()
    );
    let mut measured: Vec<(usize, f64)> = p_values
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.map(|p| (i, p)))
        .collect();
    measured.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut adjusted = vec![None; p_values.len()];
    let mut running: f64 = 0.0;
    for (rank, (index, p)) in measured.into_iter().enumerate() {
        running = running.max(((family_size - rank) as f64 * p).min(1.0));
        adjusted[index] = Some(running);
    }
    adjusted
}

/// How a measure varies over the training seeds of one author.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Spread {
    /// How many seeds measured it.
    pub seeds: usize,
    /// The smallest value.
    pub min: f64,
    /// The largest value.
    pub max: f64,
    /// The sample standard deviation; absent for a single seed, which has no
    /// spread to speak of.
    pub sd: Option<f64>,
}

/// The spread of `values`, or `None` when there are none.
#[must_use]
pub fn spread(values: &[f64]) -> Option<Spread> {
    let first = *values.first()?;
    let n = values.len();
    let (min, max) = values
        .iter()
        .fold((first, first), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let sd = (n > 1).then(|| {
        let mean = values.iter().sum::<f64>() / n as f64;
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    });
    Some(Spread {
        seeds: n,
        min,
        max,
        sd,
    })
}
