// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in reporting small-sample results with exact intervals, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The proportions the report states: k of n facts with the exact
//! (Clopper-Pearson) interval, and the difference of two proportions with
//! Newcombe's hybrid score interval built from Wilson intervals.

use serde::{Deserialize, Serialize};

/// The confidence level of every interval: two-sided 95%.
pub const ALPHA: f64 = 0.05;

/// k successes out of n, with the exact interval.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proportion {
    /// Successes.
    pub k: usize,
    /// Trials.
    pub n: usize,
}

impl Proportion {
    /// `k` of `n`.
    ///
    /// # Panics
    /// `k` exceeds `n`: a count of successes cannot.
    #[must_use]
    pub fn new(k: usize, n: usize) -> Self {
        assert!(k <= n, "{k} successes out of {n} trials");
        Self { k, n }
    }

    /// The share; absent over no trials (unmeasured is not zero).
    #[must_use]
    pub fn rate(&self) -> Option<f64> {
        (self.n > 0).then(|| self.k as f64 / self.n as f64)
    }

    /// The exact two-sided 95% interval; absent over no trials.
    #[must_use]
    pub fn clopper_pearson(&self) -> Option<(f64, f64)> {
        if self.n == 0 {
            return None;
        }
        let tail = ALPHA / 2.0;
        let lower = if self.k == 0 {
            0.0
        } else {
            // The p at which seeing k or more has probability `tail`.
            bisect(|p| upper_tail(self.k, self.n, p) - tail)
        };
        let upper = if self.k == self.n {
            1.0
        } else {
            // The p at which seeing k or fewer has probability `tail`.
            bisect(|p| tail - lower_tail(self.k, self.n, p))
        };
        Some((lower, upper))
    }

    /// The Wilson score interval; absent over no trials.
    #[must_use]
    pub fn wilson(&self) -> Option<(f64, f64)> {
        let p = self.rate()?;
        let n = self.n as f64;
        let z = 1.959_963_984_540_054_f64;
        let denominator = 1.0 + z * z / n;
        let centre = (p + z * z / (2.0 * n)) / denominator;
        let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denominator;
        Some(((centre - half).max(0.0), (centre + half).min(1.0)))
    }
}

/// The difference `a` minus `b` with Newcombe's hybrid score interval;
/// absent when either has no trials.
#[must_use]
pub fn difference(a: &Proportion, b: &Proportion) -> Option<(f64, (f64, f64))> {
    let (pa, pb) = (a.rate()?, b.rate()?);
    let ((la, ua), (lb, ub)) = (a.wilson()?, b.wilson()?);
    let d = pa - pb;
    let low = d - ((pa - la).powi(2) + (ub - pb).powi(2)).sqrt();
    let high = d + ((ua - pa).powi(2) + (pb - lb).powi(2)).sqrt();
    Some((d, (low.max(-1.0), high.min(1.0))))
}

/// ln(i!) for i in 0..=n.
fn log_factorials(n: usize) -> Vec<f64> {
    let mut table = Vec::with_capacity(n + 1);
    let mut sum = 0.0;
    table.push(0.0);
    for i in 1..=n {
        sum += (i as f64).ln();
        table.push(sum);
    }
    table
}

/// P(X = i) for X ~ Binomial(n, p), for each i in 0..=n.
fn pmf(n: usize, p: f64) -> Vec<f64> {
    let table = log_factorials(n);
    (0..=n)
        .map(|i| {
            let log_choose = table[n] - table[i] - table[n - i];
            let term = |count: usize, prob: f64| {
                if count == 0 {
                    0.0
                } else {
                    count as f64 * prob.ln()
                }
            };
            (log_choose + term(i, p) + term(n - i, 1.0 - p)).exp()
        })
        .collect()
}

/// P(X >= k).
fn upper_tail(k: usize, n: usize, p: f64) -> f64 {
    pmf(n, p)[k..].iter().sum()
}

/// P(X <= k).
fn lower_tail(k: usize, n: usize, p: f64) -> f64 {
    pmf(n, p)[..=k].iter().sum()
}

/// The p in (0, 1) at which `f` crosses zero from below, `f` being increasing.
fn bisect(f: impl Fn(f64) -> f64) -> f64 {
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        if f(mid) < 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
    (low + high) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 5e-4
    }

    #[test]
    fn forty_of_forty_has_the_lower_bound_the_protocol_quotes() {
        let (low, high) = Proportion::new(40, 40)
            .clopper_pearson()
            .expect("an interval");
        assert!(close(low, 0.912), "{low}");
        assert_eq!(high, 1.0);
        let (low, high) = Proportion::new(0, 40)
            .clopper_pearson()
            .expect("an interval");
        assert_eq!(low, 0.0);
        assert!(close(high, 0.088), "{high}");
    }

    #[test]
    fn the_exact_interval_matches_the_published_value_for_eight_of_ten() {
        let (low, high) = Proportion::new(8, 10)
            .clopper_pearson()
            .expect("an interval");
        assert!(close(low, 0.4439) && close(high, 0.9748), "{low} {high}");
    }

    #[test]
    fn nothing_measured_has_no_rate_and_no_interval() {
        let none = Proportion::new(0, 0);
        assert_eq!(none.rate(), None);
        assert_eq!(none.clopper_pearson(), None);
        assert_eq!(difference(&none, &Proportion::new(1, 2)), None);
    }

    #[test]
    fn newcombes_difference_matches_the_published_example() {
        // Newcombe (1998), 56/70 against 48/80: 0.2 (0.0524, 0.3339).
        let (d, (low, high)) =
            difference(&Proportion::new(56, 70), &Proportion::new(48, 80)).expect("a difference");
        assert!(close(d, 0.2), "{d}");
        assert!(close(low, 0.0524) && close(high, 0.3339), "{low} {high}");
    }
}
