// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! Arms measured over many corpora, with intervals.
//!
//! A corpus is the unit: each arm has one value per corpus, the interval is a
//! percentile bootstrap of their mean, and a difference between two arms is
//! taken corpus by corpus (paired), so a gain that rests on a few corpora
//! shows as a wide interval.

use serde::Serialize;
use splinter_model::stats::bootstrap_interval;

use super::compare::Arm;

/// The resamples of every interval.
pub const BOOTSTRAP_RESAMPLES: usize = 4000;
/// The level of every interval.
pub const LEVEL: f64 = 0.95;

/// A mean over corpora with its interval.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Metric {
    /// The mean.
    pub mean: f64,
    /// The interval's low end; equal to the mean for one corpus.
    pub low: f64,
    /// The interval's high end.
    pub high: f64,
    /// The corpora it is over.
    pub n: usize,
}

impl Metric {
    /// Of `values`, resampled under `seed`; `None` when there are none.
    #[must_use]
    pub fn of(values: &[f64], seed: u64) -> Option<Self> {
        let n = values.len();
        if n == 0 {
            return None;
        }
        let mean = values.iter().sum::<f64>() / n as f64;
        Some(
            match bootstrap_interval(values, BOOTSTRAP_RESAMPLES, LEVEL, seed) {
                Some(i) => Self {
                    mean: i.mean,
                    low: i.low,
                    high: i.high,
                    n,
                },
                None => Self {
                    mean,
                    low: mean,
                    high: mean,
                    n,
                },
            },
        )
    }

    /// Whether the interval excludes zero on the positive side.
    #[must_use]
    pub fn above_zero(&self) -> bool {
        self.n >= 2 && self.low > 0.0
    }
}

/// One arm over all corpora.
#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    /// The arm.
    pub name: String,
    /// Held-out pass rate.
    pub held_out: Metric,
    /// Share of learned sections labelled relevant, over the corpora where it learned any.
    pub precision: Option<Metric>,
    /// Share of the relevant sections learned.
    pub recall: Metric,
    /// Words learned.
    pub words: Metric,
    /// Held-out pass rate minus the reference arm's, corpus by corpus; absent for the reference.
    pub versus_reference: Option<Metric>,
}

/// Summaries of every arm of `runs` (one `Vec<Arm>` per corpus, the same arms
/// in each), paired against the arm named `reference`. `seed` makes the
/// intervals repeatable.
#[must_use]
pub fn summarise(runs: &[Vec<Arm>], reference: &str, seed: u64) -> Vec<Summary> {
    let Some(first) = runs.first() else {
        return Vec::new();
    };
    let value =
        |run: &[Arm], name: &str, f: fn(&Arm) -> f64| run.iter().find(|a| a.name == name).map(f);
    first
        .iter()
        .filter_map(|arm| {
            let name = arm.name.as_str();
            let column = |f: fn(&Arm) -> f64| -> Vec<f64> {
                runs.iter().filter_map(|r| value(r, name, f)).collect()
            };
            let precision: Vec<f64> = runs
                .iter()
                .filter_map(|r| r.iter().find(|a| a.name == name)?.precision)
                .collect();
            let diff: Vec<f64> = runs
                .iter()
                .filter_map(|r| {
                    Some(
                        value(r, name, |a| a.held_out_pass)?
                            - value(r, reference, |a| a.held_out_pass)?,
                    )
                })
                .collect();
            Some(Summary {
                name: arm.name.clone(),
                held_out: Metric::of(&column(|a| a.held_out_pass), seed)?,
                precision: Metric::of(&precision, seed),
                recall: Metric::of(&column(|a| a.recall), seed)?,
                words: Metric::of(&column(|a| a.words as f64), seed)?,
                versus_reference: (name != reference)
                    .then(|| Metric::of(&diff, seed))
                    .flatten(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(name: &str, held_out: f64) -> Arm {
        Arm {
            name: name.into(),
            sections: 1,
            words: 10,
            questions: 0,
            dev_pass: 0.0,
            held_out_pass: held_out,
            precision: Some(1.0),
            recall: 1.0,
            harmful: None,
            learned: Vec::new(),
            report: None,
        }
    }

    #[test]
    fn a_gain_is_paired_corpus_by_corpus() {
        let runs: Vec<Vec<Arm>> = (0..10)
            .map(|i| {
                vec![
                    arm("a", 0.5 + 0.01 * f64::from(i)),
                    arm("b", 0.2 + 0.01 * f64::from(i)),
                ]
            })
            .collect();
        let summary = summarise(&runs, "a", 1); // seed: the interval's resampling
        let b = &summary[1];
        let gain = b.versus_reference.unwrap();
        assert!(
            (gain.mean + 0.3).abs() < 1e-9 && gain.high < 0.0,
            "{gain:?}"
        );
        assert!(summary[0].versus_reference.is_none());
        assert_eq!(b.held_out.n, 10);
    }
}
