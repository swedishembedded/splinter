// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements statistical analysis of paired model
// examinations, for its clients. If your team needs expertise in telling a
// real gain of a fine-tuned model from the luck of a small exam, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What the powered exam concludes from its per-task records, so that the
//! analysis can be redone from the stored report without a model.
//!
//! Tasks about one family of text are not independent, so a family is the
//! unit of resampling: the interval of a difference between two arms is a
//! percentile bootstrap over families, each family contributing its own tasks.
//! The primary test pairs the arms by task, an exact one-sided sign test over
//! the tasks only one of them got right, reported beside the same test over
//! families (each family counted once, as the release gate counts it). The
//! binary verdict of the greedy answer is the primary measure; the mean over
//! every answer sampled for a task (its pass rate) is reported beside it as a
//! continuous score, never a new scale. The report says how large a
//! difference the exam could have shown, from its own data.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One answer an arm gave to a task and how it was graded.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Answer {
    /// The judge's verdict that it gives what the reference says; `None`
    /// where it could not decide.
    pub judged: Option<bool>,
    /// Whether every number and name in it is in the source: `Some(false)`
    /// is an invented specific; `None` where nothing could be checked.
    pub grounded: Option<bool>,
    /// Its length in characters.
    pub chars: usize,
}

/// One task: its family and what each arm answered, the first answer of an
/// arm the greedy one and the rest sampled.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskRecord {
    /// The task's address.
    pub task: String,
    /// Its family.
    pub family: String,
    /// The answers of each arm, by the arm's name.
    pub arms: BTreeMap<String, Vec<Answer>>,
}

impl TaskRecord {
    /// The arm's verdict on the greedy answer.
    fn binary(&self, arm: &str) -> Option<bool> {
        self.arms.get(arm)?.first()?.judged
    }

    /// The share of the arm's decided answers that were right.
    fn score(&self, arm: &str) -> Option<f64> {
        let decided: Vec<bool> = self
            .arms
            .get(arm)?
            .iter()
            .filter_map(|a| a.judged)
            .collect();
        (!decided.is_empty())
            .then(|| decided.iter().filter(|r| **r).count() as f64 / decided.len() as f64)
    }

    fn chars(&self, arm: &str) -> Option<usize> {
        Some(self.arms.get(arm)?.first()?.chars)
    }
}

/// An arm over the whole exam.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ArmSummary {
    /// The arm's name.
    pub arm: String,
    /// Tasks with a verdict on the greedy answer.
    pub judged: usize,
    /// Of those, the ones right.
    pub right: usize,
    /// The mean pass rate over every sampled answer, over the tasks with one.
    pub mean_score: Option<f64>,
    /// Answers checked against the source, over all resamples.
    pub checked: usize,
    /// Of those, the ones that state something the source does not hold.
    pub invented: usize,
    /// The mean length of the greedy answers, in characters.
    pub mean_chars: Option<f64>,
    /// Their median length.
    pub median_chars: Option<f64>,
    /// Their longest.
    pub max_chars: Option<usize>,
}

/// A confidence interval.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Interval {
    /// The difference.
    pub mean: f64,
    /// Its lower end.
    pub low: f64,
    /// Its upper end.
    pub high: f64,
}

/// What the exam could have shown.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Power {
    /// Families the comparison rests on.
    pub families: usize,
    /// Tasks only one of the two arms got right.
    pub discordant_tasks: usize,
    /// The spread, over families, of the family's mean difference.
    pub family_sd: f64,
    /// The smallest true difference in the share of tasks right that this
    /// many families would show at the level the sign test is held to with
    /// four chances in five, by the normal approximation to the families'
    /// mean difference, from this exam's own spread.
    pub minimum_detectable_difference: Option<f64>,
}

/// The tasks both arms answered about as long as each other.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct LengthMatched {
    /// Tasks whose two answers differ in length by at most a quarter of the
    /// longer.
    pub tasks: usize,
    /// Of those, the ones only the first arm got right.
    pub first_only: usize,
    /// And only the second.
    pub second_only: usize,
    /// The one-sided sign test that the first is better, over them.
    pub p_value: f64,
}

/// Two arms compared task by task.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Comparison {
    /// The arm said to be better.
    pub first: String,
    /// The arm it is compared with.
    pub second: String,
    /// Tasks both arms have a verdict on.
    pub tasks: usize,
    /// Both right.
    pub both_right: usize,
    /// Only the first right.
    pub first_only: usize,
    /// Only the second right.
    pub second_only: usize,
    /// Neither right.
    pub both_wrong: usize,
    /// The exact one-sided sign test over tasks (the primary test).
    pub p_tasks: f64,
    /// The same test over families, each counted once by which arm got more
    /// of its tasks right.
    pub p_families: f64,
    /// The first's share right minus the second's, with a family-clustered
    /// bootstrap interval.
    pub difference: Option<Interval>,
    /// The same over the continuous pass rate.
    pub score_difference: Option<Interval>,
    /// The sign test over the tasks the two answered at a like length.
    pub length_matched: Option<LengthMatched>,
    /// What the exam could have shown.
    pub power: Power,
}

/// Resamples of the families an interval is made from.
const RESAMPLES: usize = 4000;
/// The share of resampled differences an interval holds.
const LEVEL: f64 = 0.95;
/// The seed of the resampling: the same records give the same interval.
const SEED: u64 = 0;
/// One-sided 5% and 80% power quantiles of the standard normal.
const Z_ALPHA: f64 = 1.644_853_626_951_472_7;
const Z_POWER: f64 = 0.841_621_233_572_914_4;

/// `P(Binomial(n, 1/2) >= k)`: the one-sided exact sign test.
#[must_use]
pub fn sign_p(first_only: usize, second_only: usize) -> f64 {
    let n = first_only + second_only;
    if n == 0 {
        return 1.0;
    }
    // Summed in log space: n can be a few hundred.
    let ln_half = (0.5f64).ln();
    let ln_fact: Vec<f64> = std::iter::once(0.0)
        .chain((1..=n).scan(0.0, |acc, i| {
            *acc += (i as f64).ln();
            Some(*acc)
        }))
        .collect();
    (first_only..=n)
        .map(|k| (ln_fact[n] - ln_fact[k] - ln_fact[n - k] + n as f64 * ln_half).exp())
        .sum::<f64>()
        .min(1.0)
}

/// The arms' summaries, in the order of `arms`.
#[must_use]
pub fn summarise(records: &[TaskRecord], arms: &[String]) -> Vec<ArmSummary> {
    arms.iter()
        .map(|arm| {
            let verdicts: Vec<bool> = records.iter().filter_map(|r| r.binary(arm)).collect();
            let scores: Vec<f64> = records.iter().filter_map(|r| r.score(arm)).collect();
            let answers = || records.iter().filter_map(|r| r.arms.get(arm)).flatten();
            let mut lengths: Vec<usize> = records.iter().filter_map(|r| r.chars(arm)).collect();
            lengths.sort_unstable();
            ArmSummary {
                arm: arm.clone(),
                judged: verdicts.len(),
                right: verdicts.iter().filter(|v| **v).count(),
                mean_score: mean(&scores),
                checked: answers().filter(|a| a.grounded.is_some()).count(),
                invented: answers().filter(|a| a.grounded == Some(false)).count(),
                mean_chars: mean(&lengths.iter().map(|&l| l as f64).collect::<Vec<_>>()),
                median_chars: (!lengths.is_empty()).then(|| {
                    let mid = lengths.len() / 2;
                    if lengths.len() % 2 == 1 {
                        lengths[mid] as f64
                    } else {
                        (lengths[mid - 1] + lengths[mid]) as f64 / 2.0
                    }
                }),
                max_chars: lengths.last().copied(),
            }
        })
        .collect()
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// `first` compared with `second` over `records`.
#[must_use]
pub fn compare(records: &[TaskRecord], first: &str, second: &str) -> Comparison {
    struct Pair<'a> {
        family: &'a str,
        a: bool,
        b: bool,
        score: Option<f64>,
        chars: Option<(usize, usize)>,
    }
    let pairs: Vec<Pair<'_>> = records
        .iter()
        .filter_map(|r| {
            Some(Pair {
                family: &r.family,
                a: r.binary(first)?,
                b: r.binary(second)?,
                score: r.score(first).zip(r.score(second)).map(|(a, b)| a - b),
                chars: r.chars(first).zip(r.chars(second)),
            })
        })
        .collect();
    let count = |a: bool, b: bool| pairs.iter().filter(|p| p.a == a && p.b == b).count();
    let (first_only, second_only) = (count(true, false), count(false, true));

    // Per family: the differences of its tasks.
    let mut by_family: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut score_by_family: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for p in &pairs {
        by_family
            .entry(p.family)
            .or_default()
            .push(f64::from(p.a) - f64::from(p.b));
        if let Some(d) = p.score {
            score_by_family.entry(p.family).or_default().push(d);
        }
    }
    let (family_wins, family_losses) = by_family.values().fold((0, 0), |(w, l), diffs| {
        let net: f64 = diffs.iter().sum();
        (w + usize::from(net > 0.0), l + usize::from(net < 0.0))
    });
    let family_means: Vec<f64> = by_family
        .values()
        .map(|d| d.iter().sum::<f64>() / d.len() as f64)
        .collect();
    let family_sd = sd(&family_means);
    let matched: Vec<&Pair<'_>> = pairs
        .iter()
        .filter(|p| {
            p.chars.is_some_and(|(x, y)| {
                let longer = x.max(y);
                longer == 0 || x.abs_diff(y) * 4 <= longer
            })
        })
        .collect();
    let (m_first, m_second) = (
        matched.iter().filter(|p| p.a && !p.b).count(),
        matched.iter().filter(|p| !p.a && p.b).count(),
    );
    Comparison {
        first: first.into(),
        second: second.into(),
        tasks: pairs.len(),
        both_right: count(true, true),
        first_only,
        second_only,
        both_wrong: count(false, false),
        p_tasks: sign_p(first_only, second_only),
        p_families: sign_p(family_wins, family_losses),
        difference: clustered_interval(&by_family),
        score_difference: clustered_interval(&score_by_family),
        length_matched: (!matched.is_empty()).then(|| LengthMatched {
            tasks: matched.len(),
            first_only: m_first,
            second_only: m_second,
            p_value: sign_p(m_first, m_second),
        }),
        power: Power {
            families: by_family.len(),
            discordant_tasks: first_only + second_only,
            family_sd,
            minimum_detectable_difference: (by_family.len() >= 2)
                .then(|| (Z_ALPHA + Z_POWER) * family_sd / (by_family.len() as f64).sqrt()),
        },
    }
}

/// The sample standard deviation; zero for fewer than two values.
fn sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
}

/// The percentile bootstrap interval of the mean difference over tasks,
/// resampling whole families with replacement. The statistic is the mean
/// over all the tasks of the families drawn, so a large family weighs what
/// its tasks weigh. The generator is seeded, so the same records give the
/// same interval.
fn clustered_interval(by_family: &BTreeMap<&str, Vec<f64>>) -> Option<Interval> {
    let families: Vec<(f64, usize)> = by_family
        .values()
        .map(|f| (f.iter().sum::<f64>(), f.len()))
        .collect();
    if families.len() < 2 {
        return None;
    }
    let (total, tasks) = families
        .iter()
        .fold((0.0, 0), |(s, n), (fs, fnn)| (s + fs, n + fnn));
    let mut state = SEED;
    let mut next = || {
        // splitmix64: a small, repeatable generator, enough to pick indices.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut means: Vec<f64> = (0..RESAMPLES)
        .map(|_| {
            let (sum, count) = (0..families.len()).fold((0.0, 0usize), |(s, n), _| {
                let (fs, fnn) =
                    families[usize::try_from(next() % families.len() as u64).unwrap_or(0)];
                (s + fs, n + fnn)
            });
            sum / count as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let at = |share: f64| {
        means[((share * (means.len() - 1) as f64).round() as usize).min(means.len() - 1)]
    };
    Some(Interval {
        mean: total / tasks as f64,
        low: at((1.0 - LEVEL) / 2.0),
        high: at(1.0 - (1.0 - LEVEL) / 2.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(judged: bool, chars: usize) -> Answer {
        Answer {
            judged: Some(judged),
            grounded: Some(true),
            chars,
        }
    }

    fn record(family: &str, n: usize, a: bool, b: bool) -> TaskRecord {
        TaskRecord {
            task: format!("{family}-{n}"),
            family: family.into(),
            arms: [
                ("a".to_string(), vec![answer(a, 100)]),
                ("b".to_string(), vec![answer(b, 100)]),
            ]
            .into(),
        }
    }

    #[test]
    fn the_exact_sign_test_is_the_binomial_tail() {
        assert!((sign_p(5, 0) - 0.03125).abs() < 1e-12);
        assert!((sign_p(3, 1) - 0.3125).abs() < 1e-12);
        assert_eq!(sign_p(0, 0), 1.0);
        assert!(sign_p(0, 4) > 0.99);
    }

    /// Twelve wins in one family and nothing elsewhere: the task-level test
    /// calls it significant, the family counts it once and the interval over
    /// families holds zero.
    #[test]
    fn many_wins_in_one_family_are_not_many_units_of_evidence() {
        let mut records: Vec<TaskRecord> = (0..12).map(|n| record("one", n, true, false)).collect();
        for family in ["two", "three", "four", "five"] {
            records.extend((0..3).map(|n| record(family, n, true, true)));
        }
        let c = compare(&records, "a", "b");
        assert_eq!((c.first_only, c.second_only), (12, 0));
        assert!(c.p_tasks < 0.001, "{}", c.p_tasks);
        assert!(c.p_families >= 0.5, "{}", c.p_families);
        let interval = c.difference.unwrap();
        assert!(interval.low <= 0.0, "{interval:?}");
        assert_eq!(c.power.families, 5);
    }

    #[test]
    fn a_win_spread_over_many_families_has_an_interval_above_zero() {
        let records: Vec<TaskRecord> = (0..20)
            .flat_map(|f| {
                let family = format!("f{f}");
                (0..3).map(move |n| record(&family, n, n < 2, false))
            })
            .collect();
        let c = compare(&records, "a", "b");
        assert_eq!(c.first_only, 40);
        let interval = c.difference.unwrap();
        assert!((interval.mean - 2.0 / 3.0).abs() < 1e-9);
        assert!(interval.low > 0.0, "{interval:?}");
        assert!(c.p_families < 1e-5);
    }

    #[test]
    fn the_minimum_detectable_difference_comes_from_the_spread_over_families() {
        // Every family differs by 0 or 1: sd 0.5 over 16 families.
        let records: Vec<TaskRecord> = (0..16)
            .map(|f| record(&format!("f{f}"), 0, f % 2 == 0, false))
            .collect();
        let c = compare(&records, "a", "b");
        let expected = (Z_ALPHA + Z_POWER) * c.power.family_sd / 4.0;
        assert!((c.power.minimum_detectable_difference.unwrap() - expected).abs() < 1e-12);
        assert!(c.power.minimum_detectable_difference.unwrap() > 0.3);
    }

    #[test]
    fn the_length_matched_test_leaves_out_answers_of_unlike_length() {
        let mut win = record("f", 0, true, false);
        // The first arm's answer is three times as long: not matched.
        win.arms.get_mut("a").unwrap()[0].chars = 300;
        let like = record("g", 0, true, false);
        let c = compare(&[win, like], "a", "b");
        let matched = c.length_matched.unwrap();
        assert_eq!((matched.tasks, matched.first_only), (1, 1));
    }

    #[test]
    fn the_summary_counts_sampled_answers_and_invented_specifics() {
        let mut r = record("f", 0, true, false);
        r.arms.get_mut("a").unwrap().push(Answer {
            judged: Some(false),
            grounded: Some(false),
            chars: 300,
        });
        let summary = summarise(&[r], &["a".to_string()]);
        assert_eq!(summary[0].right, 1, "the greedy answer decides");
        assert_eq!(summary[0].mean_score, Some(0.5), "the pass rate over both");
        assert_eq!((summary[0].checked, summary[0].invented), (2, 1));
    }
}
