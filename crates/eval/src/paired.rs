// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements paired evaluations that compare two
// models on the same held-out items, for its clients. If your team needs
// expertise in model evaluation design, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Two models graded on the same items: which items both were graded on,
//! and how they compare there. Pure: no model, no files.
//!
//! An item's outcome for one model is `Some(true)` (graded right),
//! `Some(false)` (graded wrong) or `None` (no verdict: the verifiers
//! abstained or disagreed). An item only counts when both models have a
//! verdict on it; the rest are *unpaired*, excluded from every number
//! below and counted. Accuracy is over the paired items, so the two
//! accuracies are over the same items and their difference means
//! something; with no paired item there is no accuracy, never a `0`.

use serde::{Deserialize, Serialize};

/// One item's outcome for both models.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairedOutcome {
    /// What the item is (a task's address).
    pub item: String,
    /// The candidate's verdict; `None` when it has none.
    pub candidate: Option<bool>,
    /// The baseline's verdict; `None` when it has none.
    pub baseline: Option<bool>,
    /// The family of sources the item comes from, when it has one: items of
    /// one family are one unit of evidence ([`by_cluster`]), not several.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster: Option<String>,
}

impl PairedOutcome {
    /// Both verdicts, when both exist.
    #[must_use]
    pub fn paired(&self) -> Option<(bool, bool)> {
        Some((self.candidate?, self.baseline?))
    }
}

/// How two models compare over paired items.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    /// Items compared, paired or not.
    pub items: usize,
    /// Items without a verdict for both models: excluded.
    pub unpaired: usize,
    /// Items with a verdict for both.
    pub paired: usize,
    /// Paired items both got right: a tie.
    pub both_right: usize,
    /// Paired items both got wrong: a tie.
    pub both_wrong: usize,
    /// Paired items only the candidate got right.
    pub candidate_wins: usize,
    /// Paired items only the baseline got right.
    pub baseline_wins: usize,
    /// The candidate's accuracy over the paired items; `None` with none.
    pub candidate_accuracy: Option<f64>,
    /// The baseline's accuracy over the paired items; `None` with none.
    pub baseline_accuracy: Option<f64>,
}

impl Comparison {
    /// How far the candidate's accuracy falls below the baseline's
    /// (negative when it is higher); `None` with no paired item.
    #[must_use]
    pub fn drop(&self) -> Option<f64> {
        Some(self.baseline_accuracy? - self.candidate_accuracy?)
    }
}

/// `outcomes` compared; see the module documentation.
#[must_use]
pub fn compare(outcomes: &[PairedOutcome]) -> Comparison {
    let mut comparison = Comparison {
        items: outcomes.len(),
        unpaired: 0,
        paired: 0,
        both_right: 0,
        both_wrong: 0,
        candidate_wins: 0,
        baseline_wins: 0,
        candidate_accuracy: None,
        baseline_accuracy: None,
    };
    for outcome in outcomes {
        match outcome.paired() {
            None => comparison.unpaired += 1,
            Some((true, true)) => comparison.both_right += 1,
            Some((false, false)) => comparison.both_wrong += 1,
            Some((true, false)) => comparison.candidate_wins += 1,
            Some((false, true)) => comparison.baseline_wins += 1,
        }
    }
    comparison.paired = comparison.items - comparison.unpaired;
    if comparison.paired > 0 {
        let paired = comparison.paired as f64;
        let right = |wins: usize| (comparison.both_right + wins) as f64 / paired;
        comparison.candidate_accuracy = Some(right(comparison.candidate_wins));
        comparison.baseline_accuracy = Some(right(comparison.baseline_wins));
    }
    comparison
}

/// The pairs a significance test may count: one per cluster, and one per
/// item that has none.
///
/// Items drawn from one family of sources (two editions of one letter, a
/// handful of questions about one document) are not independent: a model
/// that knows the letter gets them all right, one that does not gets them all
/// wrong. Testing them as separate items counts one piece of evidence many
/// times. A cluster counts once, as a candidate win when the candidate won
/// more of its items than the baseline did, a baseline win when fewer, and a
/// tie - a pair both got right, which a sign test drops - when equal.
#[must_use]
pub fn by_cluster(outcomes: &[PairedOutcome]) -> Vec<(bool, bool)> {
    let mut net: std::collections::BTreeMap<&str, i64> = std::collections::BTreeMap::new();
    let mut pairs = Vec::new();
    for outcome in outcomes {
        let Some((candidate, baseline)) = outcome.paired() else {
            continue;
        };
        match outcome.cluster.as_deref() {
            None => pairs.push((candidate, baseline)),
            Some(cluster) => {
                *net.entry(cluster).or_default() += i64::from(candidate) - i64::from(baseline);
            }
        }
    }
    pairs.extend(net.into_values().map(|net| match net.signum() {
        1 => (true, false),
        -1 => (false, true),
        _ => (true, true),
    }));
    pairs
}

/// The fraction of `outcomes` graded right among those graded at all, and
/// how many were graded; `None` accuracy when none was.
#[must_use]
pub fn accuracy(outcomes: &[Option<bool>]) -> (Option<f64>, usize) {
    let graded: Vec<bool> = outcomes.iter().flatten().copied().collect();
    if graded.is_empty() {
        return (None, 0);
    }
    let right = graded.iter().filter(|r| **r).count();
    (Some(right as f64 / graded.len() as f64), graded.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(candidate: Option<bool>, baseline: Option<bool>) -> PairedOutcome {
        PairedOutcome {
            item: String::new(),
            candidate,
            baseline,
            cluster: None,
        }
    }

    fn in_cluster(cluster: &str, candidate: bool, baseline: bool) -> PairedOutcome {
        PairedOutcome {
            cluster: Some(cluster.into()),
            ..outcome(Some(candidate), Some(baseline))
        }
    }

    #[test]
    fn a_cluster_is_one_unit_of_evidence_whatever_its_size() {
        let outcomes = [
            // Six questions about one family, all won by the candidate.
            in_cluster("a", true, false),
            in_cluster("a", true, false),
            in_cluster("a", true, false),
            in_cluster("a", true, false),
            in_cluster("a", true, false),
            in_cluster("a", true, false),
            // One lost, in another family.
            in_cluster("b", false, true),
            // A family split evenly is a tie.
            in_cluster("c", true, false),
            in_cluster("c", false, true),
            // Items with no family count one by one.
            outcome(Some(true), Some(false)),
            // No verdict for both: not counted.
            outcome(None, Some(true)),
        ];
        let pairs = by_cluster(&outcomes);
        let wins = pairs.iter().filter(|p| **p == (true, false)).count();
        let losses = pairs.iter().filter(|p| **p == (false, true)).count();
        let ties = pairs.iter().filter(|p| p.0 == p.1).count();
        assert_eq!((wins, losses, ties), (2, 1, 1), "{pairs:?}");
    }

    #[test]
    fn unpaired_items_are_excluded_and_counted() {
        let outcomes = [
            outcome(Some(true), Some(false)),
            outcome(Some(true), Some(true)),
            outcome(Some(false), Some(false)),
            outcome(Some(false), Some(true)),
            outcome(None, Some(true)),
            outcome(Some(true), None),
        ];
        let c = compare(&outcomes);
        assert_eq!((c.items, c.unpaired, c.paired), (6, 2, 4));
        assert_eq!((c.candidate_wins, c.baseline_wins), (1, 1));
        assert_eq!((c.both_right, c.both_wrong), (1, 1));
        assert_eq!(c.candidate_accuracy, Some(0.5));
        assert_eq!(c.baseline_accuracy, Some(0.5));
        assert_eq!(c.drop(), Some(0.0));
    }

    #[test]
    fn nothing_paired_is_unmeasured_not_zero() {
        let c = compare(&[outcome(None, Some(true))]);
        assert_eq!(c.paired, 0);
        assert_eq!((c.candidate_accuracy, c.drop()), (None, None));
        assert_eq!(accuracy(&[None, None]), (None, 0));
        assert_eq!(accuracy(&[Some(true), None, Some(false)]), (Some(0.5), 2));
    }
}
