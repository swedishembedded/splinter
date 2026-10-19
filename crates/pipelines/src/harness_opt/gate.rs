// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The promotion rule for a candidate harness: what the paired comparison of
//! its verified attempts against the baseline's must show.
//!
//! A candidate is promoted only when no family of tasks got worse (a family
//! the baseline won more often than the candidate is a regression, whatever
//! the totals say) and the one-sided sign test over families - the release
//! gate's own statistic - finds the candidate better at the stated level.
//! Anything short of that is rejected (a regression) or inconclusive (no
//! regression, but no evidence of improvement), and the dense score the
//! search follows is only guidance: it never promotes.

use serde::Serialize;

use crate::compare::SetComparison;

/// What the evidence has to show.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionRule {
    /// The sign test's p-value must be below this.
    pub alpha: f64,
    /// The fewest tasks with a verdict in both arms; fewer is inconclusive.
    pub min_paired: usize,
}

/// The numbers a decision rests on, read from a comparison.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Evidence {
    /// Tasks with a verdict in both arms.
    pub paired: usize,
    /// Paired tasks the candidate got right.
    pub candidate_right: usize,
    /// Paired tasks the baseline got right.
    pub baseline_right: usize,
    /// Families the baseline won more of than the candidate.
    pub regressed_families: Vec<String>,
    /// Families the candidate won more of than the baseline.
    pub improved_families: usize,
    /// The one-sided sign test's p-value over families.
    pub p_value: f64,
}

impl Evidence {
    /// The evidence of `comparison`.
    #[must_use]
    pub fn of(comparison: &SetComparison) -> Self {
        Self {
            paired: comparison.paired,
            candidate_right: comparison.families.iter().map(|f| f.candidate_right).sum(),
            baseline_right: comparison.families.iter().map(|f| f.baseline_right).sum(),
            regressed_families: comparison
                .families
                .iter()
                .filter(|f| f.losses > f.wins)
                .map(|f| f.family.clone())
                .collect(),
            improved_families: comparison.clusters.wins,
            p_value: comparison.sign_test.p_value,
        }
    }

    /// The share of paired tasks the candidate got right; zero with none.
    /// Dense guidance for the search, not a verdict.
    #[must_use]
    pub fn utility(&self) -> f64 {
        if self.paired == 0 {
            0.0
        } else {
            self.candidate_right as f64 / self.paired as f64
        }
    }
}

/// What to do with a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Verified improvement with no regression.
    Promote,
    /// A family got worse.
    Reject,
    /// No regression, no evidence of improvement.
    Inconclusive,
}

/// A verdict and why.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Decision {
    /// The verdict.
    pub verdict: Verdict,
    /// Why, in a sentence.
    pub reason: String,
}

/// Decides on `evidence` under `rule`.
#[must_use]
pub fn decide(evidence: &Evidence, rule: &PromotionRule) -> Decision {
    let decision = |verdict, reason: String| Decision { verdict, reason };
    if !evidence.regressed_families.is_empty() {
        return decision(
            Verdict::Reject,
            format!("regressed: {}", evidence.regressed_families.join(", ")),
        );
    }
    if evidence.paired < rule.min_paired {
        return decision(
            Verdict::Inconclusive,
            format!(
                "{} tasks with a verdict in both arms, at least {} needed",
                evidence.paired, rule.min_paired
            ),
        );
    }
    if evidence.improved_families > 0 && evidence.p_value < rule.alpha {
        return decision(
            Verdict::Promote,
            format!(
                "{} families improved, none regressed, sign test p = {:.4} < {}",
                evidence.improved_families, evidence.p_value, rule.alpha
            ),
        );
    }
    decision(
        Verdict::Inconclusive,
        format!(
            "no family regressed, but sign test p = {:.4} is not below {}",
            evidence.p_value, rule.alpha
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULE: PromotionRule = PromotionRule {
        alpha: 0.05,
        min_paired: 4,
    };

    fn evidence(regressed: &[&str], improved: usize, p_value: f64, paired: usize) -> Evidence {
        Evidence {
            paired,
            candidate_right: paired / 2,
            baseline_right: paired / 2,
            regressed_families: regressed.iter().map(ToString::to_string).collect(),
            improved_families: improved,
            p_value,
        }
    }

    #[test]
    fn an_improvement_with_no_regression_and_a_small_p_value_is_promoted() {
        let d = decide(&evidence(&[], 5, 0.03, 10), &RULE);
        assert_eq!(d.verdict, Verdict::Promote, "{}", d.reason);
    }

    #[test]
    fn one_regressed_family_rejects_whatever_the_totals_say() {
        let d = decide(&evidence(&["parsing"], 9, 0.001, 20), &RULE);
        assert_eq!(d.verdict, Verdict::Reject);
        assert!(d.reason.contains("parsing"));
    }

    #[test]
    fn too_little_evidence_or_a_large_p_value_is_inconclusive_never_a_promotion() {
        assert_eq!(
            decide(&evidence(&[], 2, 0.01, 3), &RULE).verdict,
            Verdict::Inconclusive
        );
        assert_eq!(
            decide(&evidence(&[], 3, 0.2, 10), &RULE).verdict,
            Verdict::Inconclusive
        );
        assert_eq!(
            decide(&evidence(&[], 0, 1.0, 10), &RULE).verdict,
            Verdict::Inconclusive
        );
    }

    #[test]
    fn utility_is_the_candidates_share_and_absent_without_pairs() {
        assert_eq!(evidence(&[], 0, 1.0, 0).utility(), 0.0);
        assert!((evidence(&[], 0, 1.0, 10).utility() - 0.5).abs() < 1e-12);
    }
}
