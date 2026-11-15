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
//!
//! The sign test is the default. [`PromotionTest::Evidence`] replaces it with
//! the evidence gate: a noninferiority e-process over the paired tasks, each
//! evaluation a round of a false evolution budget planned over the search
//! ([`evidence_outcome`]); the family-regression veto applies to both.

use serde::Serialize;
use splinter_eval::campaign::evidence::{Allocation, EProcess, EvidenceError, EvidenceGate};

use crate::compare::SetComparison;

/// The test a candidate's evidence must pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PromotionTest {
    /// The one-sided sign test over families, its p-value below `alpha`.
    SignTest {
        /// The level.
        alpha: f64,
    },
    /// The evidence gate over the paired tasks.
    Evidence(EvidenceGate),
}

/// What the evidence has to show.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionRule {
    /// The test.
    pub test: PromotionTest,
    /// The fewest tasks with a verdict in both arms; fewer is inconclusive.
    pub min_paired: usize,
}

/// What the evidence gate found for one candidate.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EvidenceOutcome {
    /// The evaluation's level and budget.
    pub allocation: Allocation,
    /// The mixture e-value at the stop.
    pub e_value: f64,
    /// The threshold, `1 / alpha`.
    pub threshold: f64,
    /// Paired tasks the e-process took (up to the crossing).
    pub observed: usize,
    /// The paired task, from 1, at which it crossed.
    pub crossed_at: Option<usize>,
    /// The exposure of every task: one once an earlier evaluation was graded
    /// on the same frozen set, zero for the first.
    pub zeta: f64,
}

/// The paired differences of a comparison, one per paired task (a win
/// `1`, a loss `-1`, a tie `0`), family by family; the caller orders them
/// at random before the e-process reads them.
#[must_use]
pub fn differences(comparison: &SetComparison) -> Vec<f64> {
    comparison
        .families
        .iter()
        .flat_map(|f| {
            std::iter::repeat_n(1.0, f.wins)
                .chain(std::iter::repeat_n(-1.0, f.losses))
                .chain(std::iter::repeat_n(0.0, f.ties))
        })
        .collect()
}

/// The evidence gate's e-process over `differences`, in the order given, at
/// the level of `allocation`, every task with exposure `zeta`.
///
/// # Errors
/// A difference outside `[-1, 1]` or an exposure outside `[0, 1]`.
pub fn evidence_outcome(
    gate: &EvidenceGate,
    allocation: &Allocation,
    differences: &[f64],
    zeta: f64,
) -> Result<EvidenceOutcome, EvidenceError> {
    let mut process = EProcess::new(gate, allocation.alpha);
    if allocation.alpha > 0.0 {
        for &difference in differences {
            if process.observe(difference, zeta)? {
                break;
            }
        }
    }
    Ok(EvidenceOutcome {
        allocation: *allocation,
        e_value: process.value(),
        threshold: process.threshold(),
        observed: process.observed(),
        crossed_at: process.crossed_at(),
        zeta,
    })
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
    /// What the evidence gate found; only under [`PromotionTest::Evidence`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<EvidenceOutcome>,
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
            evidence: None,
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
    let alpha = match rule.test {
        PromotionTest::SignTest { alpha } => alpha,
        PromotionTest::Evidence(_) => {
            return match &evidence.evidence {
                Some(found) if found.crossed_at.is_some() => decision(
                    Verdict::Promote,
                    format!(
                        "no family regressed, e-value {:.3} reached 1/{:.4} after {} paired tasks",
                        found.e_value, found.allocation.alpha, found.observed
                    ),
                ),
                Some(found) => decision(
                    Verdict::Inconclusive,
                    format!(
                        "no family regressed, but e-value {:.3} stayed below 1/{:.4} over {} paired tasks",
                        found.e_value, found.allocation.alpha, found.observed
                    ),
                ),
                None => decision(
                    Verdict::Inconclusive,
                    "the evidence gate was not computed for this candidate".to_string(),
                ),
            };
        }
    };
    if evidence.improved_families > 0 && evidence.p_value < alpha {
        return decision(
            Verdict::Promote,
            format!(
                "{} families improved, none regressed, sign test p = {:.4} < {alpha}",
                evidence.improved_families, evidence.p_value
            ),
        );
    }
    decision(
        Verdict::Inconclusive,
        format!(
            "no family regressed, but sign test p = {:.4} is not below {alpha}",
            evidence.p_value
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_eval::campaign::evidence::Budget;

    const RULE: PromotionRule = PromotionRule {
        test: PromotionTest::SignTest { alpha: 0.05 },
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
            evidence: None,
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
    fn under_the_evidence_gate_only_a_crossing_promotes_and_a_regression_still_rejects() {
        let gate = EvidenceGate::new(0.02, 0.3, 0.01).unwrap();
        let rule = PromotionRule {
            test: PromotionTest::Evidence(gate),
            min_paired: 4,
        };
        let allocation = Budget::start(&gate).allocate(&gate, 4);
        let wins = [1.0; 12];
        let ties = [0.0; 12];
        let with = |found: EvidenceOutcome, regressed: &[&str]| {
            let mut e = evidence(regressed, 0, 1.0, 12);
            e.evidence = Some(found);
            decide(&e, &rule)
        };
        let crossed = evidence_outcome(&gate, &allocation, &wins, 1.0).unwrap();
        assert!(crossed.crossed_at.is_some());
        assert_eq!(
            with(crossed.clone(), &[]).verdict,
            Verdict::Promote,
            "the p-value is not read"
        );
        assert_eq!(with(crossed, &["parsing"]).verdict, Verdict::Reject);
        let flat = evidence_outcome(&gate, &allocation, &ties, 1.0).unwrap();
        assert_eq!((flat.crossed_at, flat.observed), (None, 12));
        assert_eq!(with(flat, &[]).verdict, Verdict::Inconclusive);
        assert_eq!(
            decide(&evidence(&[], 5, 0.001, 12), &rule).verdict,
            Verdict::Inconclusive,
            "no e-process, no promotion"
        );
        let skipped = Allocation {
            alpha: 0.0,
            ..allocation
        };
        assert_eq!(
            evidence_outcome(&gate, &skipped, &wins, 1.0)
                .unwrap()
                .observed,
            0
        );
    }

    #[test]
    fn utility_is_the_candidates_share_and_absent_without_pairs() {
        assert_eq!(evidence(&[], 0, 1.0, 0).utility(), 0.0);
        assert!((evidence(&[], 0, 1.0, 10).utility() - 0.5).abs() < 1e-12);
    }
}
