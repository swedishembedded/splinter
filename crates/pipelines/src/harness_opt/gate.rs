// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The promotion rule for a candidate harness: the one promotion gate
//! ([`splinter_eval::promotion`]) on the paired comparison of its verified
//! attempts against the baseline's, and the refusals a search adds to it.
//!
//! No examiner draws a sealed slice for a harness, so the gate's explicit
//! rule for that case decides ([`promotion::decide_unexamined`]): the paired
//! gain, its two-sided interval at the rule's level, resampling whole families, above zero,
//! and retention, the lower bound of its one-sided 95% interval above
//! `-RETENTION_MARGIN`. A harness change keeps the model's weights, so its one
//! frozen task set is both its development split and its retention suite.
//! The decision record says it is not exam-verified ([`NOT_EXAM_VERIFIED`]).
//!
//! Beside the rule, and never instead of it, a candidate is refused when a
//! family of tasks got worse (a family the baseline won more often than the
//! candidate is a regression, whatever the totals say), when too few tasks
//! have a verdict in both arms, and, under the evidence gate, when its
//! noninferiority e-process did not cross: each evaluation is a round of one
//! false evolution budget planned over the search ([`evidence_outcome`]),
//! which bounds the false promotions of a search over many candidates.
//! Anything short of a promotion is rejected (a regression, or retention
//! not shown) or inconclusive, and the dense score the search follows is
//! only guidance: it never promotes.

use serde::Serialize;
use splinter_eval::campaign::evidence::{Allocation, EProcess, EvidenceError, EvidenceGate};
use splinter_eval::promotion::{self, Measured, Measurement, Pair, Promotion, Reason};

use crate::compare::SetComparison;
use crate::promotion::measurement;

/// Why a harness promotion is not exam-verified: the record says so beside
/// every decision.
pub const NOT_EXAM_VERIFIED: &str = "a harness change keeps the model's weights: its one frozen \
     task set is both its development split and its retention suite, and no examiner drew a \
     sealed slice for it";

/// What the evidence has to show beside the promotion rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionRule {
    /// The evidence gate over the paired tasks, when the search holds its
    /// candidates to a false evolution budget; `None`: the promotion rule
    /// alone.
    pub evidence: Option<EvidenceGate>,
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
    /// The one-sided sign test's p-value over families: reported, it
    /// decides nothing.
    pub p_value: f64,
    /// The paired gain in the share of tasks right, with its intervals
    /// resampling whole families: what the promotion rule decides on.
    pub measurement: Measurement,
    /// What the evidence gate found; only under [`PromotionRule::evidence`].
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
            measurement: measurement(
                &comparison
                    .families
                    .iter()
                    .map(|f| (f.candidate_right as f64 - f.baseline_right as f64, f.paired))
                    .collect::<Vec<_>>(),
                comparison.interval_seed,
            ),
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
    /// Promoted by the promotion rule, and refused by nothing beside it.
    Promote,
    /// A family got worse, or retention was not shown.
    Reject,
    /// Nothing got worse, but no gain was shown.
    Inconclusive,
}

/// What a search refuses a candidate for beside the promotion rule.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "refusal", rename_all = "snake_case")]
pub enum Refusal {
    /// Families the baseline won more of than the candidate.
    RegressedFamilies {
        /// Their names.
        families: Vec<String>,
    },
    /// Too few tasks with a verdict in both arms.
    TooFewPaired {
        /// Tasks with one.
        paired: usize,
        /// The fewest asked for.
        needed: usize,
    },
    /// The evidence gate's e-process did not cross its threshold.
    EvidenceShort {
        /// The e-value at the stop.
        e_value: f64,
        /// The threshold, `1 / alpha`.
        threshold: f64,
        /// Paired tasks it took.
        observed: usize,
    },
    /// The evidence gate was asked for and not computed.
    EvidenceMissing,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RegressedFamilies { families } => write!(f, "regressed: {}", families.join(", ")),
            Self::TooFewPaired { paired, needed } => write!(
                f,
                "{paired} tasks with a verdict in both arms, at least {needed} needed"
            ),
            Self::EvidenceShort {
                e_value,
                threshold,
                observed,
            } => write!(
                f,
                "e-value {e_value:.3} stayed below {threshold:.3} over {observed} paired tasks"
            ),
            Self::EvidenceMissing => {
                write!(f, "the evidence gate was not computed for this candidate")
            }
        }
    }
}

/// A verdict, why, and what it was decided by.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Decision {
    /// The verdict.
    pub verdict: Verdict,
    /// Why, in a sentence.
    pub reason: String,
    /// The promotion rule's decision, labelled not exam-verified.
    pub promotion: Promotion,
    /// What the search refused the candidate for beside it.
    pub refusals: Vec<Refusal>,
}

/// Decides on `evidence` about `pair` (the candidate harness against the
/// baseline) under `rule`.
#[must_use]
pub fn decide(evidence: &Evidence, rule: &PromotionRule, pair: &Pair) -> Decision {
    let promotion = promotion::decide_unexamined(
        pair,
        &Measured {
            dev: evidence.measurement.clone(),
            retention: evidence.measurement.clone(),
        },
        NOT_EXAM_VERIFIED,
    );
    let mut refusals = Vec::new();
    if !evidence.regressed_families.is_empty() {
        refusals.push(Refusal::RegressedFamilies {
            families: evidence.regressed_families.clone(),
        });
    }
    if evidence.paired < rule.min_paired {
        refusals.push(Refusal::TooFewPaired {
            paired: evidence.paired,
            needed: rule.min_paired,
        });
    }
    if rule.evidence.is_some() {
        match &evidence.evidence {
            Some(found) if found.crossed_at.is_some() => {}
            Some(found) => refusals.push(Refusal::EvidenceShort {
                e_value: found.e_value,
                threshold: found.threshold,
                observed: found.observed,
            }),
            None => refusals.push(Refusal::EvidenceMissing),
        }
    }
    let regressed = refusals
        .iter()
        .any(|r| matches!(r, Refusal::RegressedFamilies { .. }))
        || promotion
            .reasons
            .iter()
            .any(|r| matches!(r, Reason::Inferior { .. }));
    let verdict = if promotion.promoted && refusals.is_empty() {
        Verdict::Promote
    } else if regressed {
        Verdict::Reject
    } else {
        Verdict::Inconclusive
    };
    let reason = match verdict {
        Verdict::Promote => {
            let m = &evidence.measurement;
            let (low, high) = m.interval.unwrap_or_default();
            format!(
                "no family regressed, the gain {:+.3} has its two-sided interval [{low:+.3}, {high:+.3}] \
                 above 0 over {} paired tasks (not exam-verified)",
                m.gain.unwrap_or_default(),
                m.items
            )
        }
        Verdict::Reject | Verdict::Inconclusive => refusals
            .iter()
            .map(ToString::to_string)
            .chain(promotion.reasons.iter().map(ToString::to_string))
            .collect::<Vec<_>>()
            .join("; "),
    };
    Decision {
        verdict,
        reason,
        promotion,
        refusals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_eval::campaign::evidence::Budget;
    use splinter_eval::promotion::Rule;

    const RULE: PromotionRule = PromotionRule {
        evidence: None,
        min_paired: 4,
    };

    fn pair() -> Pair {
        Pair {
            candidate: Some("candidate".into()),
            predecessor: Some("baseline".into()),
        }
    }

    /// Evidence over `paired` tasks whose gain has the interval `interval`
    /// (and its low end as the one-sided bound).
    fn evidence(regressed: &[&str], interval: Option<(f64, f64)>, paired: usize) -> Evidence {
        Evidence {
            paired,
            candidate_right: paired / 2,
            baseline_right: paired / 2,
            regressed_families: regressed.iter().map(ToString::to_string).collect(),
            improved_families: 0,
            p_value: 1.0,
            measurement: Measurement {
                items: paired as u64,
                families: paired as u64,
                gain: interval.map(|(low, high)| (low + high) / 2.0),
                interval,
                one_sided_lower: interval.map(|(low, _)| low),
            },
            evidence: None,
        }
    }

    #[test]
    fn a_gain_whose_interval_is_above_zero_with_no_regression_is_promoted_not_exam_verified() {
        let d = decide(&evidence(&[], Some((0.1, 0.4)), 10), &RULE, &pair());
        assert_eq!(d.verdict, Verdict::Promote, "{}", d.reason);
        assert!(d.promotion.promoted && !d.promotion.exam_verified);
        assert_eq!(
            d.promotion.rule,
            Rule::NotExamVerified {
                why: NOT_EXAM_VERIFIED.into()
            }
        );
        assert!(d.reason.contains("not exam-verified"), "{}", d.reason);
    }

    #[test]
    fn one_regressed_family_rejects_whatever_the_gain_says() {
        let d = decide(
            &evidence(&["parsing"], Some((0.2, 0.6)), 20),
            &RULE,
            &pair(),
        );
        assert_eq!(d.verdict, Verdict::Reject);
        assert!(d.promotion.promoted, "the rule alone would promote");
        assert!(d.reason.contains("parsing"));
    }

    #[test]
    fn a_gain_not_shown_is_inconclusive_and_a_loss_is_rejected_never_a_promotion() {
        let flat = decide(&evidence(&[], Some((-0.02, 0.2)), 10), &RULE, &pair());
        assert_eq!(flat.verdict, Verdict::Inconclusive, "{}", flat.reason);
        assert!(flat.reason.contains("no dev gain"), "{}", flat.reason);
        let few = decide(&evidence(&[], Some((0.1, 0.4)), 3), &RULE, &pair());
        assert_eq!(few.verdict, Verdict::Inconclusive);
        let one_family = decide(&evidence(&[], None, 10), &RULE, &pair());
        assert_eq!(one_family.verdict, Verdict::Inconclusive);
        let worse = decide(&evidence(&[], Some((-0.4, -0.1)), 10), &RULE, &pair());
        assert_eq!(
            worse.verdict,
            Verdict::Reject,
            "retention not shown rejects: {}",
            worse.reason
        );
    }

    #[test]
    fn under_the_evidence_gate_a_crossing_is_also_needed_and_a_regression_still_rejects() {
        let gate = EvidenceGate::new(0.02, 0.3, 0.01).unwrap();
        let rule = PromotionRule {
            evidence: Some(gate),
            min_paired: 4,
        };
        let allocation = Budget::start(&gate).allocate(&gate, 4);
        let wins = [1.0; 12];
        let ties = [0.0; 12];
        let with = |found: EvidenceOutcome, regressed: &[&str]| {
            let mut e = evidence(regressed, Some((0.2, 0.6)), 12);
            e.evidence = Some(found);
            decide(&e, &rule, &pair())
        };
        let crossed = evidence_outcome(&gate, &allocation, &wins, 1.0).unwrap();
        assert!(crossed.crossed_at.is_some());
        assert_eq!(with(crossed.clone(), &[]).verdict, Verdict::Promote);
        assert_eq!(with(crossed, &["parsing"]).verdict, Verdict::Reject);
        let flat = evidence_outcome(&gate, &allocation, &ties, 1.0).unwrap();
        assert_eq!((flat.crossed_at, flat.observed), (None, 12));
        assert_eq!(with(flat, &[]).verdict, Verdict::Inconclusive);
        assert_eq!(
            decide(&evidence(&[], Some((0.2, 0.6)), 12), &rule, &pair()).verdict,
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
        assert_eq!(evidence(&[], None, 0).utility(), 0.0);
        assert!((evidence(&[], None, 10).utility() - 0.5).abs() < 1e-12);
    }
}
