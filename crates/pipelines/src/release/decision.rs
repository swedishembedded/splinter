// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Whether a candidate gains and retains, decided by the one promotion gate
//! ([`splinter_eval::promotion`]): on the examiner's signed verdicts when the
//! caller holds them ([`promotion::decide`]), else by the explicit rule for
//! a decision no sealed slice applies to ([`promotion::decide_unexamined`])
//! over the gate's own paired outcomes, recorded as not exam-verified.
//!
//! The gate's improvement and retention checks carry that decision: each
//! fails with the promotion's reasons about its rule. What they measured
//! beside it (the sign test, its power, each earlier release's drop) is
//! reported and decides nothing. The checks that are not about gain or
//! retention - every assigned task counted for each arm, the anchor suite's
//! floor, the served candidate, the canaries - stay refusals of their own,
//! recorded in the same report.

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};
use serde_json::Value;
use splinter_eval::gate::{Check, Improvement, Retention};
use splinter_eval::promotion::{
    self, Measured, Pair, Promotion, PublicKey, Reason, Role, Submitted,
};

/// Why a release decided by the gate's own suites is not exam-verified:
/// the record says so beside every such decision.
pub const NOT_EXAM_VERIFIED: &str = "the release gate grades the candidate on Splinter's own \
     held-out tasks, retention suites and anchor suite, whose exam the learner made itself: no \
     examiner's sealed slice applies";

/// The examiner's three signed verdicts on a candidate against the champion
/// it would replace, and the public key they are verified with.
#[derive(Clone)]
pub struct ExaminerVerdicts {
    /// The examiner's public key.
    pub key: PublicKey,
    /// The verdict over the development split.
    pub dev: Value,
    /// The verdict over the retention suite.
    pub retention: Value,
    /// The verdict over the sealed slice drawn for this decision.
    pub sealed: Value,
}

impl std::fmt::Debug for ExaminerVerdicts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExaminerVerdicts")
            .field("key", &self.key.key_id())
            .finish_non_exhaustive()
    }
}

/// A request is recorded with the key's id and the verdicts as handed in.
impl Serialize for ExaminerVerdicts {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut out = serializer.serialize_struct("ExaminerVerdicts", 4)?;
        out.serialize_field("key", &self.key.key_id())?;
        out.serialize_field("dev", &self.dev)?;
        out.serialize_field("retention", &self.retention)?;
        out.serialize_field("sealed", &self.sealed)?;
        out.end()
    }
}

/// The promotion decision on `pair`: on `examined` when the caller holds the
/// examiner's verdicts, else on the gate's own `measured` outcomes, not
/// exam-verified.
#[must_use]
pub(super) fn decide(
    pair: &Pair,
    examined: Option<&ExaminerVerdicts>,
    measured: &Measured,
) -> Promotion {
    match examined {
        Some(verdicts) => promotion::decide(
            &verdicts.key,
            pair,
            Submitted {
                dev: &verdicts.dev,
                retention: &verdicts.retention,
                sealed: &verdicts.sealed,
            },
        ),
        None => promotion::decide_unexamined(pair, measured, NOT_EXAM_VERIFIED),
    }
}

/// The promotion's reasons about gain (the dev split, and the sealed slice
/// it is exam-verified on) or, with `dev` false, about retention.
fn reasons_about(promotion: &Promotion, dev: bool) -> Vec<String> {
    promotion
        .reasons
        .iter()
        .filter(|reason| {
            let about_dev = match reason {
                Reason::NoDevGain { .. }
                | Reason::TooFewFamilies { .. }
                | Reason::DevUnmeasured => true,
                Reason::Inferior { .. } | Reason::RetentionUnmeasured => false,
                Reason::Verdict { role, .. } => *role != Role::Retention,
            };
            about_dev == dev
        })
        .map(ToString::to_string)
        .collect()
}

/// `check` failed with `own` (its own refusal) and `rule` (the promotion's
/// reasons about it); passed when both are empty.
fn decided<T>(mut check: Check<T>, own: Option<String>, rule: Vec<String>) -> Check<T> {
    let why: Vec<String> = own.into_iter().chain(rule).collect();
    check.passed = why.is_empty();
    check.reason = (!why.is_empty()).then(|| why.join("; "));
    check
}

/// The improvement check decided by `promotion`'s gain rule, and by its own
/// count over every assigned task: a candidate that solved fewer than the
/// champion is refused, so an attempt set aside cannot remove a loss. A gain
/// the exam was too small to show stays labelled inconclusive.
#[must_use]
pub(super) fn improvement(check: Check<Improvement>, promotion: &Promotion) -> Check<Improvement> {
    let rule = reasons_about(promotion, true);
    let Some(measured) = &check.measured else {
        let own = check.reason.clone();
        return decided(check, own, rule);
    };
    let verified = measured.verified;
    let own = (verified.candidate < verified.baseline).then(|| {
        format!(
            "the candidate solved fewer tasks than the champion ({} against {} of {}): a task \
             the candidate was not graded on counts as a task it did not solve, so an attempt \
             set aside as exposed or as an environment failure cannot remove a loss",
            verified.candidate, verified.baseline, verified.tasks
        )
    });
    // The power assessment's label: the exam could not have shown the lead
    // asked about. It describes a miss, so it holds only for one.
    let underpowered = check.inconclusive && own.is_none() && !rule.is_empty();
    let label = underpowered.then(|| check.reason.clone()).flatten();
    let mut check = decided(check, own.or(label), rule);
    check.inconclusive = underpowered;
    check
}

/// The retention check decided by `promotion`'s retention rule. Each earlier
/// release's drop against the bound is reported, and names what a suite
/// lost for the curriculum; it decides nothing.
#[must_use]
pub(super) fn retention(check: Check<Retention>, promotion: &Promotion) -> Check<Retention> {
    let own = check
        .measured
        .is_none()
        .then(|| check.reason.clone())
        .flatten();
    let rule = reasons_about(promotion, false);
    let mut check = decided(check, own, rule);
    check.inconclusive = false;
    check
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_eval::gate::{SuiteSummary, Verified};
    use splinter_eval::paired::compare;
    use splinter_eval::promotion::{Measurement, Rule};
    use splinter_eval::significance::SignTest;

    fn pair() -> Pair {
        Pair {
            candidate: Some("cand".into()),
            predecessor: None,
        }
    }

    fn measured(dev: (f64, f64), retention_lower: f64) -> Measured {
        Measured {
            dev: Measurement {
                items: 10,
                families: 10,
                gain: Some((dev.0 + dev.1) / 2.0),
                interval: Some(dev),
                one_sided_lower: Some(dev.0),
            },
            retention: Measurement {
                items: 10,
                families: 10,
                gain: Some(0.0),
                interval: Some((retention_lower - 0.01, 0.01)),
                one_sided_lower: Some(retention_lower),
            },
        }
    }

    fn improvement_check(verified: Verified) -> Check<Improvement> {
        let suite = SuiteSummary {
            name: "held-out".into(),
            tasks: verified.tasks,
            excluded: Default::default(),
        };
        Check {
            passed: false,
            measured: Some(Improvement {
                suite,
                verified,
                variants: None,
                comparison: compare(&[]),
                sign_test: SignTest {
                    candidate_wins: 0,
                    discordant: 0,
                    p_value: 1.0,
                },
                alpha: 0.05,
                generalisation: None,
                recall: None,
                task_level: None,
                agentic: None,
                power: None,
                hidden_strength: None,
            }),
            reason: Some("no significant improvement: p = 1 > alpha 0.05".into()),
            inconclusive: false,
        }
    }

    fn retention_check() -> Check<Retention> {
        Check {
            passed: false,
            measured: Some(Retention {
                bound: 0.05,
                suites: Vec::new(),
            }),
            reason: Some("release r's suite dropped 0.1000 > bound 0.05".into()),
            inconclusive: false,
        }
    }

    const EVEN: Verified = Verified {
        candidate: 5,
        baseline: 5,
        tasks: 10,
    };

    #[test]
    fn without_examiner_verdicts_the_decision_is_labelled_not_exam_verified() {
        let promotion = decide(&pair(), None, &measured((0.1, 0.3), -0.02));
        assert!(promotion.promoted, "{promotion:?}");
        assert!(!promotion.exam_verified);
        assert_eq!(
            promotion.rule,
            Rule::NotExamVerified {
                why: NOT_EXAM_VERIFIED.into()
            }
        );
    }

    #[test]
    fn the_checks_carry_the_promotion_and_not_the_sign_test_or_the_per_suite_bound() {
        let promoted = decide(&pair(), None, &measured((0.1, 0.3), -0.02));
        let improved = improvement(improvement_check(EVEN), &promoted);
        assert!(improved.passed && improved.reason.is_none(), "{improved:?}");
        let retained = retention(retention_check(), &promoted);
        assert!(retained.passed, "{retained:?}");

        let flat = decide(&pair(), None, &measured((-0.1, 0.2), -0.02));
        let improved = improvement(improvement_check(EVEN), &flat);
        assert!(!improved.passed, "a dev interval reaching 0 rejects");
        assert!(improved.reason.unwrap().contains("no dev gain"));

        let inferior = decide(&pair(), None, &measured((0.1, 0.3), -0.08));
        let retained = retention(retention_check(), &inferior);
        assert!(
            !retained.passed,
            "a retention bound below the margin rejects"
        );
        assert!(retained.reason.unwrap().contains("retention not shown"));
        assert!(improvement(improvement_check(EVEN), &inferior).passed);
    }

    #[test]
    fn a_candidate_that_solved_fewer_assigned_tasks_is_refused_whatever_the_gain() {
        let promoted = decide(&pair(), None, &measured((0.1, 0.3), -0.02));
        let fewer = Verified {
            candidate: 4,
            ..EVEN
        };
        let check = improvement(improvement_check(fewer), &promoted);
        assert!(!check.passed);
        assert!(check.reason.unwrap().contains("fewer tasks"));
    }

    #[test]
    fn verdicts_that_do_not_verify_reject_and_the_record_says_it_was_examined() {
        // The neutral point's encoding: a point of the curve, and the key of
        // no examiner.
        let mut neutral = [0; 32];
        neutral[0] = 1;
        let key = PublicKey::from_bytes(neutral).unwrap();
        let verdicts = ExaminerVerdicts {
            key,
            dev: Value::Null,
            retention: Value::Null,
            sealed: Value::Null,
        };
        let promotion = decide(&pair(), Some(&verdicts), &measured((0.1, 0.3), -0.02));
        assert!(!promotion.promoted);
        assert!(promotion.exam_verified);
        assert_eq!(promotion.rule, Rule::Examined);
        let check = improvement(improvement_check(EVEN), &promotion);
        assert!(
            !check.passed,
            "the dev and sealed verdicts' rejections are the gain's"
        );
        assert!(!retention(retention_check(), &promotion).passed);
    }
}
