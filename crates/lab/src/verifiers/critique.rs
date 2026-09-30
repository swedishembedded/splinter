// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade feedback by what it
// achieved, for its clients. If your team needs expertise in reward
// modelling or verifier design, you can procure our services by sending an
// email to info@swedishembedded.com.

//! A critique verified by outcome: it is right when the retry that
//! received it passes where the attempt it critiques failed.
//!
//! A critique's text cannot be run or matched against a reference, but
//! what it was for can be checked: a retry of the same task, told the
//! critique, is graded by the task's own verifiers. [`critique_verdict`]
//! turns the two decisions into the critique's verdict:
//!
//! * the critiqued attempt must be decided fail; otherwise there was
//!   nothing to critique and the verdict abstains;
//! * a retry that did not finish (stopped by a deadline, a budget, a
//!   cancel) or reached no decision tests nothing, and the verdict
//!   abstains;
//! * a retry decided pass makes the critique pass; decided fail, fail.
//!
//! The strength is [`Strength::Consistency`], capped at the weaker of the
//! two decisions it rests on. Two attempts at one task compared - one
//! without the critique, one with it - is agreement-type evidence about the
//! critique, not a check of its text, so it ranks above any judged opinion
//! of the critique and below a formal or executable check whatever decided
//! the attempts; and a verdict is never stronger than what it rests on.
//! The comparison attributes the change of outcome to the critique; a retry
//! that passes by chance passes the critique with it, which repeated
//! retries of one critique would expose.
//!
//! [`preferred`] is the rule for recording the revision as preferred over
//! the attempt it retries: it passed, the other failed, and at the same
//! strength - the condition under which two decisions are comparable.

use serde_json::json;
use splinter_store::annotation::{
    Annotation, AnnotationBody, Decision, Outcome, Producer, Strength,
};
use splinter_store::experience::ExperienceId;

/// The producer name of a critique's outcome verdict.
pub const PRODUCER: &str = "splinter-lab/critique-outcome";

/// The version of the rule; bumped whenever it changes.
pub const VERSION: &str = "1";

/// How the retry a critique was tested by ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryOutcome {
    /// Whether the retry's run finished (concluded successfully) rather
    /// than being stopped by a bound.
    pub finished: bool,
    /// The decision its verdicts add up to, when there is one.
    pub decision: Option<Decision>,
}

/// A critique's verdict by outcome, before it is attached to the critique.
#[derive(Clone, Debug, PartialEq)]
pub struct CritiqueVerdict {
    /// Pass, fail, or no judgement.
    pub outcome: Outcome,
    /// The strength it is recorded at.
    pub strength: Strength,
    /// The two decisions it rests on, and the experiences they are of.
    pub evidence: serde_json::Value,
}

impl CritiqueVerdict {
    /// This verdict as an annotation of `critique`.
    #[must_use]
    pub fn annotation(&self, critique: &ExperienceId) -> Annotation {
        Annotation {
            experience: critique.clone(),
            producer: Producer {
                name: PRODUCER.into(),
                version: VERSION.into(),
            },
            body: AnnotationBody::Verdict {
                outcome: self.outcome,
                strength: self.strength,
                evidence: self.evidence.clone(),
            },
        }
    }
}

/// The verdict on a critique of `critiqued` (decided `critiqued_decision`)
/// that `retry` received and ended as `outcome`, under the module's rule.
#[must_use]
pub fn critique_verdict(
    critiqued: &ExperienceId,
    critiqued_decision: Option<Decision>,
    retry: &ExperienceId,
    outcome: RetryOutcome,
) -> CritiqueVerdict {
    let evidence = json!({
        "critiqued": critiqued,
        "critiqued_decision": decision_json(critiqued_decision),
        "retry": retry,
        "retry_finished": outcome.finished,
        "retry_decision": decision_json(outcome.decision),
    });
    match (critiqued_decision, outcome) {
        (
            Some(failed),
            RetryOutcome {
                finished: true,
                decision: Some(retried),
            },
        ) if !failed.passed => CritiqueVerdict {
            outcome: if retried.passed {
                Outcome::Pass
            } else {
                Outcome::Fail
            },
            strength: Strength::Consistency
                .min(failed.strength)
                .min(retried.strength),
            evidence,
        },
        _ => CritiqueVerdict {
            outcome: Outcome::Abstain,
            strength: Strength::Consistency,
            evidence: json!({
                "abstained": abstain_reason(critiqued_decision, outcome),
                "detail": evidence,
            }),
        },
    }
}

/// Whether a revision decided `revision` is preferred over the failed
/// attempt it retries, decided `failed`: it passed, the other failed, at
/// the same strength.
#[must_use]
pub fn preferred(revision: Option<Decision>, failed: Option<Decision>) -> bool {
    matches!(
        (revision, failed),
        (Some(r), Some(f)) if r.passed && !f.passed && r.strength == f.strength
    )
}

fn abstain_reason(critiqued: Option<Decision>, outcome: RetryOutcome) -> &'static str {
    if critiqued.is_none_or(|d| d.passed) {
        "the critiqued attempt was not decided fail"
    } else if !outcome.finished {
        "the retry did not finish"
    } else {
        "the retry was not decided"
    }
}

fn decision_json(decision: Option<Decision>) -> serde_json::Value {
    decision.map_or(
        serde_json::Value::Null,
        |d| json!({ "passed": d.passed, "strength": d.strength }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_store::digest::Digest;

    fn id(tag: &str) -> ExperienceId {
        ExperienceId(Digest::of(tag.as_bytes()))
    }

    fn decided(passed: bool, strength: Strength) -> Option<Decision> {
        Some(Decision { passed, strength })
    }

    fn verdict(failed: Option<Decision>, outcome: RetryOutcome) -> (Outcome, Strength) {
        let verdict = critique_verdict(&id("f"), failed, &id("r"), outcome);
        (verdict.outcome, verdict.strength)
    }

    #[test]
    fn a_critique_is_graded_by_the_retry_it_led_to() {
        let failed = decided(false, Strength::Executable);
        let finished = |d| RetryOutcome {
            finished: true,
            decision: d,
        };
        assert_eq!(
            verdict(failed, finished(decided(true, Strength::Executable))),
            (Outcome::Pass, Strength::Consistency)
        );
        assert_eq!(
            verdict(failed, finished(decided(false, Strength::Executable))),
            (Outcome::Fail, Strength::Consistency)
        );
        assert_eq!(
            verdict(failed, finished(decided(true, Strength::Judged))),
            (Outcome::Pass, Strength::Judged),
            "never stronger than the decisions it rests on"
        );
        assert_eq!(verdict(failed, finished(None)).0, Outcome::Abstain);
        let stopped = RetryOutcome {
            finished: false,
            decision: decided(false, Strength::Executable),
        };
        assert_eq!(verdict(failed, stopped).0, Outcome::Abstain);
        assert_eq!(
            verdict(
                decided(true, Strength::Executable),
                finished(decided(true, Strength::Executable))
            )
            .0,
            Outcome::Abstain,
            "a pass was never critiqued"
        );
    }

    #[test]
    fn a_revision_is_preferred_only_over_a_comparable_failure() {
        let pass = decided(true, Strength::Executable);
        assert!(preferred(pass, decided(false, Strength::Executable)));
        assert!(!preferred(pass, decided(false, Strength::Judged)));
        assert!(!preferred(
            decided(false, Strength::Executable),
            decided(false, Strength::Executable)
        ));
        assert!(!preferred(None, decided(false, Strength::Executable)));
    }
}
