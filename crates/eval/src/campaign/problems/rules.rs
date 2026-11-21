// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for choosing the next training
// experiment from the evidence already paid for, for its clients. If your team
// needs expertise in planning model-improvement campaigns you can procure our
// services by sending an email to info@swedishembedded.com.

//! The rules of the problem state: which candidate to try next, what a
//! verdict does to a belief, and what verdict a measurement is.
//!
//! **Priority.** `b / (N + 1)`, `b` the candidate's belief and `N` the
//! experiments run against its problem: a problem that has had many tries
//! yields to one that has had few, unless its candidates are much more
//! believed.
//!
//! **Belief.** Moved by the verdict, never by a raw difference: a training
//! measurement is expensive and noisy, and its interval is often wider than
//! its effect. Supported adds `belief_up` (at most 1), falsified takes
//! `belief_down` (at least 0); inconclusive and invalid leave it where it was.
//! An inconclusive result is not a falsification: it counts as an attempt
//! (`N` grows, so the priority falls) and nothing more.
//!
//! **Verdict of a measured effect.** Against the problem's target `d`:
//! supported when the interval is above 0, the difference at least `d` and
//! the paired test's p below `alpha`; falsified when the interval's upper
//! bound is below `d` (a gain of the claimed size is ruled out, even if a
//! smaller one is real); invalid with no finite effect; otherwise
//! inconclusive.
//!
//! **Verdict of the promotion gate.** Supported when it promoted (the
//! champion moved: the new best); invalid when it refused a verdict as
//! evidence; falsified when the dev interval is below `d`, or the retention
//! interval lies wholly past the margin (harm shown, not merely
//! non-inferiority unshown); otherwise inconclusive.

use serde::{Deserialize, Serialize};

use super::{CandidateStatus, ProblemError};
use super::{Effect, Parameters, Phase, ProblemState, ProblemStatus, Verdict};
use crate::promotion::{Promotion, Reason, RETENTION_MARGIN};

/// What a supported verdict adds to a belief by default: a support is weaker
/// evidence than its interval suggests under repeated gating, so three take
/// a 0.5 prior to 0.95.
pub const DEFAULT_BELIEF_UP: f64 = 0.15;

/// What a falsified verdict takes from a belief by default: three take a 0.5
/// prior to 0, so no single measurement, whose baseline can move by the
/// effect size between time slots, retires an idea.
pub const DEFAULT_BELIEF_DOWN: f64 = 0.20;

/// The level every interval is at and every test decides at by default.
pub const DEFAULT_ALPHA: f64 = 0.05;

/// The priority of a candidate believed `belief` against a problem with
/// `attempts` experiments: `belief / (attempts + 1)`.
#[must_use]
pub fn priority(belief: f64, attempts: u32) -> f64 {
    belief / (f64::from(attempts) + 1.0)
}

/// `belief` after `verdict` under `parameters`.
#[must_use]
pub fn updated_belief(belief: f64, verdict: Verdict, parameters: &Parameters) -> f64 {
    match verdict {
        Verdict::Supported => (belief + parameters.belief_up).min(1.0),
        Verdict::Falsified => (belief - parameters.belief_down).max(0.0),
        Verdict::Inconclusive | Verdict::Invalid => belief,
    }
}

/// The verdict of a measured `effect` against the `target`, its interval at
/// `1 - alpha`.
#[must_use]
pub fn classify_effect(effect: Option<&Effect>, target: f64, alpha: f64) -> Verdict {
    let Some(e) = effect else {
        return Verdict::Invalid;
    };
    if !(e.difference.is_finite() && e.low.is_finite() && e.high.is_finite()) || e.low > e.high {
        return Verdict::Invalid;
    }
    let tested = e.p.is_some_and(|p| p < alpha);
    if e.low > 0.0 && e.difference >= target && tested {
        Verdict::Supported
    } else if e.high < target {
        Verdict::Falsified
    } else {
        Verdict::Inconclusive
    }
}

/// The verdict of the promotion gate's decision against the `target`, with
/// the dev effect it rests on when the dev verdict verified.
#[must_use]
pub fn classify_promotion(promotion: &Promotion, target: f64) -> (Verdict, Option<Effect>) {
    let effect = promotion.dev.as_ref().and_then(|dev| {
        let (low, high) = dev.interval?;
        Some(Effect {
            difference: dev.gain?,
            low,
            high,
            p: None,
        })
    });
    let verdict = if promotion.promoted {
        Verdict::Supported
    } else if promotion
        .reasons
        .iter()
        .any(|r| matches!(r, Reason::Verdict { .. }))
    {
        Verdict::Invalid
    } else if effect.as_ref().is_some_and(|e| e.high < target)
        || promotion
            .retention
            .as_ref()
            .and_then(|r| r.interval)
            .is_some_and(|(_, high)| high < -RETENTION_MARGIN)
    {
        Verdict::Falsified
    } else {
        Verdict::Inconclusive
    };
    (verdict, effect)
}

/// A candidate proposed as the next hypothesis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    /// Its problem.
    pub problem: String,
    /// The candidate.
    pub candidate: String,
    /// Its priority.
    pub priority: f64,
    /// Its belief.
    pub belief: f64,
    /// Experiments run against its problem.
    pub attempts: u32,
    /// The one variable its arm changes.
    pub variable: String,
    /// The value at it, when the configuration can express it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
}

impl ProblemState {
    /// Every pending candidate of `phase` with a belief above 0, of every
    /// open problem, highest priority first. Ties go to the higher belief,
    /// then the problem with fewer attempts, then the problem's and the
    /// candidate's ids: a total order, the same on every replay.
    #[must_use]
    pub fn ranked(&self, phase: Phase) -> Vec<Proposal> {
        let mut proposals: Vec<Proposal> = self
            .problems
            .iter()
            .filter(|p| p.status == ProblemStatus::Open)
            .flat_map(|p| {
                p.candidates
                    .iter()
                    .filter(move |c| {
                        c.status == CandidateStatus::Pending && c.phase == phase && c.belief > 0.0
                    })
                    .map(move |c| Proposal {
                        problem: p.id.clone(),
                        candidate: c.id.clone(),
                        priority: priority(c.belief, p.attempts),
                        belief: c.belief,
                        attempts: p.attempts,
                        variable: c.variable.clone(),
                        value: c.value.clone(),
                    })
            })
            .collect();
        proposals.sort_by(|a, b| {
            b.priority
                .total_cmp(&a.priority)
                .then(b.belief.total_cmp(&a.belief))
                .then(a.attempts.cmp(&b.attempts))
                .then(a.problem.cmp(&b.problem))
                .then(a.candidate.cmp(&b.candidate))
        });
        proposals
    }

    /// The next hypothesis of `phase`: the first of [`Self::ranked`].
    #[must_use]
    pub fn next(&self, phase: Phase) -> Option<Proposal> {
        self.ranked(phase).into_iter().next()
    }

    /// The verdict of `effect` on `candidate`, against its problem's target
    /// at the state's level.
    ///
    /// # Errors
    /// An unknown candidate, or a metric with no target in force.
    pub fn classify(
        &self,
        candidate: &str,
        effect: Option<&Effect>,
    ) -> Result<Verdict, ProblemError> {
        let (problem, _) = self
            .candidate(candidate)
            .ok_or_else(|| ProblemError::UnknownCandidate(candidate.to_string()))?;
        Ok(classify_effect(
            effect,
            self.target(problem)?,
            self.parameters.alpha,
        ))
    }
}
