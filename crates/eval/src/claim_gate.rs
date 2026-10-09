// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that decide a model update
// on declared rules and the numbers behind them, for its clients. If your
// team needs expertise in evaluation-gated fine-tuning, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The release gate for a model updated from what a person taught it.
//!
//! A handful of facts a night cannot reach a significance test, so the gate
//! is a declared rule on counts:
//!
//! * a claim is **answered** when every one of its stopping paraphrases (the
//!   questions about it no training record contains) passes, graded greedily
//!   by the claim's own verifiers; a claim with none measured is not
//!   answered;
//! * **improvement** passes when the candidate answers a claim the champion
//!   did not, or when there is none to gain (the champion answered every
//!   claim);
//! * **retention** passes when every claim the champion answered is still
//!   answered: one regression fails the gate, with no tolerance;
//! * the **anchor** suite, when one is frozen, may not lose an item count;
//!   when none is frozen the check is absent;
//! * **serve** is the existing check that plain brain serves the candidate.

use serde::{Deserialize, Serialize};

use crate::gate::{Check, Serve};

/// Passes out of measured, for one model on one set of paraphrases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tally {
    /// Paraphrases answered right.
    pub passed: usize,
    /// Paraphrases a verifier decided.
    pub measured: usize,
}

impl Tally {
    /// Whether every paraphrase passed, and at least one was measured.
    #[must_use]
    pub fn answered(&self) -> bool {
        self.measured > 0 && self.passed == self.measured
    }
}

/// One live claim's stopping paraphrases, graded for both models.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimTally {
    /// The claim.
    pub claim: String,
    /// The candidate on them.
    pub candidate: Tally,
    /// The champion on them; `None` when it was not graded.
    pub champion: Option<Tally>,
}

/// The anchor suite's items right, for both models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorTally {
    /// The candidate.
    pub candidate: Tally,
    /// The champion.
    pub champion: Tally,
}

/// The gate's decision and every number it rests on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClaimGate {
    /// Every live claim's counts.
    pub claims: Vec<ClaimTally>,
    /// Claims the candidate answers.
    pub answered: usize,
    /// Claims the candidate answers and the champion did not.
    pub gained: Vec<String>,
    /// Claims the candidate does not answer.
    pub unanswered: Vec<String>,
    /// Claims the champion answered and the candidate does not.
    pub regressed: Vec<String>,
    /// A claim newly answered.
    pub improvement: Check<usize>,
    /// No claim regressed.
    pub retention: Check<usize>,
    /// The anchor suite, when one is frozen.
    pub anchor: Option<Check<AnchorTally>>,
    /// Serving on plain brain.
    pub serve: Check<Serve>,
    /// Whether every check passed.
    pub passed: bool,
}

fn decided<T>(measured: T, failure: Option<String>) -> Check<T> {
    Check {
        passed: failure.is_none(),
        measured: Some(measured),
        reason: failure,
    }
}

/// Decides the gate over `claims`, the anchor counts if a suite is frozen,
/// and the serve check.
#[must_use]
pub fn decide(
    claims: Vec<ClaimTally>,
    anchor: Option<AnchorTally>,
    serve: Check<Serve>,
) -> ClaimGate {
    let champion_answered = |c: &ClaimTally| c.champion.is_some_and(|t| t.answered());
    let gained: Vec<String> = claims
        .iter()
        .filter(|c| c.candidate.answered() && !champion_answered(c))
        .map(|c| c.claim.clone())
        .collect();
    let unanswered: Vec<String> = claims
        .iter()
        .filter(|c| !c.candidate.answered())
        .map(|c| c.claim.clone())
        .collect();
    let regressed: Vec<String> = claims
        .iter()
        .filter(|c| champion_answered(c) && !c.candidate.answered())
        .map(|c| c.claim.clone())
        .collect();
    let answered = claims.len() - unanswered.len();
    // A night with nothing left to gain (every claim was already answered, as
    // when it only took a forgotten claim out) has no improvement to show.
    let anything_to_gain = claims.iter().any(|c| !champion_answered(c));
    let improvement = decided(
        gained.len(),
        (gained.is_empty() && anything_to_gain)
            .then(|| "no claim is newly answered on its stopping paraphrases".to_string()),
    );
    let retention = decided(
        regressed.len(),
        (!regressed.is_empty()).then(|| {
            format!(
                "claims answered before and not now: {}",
                regressed.join(", ")
            )
        }),
    );
    let anchor = anchor.map(|a| {
        let failure = (a.candidate.passed < a.champion.passed).then(|| {
            format!(
                "the anchor suite lost items: {} right of {}, the champion had {}",
                a.candidate.passed, a.candidate.measured, a.champion.passed
            )
        });
        decided(a, failure)
    });
    let passed = improvement.passed
        && retention.passed
        && anchor.as_ref().is_none_or(|a| a.passed)
        && serve.passed;
    ClaimGate {
        claims,
        answered,
        gained,
        unanswered,
        regressed,
        improvement,
        retention,
        anchor,
        serve,
        passed,
    }
}
