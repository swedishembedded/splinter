// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The release gate's four checks, the numbers each records, and how each
//! is decided. A candidate is released only when all four pass:
//!
//! 1. **Improvement** on the new data's held-out tasks: brain's one-sided
//!    paired sign test over the tasks both models were graded on,
//!    candidate right and champion wrong against the reverse, significant
//!    at `alpha` ([`DEFAULT_ALPHA`]). Ties and unpaired tasks carry no
//!    evidence; they are excluded and counted.
//! 2. **Retention** on every earlier release's held-out tasks: per suite,
//!    the candidate's accuracy may fall at most `retention_bound`
//!    ([`DEFAULT_RETENTION_BOUND`]) below the champion's. With no earlier
//!    release there is nothing to retain, and the check passes over zero
//!    suites.
//! 3. **Anchor**: on the frozen anchor suite of general tasks, the same
//!    kind of bound, `anchor_bound` ([`DEFAULT_ANCHOR_BOUND`]).
//! 4. **Serve**: plain `brain serve --adapter` loads the candidate, reports
//!    its digest, and re-answers a sample of the held-out tasks with the
//!    same verdicts as in-process.
//!
//! A check that could not be measured - no task graded by both models, no
//! anchor suite, no brain binary - fails, and says why: an unmeasured
//! check is never a pass.

use serde::{Deserialize, Serialize};
use splinter_lab::paired::{compare, Comparison, PairedOutcome};
use splinter_policy::stats::{sign_test, SignTest};
use splinter_store::digest::Digest;

use crate::release::probe::SuiteSummary;
use crate::release::store::ReleaseId;

/// The sign test's significance level: the chance of releasing a candidate
/// no better than the champion that the gate accepts. Brain's own promote
/// gate uses the same level.
pub const DEFAULT_ALPHA: f64 = 0.05;
/// How far, as a fraction of the suite, the candidate's accuracy on an
/// earlier release's held-out tasks may fall below the champion's.
pub const DEFAULT_RETENTION_BOUND: f64 = 0.05;
/// How far, as a fraction of the suite, the candidate's accuracy on the
/// anchor suite may fall below the champion's.
pub const DEFAULT_ANCHOR_BOUND: f64 = 0.02;
/// How many held-out tasks the served candidate re-answers.
pub const DEFAULT_SERVE_SAMPLE: usize = 8;
/// How long `brain serve` may take to load the candidate and report ready.
pub const DEFAULT_SERVE_STARTUP_SECS: u64 = 600;

/// The gate's thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateConfig {
    /// The sign test's significance level.
    pub alpha: f64,
    /// The largest accuracy drop allowed on an earlier release's suite.
    pub retention_bound: f64,
    /// The largest accuracy drop allowed on the anchor suite.
    pub anchor_bound: f64,
    /// Held-out tasks re-answered by the served candidate.
    pub serve_sample: usize,
    /// Seconds `brain serve` may take to start.
    pub serve_startup_secs: u64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            retention_bound: DEFAULT_RETENTION_BOUND,
            anchor_bound: DEFAULT_ANCHOR_BOUND,
            serve_sample: DEFAULT_SERVE_SAMPLE,
            serve_startup_secs: DEFAULT_SERVE_STARTUP_SECS,
        }
    }
}

/// One check: whether it passed, what it measured, and why it failed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Check<T> {
    /// Whether it passed; an unmeasured check never does.
    pub passed: bool,
    /// What it measured; `None` when it could not be measured.
    pub measured: Option<T>,
    /// Why it failed or could not be measured.
    pub reason: Option<String>,
}

impl<T> Check<T> {
    /// A check that could not be measured, for `reason`: failed.
    #[must_use]
    pub fn unmeasured(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            measured: None,
            reason: Some(format!("not measured: {}", reason.into())),
        }
    }

    fn decided(measured: T, failure: Option<String>) -> Self {
        Self {
            passed: failure.is_none(),
            measured: Some(measured),
            reason: failure,
        }
    }
}

/// The improvement check's numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Improvement {
    /// The new data's held-out suite.
    pub suite: SuiteSummary,
    /// Candidate against champion on it.
    pub comparison: Comparison,
    /// The sign test over the paired tasks.
    pub sign_test: SignTest,
    /// The significance level it was held to.
    pub alpha: f64,
}

/// One earlier release's suite in the retention check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetainedSuite {
    /// The release whose held-out tasks these are.
    pub release: ReleaseId,
    /// The suite.
    pub suite: SuiteSummary,
    /// Candidate against champion on it.
    pub comparison: Comparison,
    /// How far the candidate's accuracy falls below the champion's;
    /// `None` when no task was graded by both.
    pub drop: Option<f64>,
    /// Whether this suite held.
    pub passed: bool,
}

/// The retention check's numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Retention {
    /// The largest drop allowed per suite.
    pub bound: f64,
    /// Each earlier release's suite, newest release first.
    pub suites: Vec<RetainedSuite>,
}

/// The anchor check's numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    /// The frozen suite's version.
    pub version: u32,
    /// Its digest.
    pub digest: Digest,
    /// The suite.
    pub suite: SuiteSummary,
    /// Candidate against champion on it.
    pub comparison: Comparison,
    /// How far the candidate's accuracy falls below the champion's.
    pub drop: Option<f64>,
    /// The largest drop allowed.
    pub bound: f64,
}

/// The serve check's numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Serve {
    /// The `brain` binary run.
    pub binary: std::path::PathBuf,
    /// The line it printed naming what it serves.
    pub startup_line: String,
    /// The adapter digest that line reports.
    pub served_digest: String,
    /// The candidate adapter's digest.
    pub expected_digest: String,
    /// Held-out tasks re-answered through the served endpoint; `0` when
    /// the digest did not match and nothing was asked.
    pub sampled: usize,
    /// Of those, the ones graded the same as in-process.
    pub agreed: usize,
    /// The tasks graded differently.
    pub disagreed: Vec<String>,
}

/// The whole gate: every check with its numbers, and the decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateReport {
    /// The thresholds the checks were held to.
    pub config: GateConfig,
    /// Improvement on the new data's held-out tasks.
    pub improvement: Check<Improvement>,
    /// Retention of every earlier release's held-out tasks.
    pub retention: Check<Retention>,
    /// No regression on the anchor suite.
    pub anchor: Check<Anchor>,
    /// Serving on plain brain.
    pub serve: Check<Serve>,
    /// Whether every check passed.
    pub passed: bool,
}

impl GateReport {
    /// The report of these four checks.
    #[must_use]
    pub fn new(
        config: GateConfig,
        improvement: Check<Improvement>,
        retention: Check<Retention>,
        anchor: Check<Anchor>,
        serve: Check<Serve>,
    ) -> Self {
        let passed = improvement.passed && retention.passed && anchor.passed && serve.passed;
        Self {
            config,
            improvement,
            retention,
            anchor,
            serve,
            passed,
        }
    }
}

/// The improvement check over `outcomes` on `suite`.
#[must_use]
pub fn improvement(
    suite: SuiteSummary,
    outcomes: &[PairedOutcome],
    alpha: f64,
) -> Check<Improvement> {
    let comparison = compare(outcomes);
    if comparison.paired == 0 {
        return Check::unmeasured(format!(
            "no held-out task of {} was graded for both models ({} task(s), {} excluded before \
             grading)",
            suite.name,
            suite.tasks,
            suite.excluded.values().sum::<usize>()
        ));
    }
    let pairs: Vec<(bool, bool)> = outcomes.iter().filter_map(PairedOutcome::paired).collect();
    let test = sign_test(&pairs);
    let failure = (test.p_value > alpha).then(|| {
        format!(
            "no significant improvement: the candidate won {} of {} discordant task(s), p = \
             {:.4} > alpha {alpha}",
            test.candidate_wins, test.discordant, test.p_value
        )
    });
    Check::decided(
        Improvement {
            suite,
            comparison,
            sign_test: test,
            alpha,
        },
        failure,
    )
}

/// The retention check over each earlier release's suite and outcomes.
#[must_use]
pub fn retention(
    suites: Vec<(ReleaseId, SuiteSummary, Vec<PairedOutcome>)>,
    bound: f64,
) -> Check<Retention> {
    let mut failures = Vec::new();
    let mut retained = Vec::new();
    for (release, suite, outcomes) in suites {
        let comparison = compare(&outcomes);
        let drop = comparison.drop();
        let passed = match drop {
            None => {
                failures.push(format!(
                    "release {release}'s suite was not measured: no task graded for both models"
                ));
                false
            }
            Some(drop) if drop > bound => {
                failures.push(format!(
                    "release {release}'s suite dropped {drop:.4} > bound {bound}"
                ));
                false
            }
            Some(_) => true,
        };
        retained.push(RetainedSuite {
            release,
            suite,
            comparison,
            drop,
            passed,
        });
    }
    let failure = (!failures.is_empty()).then(|| failures.join("; "));
    Check::decided(
        Retention {
            bound,
            suites: retained,
        },
        failure,
    )
}

/// The anchor check of suite `version`/`digest` over `outcomes`.
#[must_use]
pub fn anchor(
    version: u32,
    digest: Digest,
    suite: SuiteSummary,
    outcomes: &[PairedOutcome],
    bound: f64,
) -> Check<Anchor> {
    let comparison = compare(outcomes);
    let Some(drop) = comparison.drop() else {
        return Check::unmeasured(format!(
            "no task of anchor suite version {version} was graded for both models"
        ));
    };
    let failure = (drop > bound)
        .then(|| format!("the anchor suite (version {version}) dropped {drop:.4} > bound {bound}"));
    Check::decided(
        Anchor {
            version,
            digest,
            suite,
            comparison,
            drop: Some(drop),
            bound,
        },
        failure,
    )
}

/// The serve check over what the served candidate reported and answered.
#[must_use]
pub fn serve(measured: Serve) -> Check<Serve> {
    let failure = if measured.served_digest != measured.expected_digest {
        Some(format!(
            "brain serve reported adapter {} but the candidate is {}",
            measured.served_digest, measured.expected_digest
        ))
    } else if measured.sampled == 0 {
        return Check::unmeasured("no held-out task to re-answer through the served endpoint");
    } else if measured.agreed < measured.sampled {
        Some(format!(
            "the served candidate was graded differently from in-process on {} of {} task(s)",
            measured.sampled - measured.agreed,
            measured.sampled
        ))
    } else {
        None
    };
    Check::decided(measured, failure)
}
