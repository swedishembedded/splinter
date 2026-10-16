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
//! 1. **Improvement** on the new data's held-out tasks - the records
//!    training held out, and the variants of the tasks it trained on (the
//!    same fact in other words, which is what shows a fact was learned) -
//!    decided over both together, and reported for each alone: the held-out
//!    records measure generalisation to text the candidate was not trained
//!    on, the variants measure recall of trained facts under paraphrase, and
//!    a report never lets one pass for the other. The decision is a one-sided paired sign test ([`Significance`]) over the tasks both
//!    models were graded on,
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
//!    same answers as in-process (whitespace aside), graded the same. Both
//!    sides decode greedily, so the same weights give the same text; a
//!    verdict alone would let two different wrong answers agree.
//!
//! The four checks are the same for a candidate that works in a workspace,
//! over suites of tasks that are changes to a repository: the held-out
//! suite is the families the exam reserved, retention is every earlier
//! release's exam, the anchor suite is the frozen one, and the served
//! agent's run is compared with the in-process one by verdict
//! ([`Agreement::Verdicts`]), since two multi-step runs share no text worth
//! comparing. An attempt whose trajectory shows the solver saw what only the
//! grader holds, and an attempt the harness could not carry out, are set
//! aside and counted ([`Withheld`]); they are in no denominator. The
//! frozen anchor suite may hold no task that works in a workspace: the
//! report then says so ([`Anchor::workspace_tasks`]) and the check is decided
//! on the tasks it does hold.
//!
//! A check that could not be measured - no task graded by both models, no
//! anchor suite, no brain binary - fails, and says why: an unmeasured
//! check is never a pass.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::rng::Seed;

use crate::paired::{by_cluster, compare, Comparison, PairedOutcome};
use crate::significance::{SignTest, Significance};

/// What a suite was, as a report states it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SuiteSummary {
    /// Its name.
    pub name: String,
    /// Tasks in it.
    pub tasks: usize,
    /// Left out, by reason.
    pub excluded: BTreeMap<String, usize>,
}

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
/// How many times a workspace task is attempted per arm when a gate names no
/// other number.
pub const DEFAULT_PROBE_ATTEMPTS: u32 = 1;
/// How many follow-up turns an agent is sent after the checks it was shown
/// stay red, when nothing names another number.
pub const DEFAULT_FOLLOW_UPS: u32 = 2;
/// How many tool calls an agent may make in one attempt, follow-ups
/// included, when nothing names another number.
pub const DEFAULT_MAX_TOOL_CALLS: u32 = 60;
/// How many times one tool call may return the same answer in an attempt
/// before the attempt is stopped as going nowhere, when nothing names
/// another number.
pub const DEFAULT_MAX_REPEATED_RESULTS: u32 = 3;

/// How an agent is probed on tasks that are changes to a repository: the
/// same for both arms of a comparison and for the served candidate, so a
/// difference between them is the weights'.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgenticConfig {
    /// Attempts per task. A task's verdict is what a strict majority of its
    /// graded attempts say; with no majority it has none.
    pub attempts: u32,
    /// Follow-up turns after the checks the task shows stay red.
    pub follow_ups: u32,
    /// Tool calls in one attempt, follow-ups included; `None` sets no cap.
    pub max_tool_calls: Option<u32>,
    /// Identical answers to one call before an attempt is stopped; `None`
    /// sets no bound.
    pub max_repeated_results: Option<u32>,
}

impl Default for AgenticConfig {
    fn default() -> Self {
        Self {
            attempts: DEFAULT_PROBE_ATTEMPTS,
            follow_ups: DEFAULT_FOLLOW_UPS,
            max_tool_calls: Some(DEFAULT_MAX_TOOL_CALLS),
            max_repeated_results: Some(DEFAULT_MAX_REPEATED_RESULTS),
        }
    }
}

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
    /// How tasks that are changes to a repository are probed; unused by a
    /// gate over closed-book suites. A report written before it was kept
    /// reads the defaults.
    #[serde(default)]
    pub agentic: AgenticConfig,
    /// Whether the improvement check spends `alpha` across the releases of a
    /// lineage by the fixed schedule of [`crate::power::spent_alpha`], so that
    /// a long sequence of candidates no better than their champions
    /// releases one with probability below `alpha` and not near
    /// `1 - (1 - alpha)^n`. A report written before it existed reads `false`.
    #[serde(default)]
    pub alpha_spending: bool,
    /// The lead, in the share of independent units won, the improvement check
    /// is asked to be able to see. With it, a miss on an exam too small to
    /// see that lead is reported inconclusive and not as a failure. `None`
    /// holds the exam to no power.
    #[serde(default)]
    pub target_effect: Option<f64>,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            retention_bound: DEFAULT_RETENTION_BOUND,
            anchor_bound: DEFAULT_ANCHOR_BOUND,
            serve_sample: DEFAULT_SERVE_SAMPLE,
            serve_startup_secs: DEFAULT_SERVE_STARTUP_SECS,
            agentic: AgenticConfig::default(),
            alpha_spending: false,
            target_effect: None,
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
    /// Whether the check could not decide because the evidence was too small
    /// to see the effect asked about: neither a pass nor a failure, though
    /// nothing is released on it. See [`crate::power::Verdict`].
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub inconclusive: bool,
}

impl<T> Check<T> {
    /// A check that could not be measured, for `reason`: failed.
    #[must_use]
    pub fn unmeasured(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            measured: None,
            reason: Some(format!("not measured: {}", reason.into())),
            inconclusive: false,
        }
    }

    pub(crate) fn decided(measured: T, failure: Option<String>) -> Self {
        Self {
            passed: failure.is_none(),
            measured: Some(measured),
            reason: failure,
            inconclusive: false,
        }
    }
}

/// The attempts at workspace tasks one model's probe set aside instead of
/// grading. A count of attempts, so zero is a measurement: nothing was set
/// aside.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Withheld {
    /// Attempts whose trajectory shows the solver saw what only the grader
    /// holds (the exposure audit failed them). They are no one's win or
    /// loss, whatever their patch did.
    pub exposed: usize,
    /// Attempts the harness could not carry out or grade: the workspace
    /// could not be laid out, or the verifier could not run. They say
    /// nothing about the model.
    pub environment_failures: usize,
}

impl Withheld {
    /// This and `other` together.
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        Self {
            exposed: self.exposed + other.exposed,
            environment_failures: self.environment_failures + other.environment_failures,
        }
    }
}

/// What probing both arms on workspace tasks set aside; see [`Withheld`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgenticProbe {
    /// Attempts per task.
    pub attempts: u32,
    /// What the candidate's probe set aside.
    pub candidate: Withheld,
    /// What the champion's probe set aside.
    pub champion: Withheld,
}

/// The improvement check's numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Improvement {
    /// The new data's held-out suite: the records training held out and
    /// the variants of the tasks it trained on.
    pub suite: SuiteSummary,
    /// The variants among them: how many were measured and how many were
    /// left out, by reason; `None` when no variant was written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variants: Option<SuiteSummary>,
    /// Candidate against champion on the two together: what the decision
    /// rests on.
    pub comparison: Comparison,
    /// The sign test over the paired tasks of both.
    pub sign_test: SignTest,
    /// The significance level it was held to.
    pub alpha: f64,
    /// Generalisation: the held-out records alone, text the candidate was
    /// not trained on. `None` in a report written before the suites were
    /// reported apart, or when the decision was not measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generalisation: Option<PartResult>,
    /// Recall under paraphrase: the variants of trained tasks alone. `None`
    /// when no variant was measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall: Option<PartResult>,
    /// The tasks' own evidence over the whole suite the decision rests on;
    /// see [`TaskLevel`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_level: Option<TaskLevel>,
    /// What the probes of workspace tasks in the suite set aside; `None`
    /// when the suite holds none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agentic: Option<AgenticProbe>,
    /// The power and the sequential level the decision was held to; `None`
    /// when the gate ran without a planned effect or a generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power: Option<crate::power::PowerAssessment>,
}

/// What the tasks themselves say, beside the family-level sign test the gate
/// decides on: that test rests on as many units as there are discordant
/// families, a dozen or so, so a decision by it moves with one family. This
/// pairs by task, tests the tasks only one model got right exactly, and gives
/// the difference in the share right with an interval that resamples families.
/// It informs the decision and does not make it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskLevel {
    /// The one-sided exact sign test over the discordant tasks.
    pub p_value: f64,
    /// The candidate's share right minus the champion's.
    pub difference: f64,
    /// The lower end of the family-clustered bootstrap interval.
    pub low: f64,
    /// Its upper end.
    pub high: f64,
}

/// One part of the improvement suite compared on its own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartResult {
    /// The part's tasks.
    pub suite: SuiteSummary,
    /// Candidate against champion on it.
    pub comparison: Comparison,
    /// The sign test over its paired tasks, a family counted once.
    pub sign_test: SignTest,
    /// The tasks' own evidence; see [`TaskLevel`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_level: Option<TaskLevel>,
}

/// One part of the improvement suite, whose paired `outcomes` are compared
/// on their own by the rule the decision uses; `None` when no task of it was
/// graded for both models.
#[must_use]
pub fn part_result(
    suite: SuiteSummary,
    outcomes: &[PairedOutcome],
    significance: &dyn Significance,
) -> Option<PartResult> {
    let comparison = compare(outcomes);
    (comparison.paired > 0).then(|| PartResult {
        suite,
        sign_test: significance.sign_test(&by_cluster(outcomes)),
        task_level: None,
        comparison,
    })
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
    /// The tasks (by address) the champion got right and the candidate got
    /// wrong: what the candidate forgot.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lost: Vec<String>,
    /// What the probes of workspace tasks in the suite set aside; `None`
    /// when the suite holds none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agentic: Option<AgenticProbe>,
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
    /// The drop's bootstrap interval over the paired items: what a drop on a
    /// suite this size is to be read by. `None` when too few items were
    /// paired to resample, or in a report written before it was kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drop_interval: Option<DropInterval>,
    /// How many paired items a suite would need for four chances in five of
    /// seeing a drop of two points at the share of items this one found
    /// discordant (one-sided, at the level the sign test is held to): what a
    /// suite this size can and cannot detect. `None` when no item was
    /// discordant or none paired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items_to_see_two_points: Option<usize>,
    /// For a candidate that works in a workspace: how many of the frozen
    /// version's tasks are changes to a repository. `Some(0)` is an absence,
    /// not a score: the version holds no such task, and the check was decided
    /// on the tasks it does hold. `None` for a candidate judged closed-book.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_tasks: Option<usize>,
    /// What the probes of the version's workspace tasks set aside; `None`
    /// when it holds none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agentic: Option<AgenticProbe>,
}

/// A bootstrap interval of the champion's accuracy minus the candidate's,
/// resampling the paired items.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DropInterval {
    /// The share of resampled drops it holds.
    pub level: f64,
    /// Its lower end.
    pub low: f64,
    /// Its upper end.
    pub high: f64,
}

/// What the serve check compares between a run in-process and the same run
/// served.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Agreement {
    /// The final answers: the same text up to whitespace and the noise of two
    /// processes decoding one model, or the same meaning, and the same
    /// verdict. For a task answered in one reply.
    #[default]
    Answers,
    /// The verdicts alone. For a task an agent works on over many steps,
    /// whose runs share no text worth comparing: what matters is whether the
    /// work the served agent left is graded as the in-process agent's was.
    Verdicts,
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
    /// Of those, the ones that agreed in what [`Serve::agreement`] compares:
    /// for answers, the same answer, whitespace aside, graded the same.
    pub agreed: usize,
    /// The tasks answered differently, with both answers.
    pub disagreed: Vec<Disagreement>,
    /// What was compared. A report written before it was kept compared
    /// answers.
    #[serde(default)]
    pub agreement: Agreement,
}

/// One held-out task the served candidate answered differently from the
/// candidate in-process.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Disagreement {
    /// The task.
    pub task: String,
    /// The answer in-process; `None` when there was none.
    pub in_process: Option<String>,
    /// The answer through `brain serve`; `None` when there was none.
    pub served: Option<String>,
    /// How the answer in-process was graded.
    pub in_process_verdict: Option<bool>,
    /// How the served answer was graded.
    pub served_verdict: Option<bool>,
}

/// The system prompt each arm was asked a suite under: the numbers of two
/// arms are comparable only when this is known.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SuitePrompts {
    /// The suite's name.
    pub suite: String,
    /// What the candidate was asked under: `default`, or `persona: ` and the
    /// start of the prompt it was trained under.
    pub candidate: String,
    /// The same for the champion.
    pub champion: String,
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
    /// The prompt each arm was asked each suite under; empty in a report
    /// written before it was recorded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompts: Vec<SuitePrompts>,
    /// The seed the bootstrap intervals of the report (the tasks' own
    /// evidence, the anchor's drop) were resampled with: the same outcomes and
    /// seed give the same intervals. `None` in a report written before it
    /// was kept, whose intervals were resampled from seed 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_seed: Option<Seed>,
}

impl GateReport {
    /// This report stating the seed its intervals were resampled with.
    #[must_use]
    pub fn with_interval_seed(mut self, seed: Seed) -> Self {
        self.interval_seed = Some(seed);
        self
    }

    /// This report stating which prompt each arm was asked each suite under.
    #[must_use]
    pub fn with_prompts(mut self, prompts: Vec<SuitePrompts>) -> Self {
        self.prompts = prompts;
        self
    }

    /// Why the gate did not pass, one entry per check that failed or was not
    /// measured, each naming its check; empty when the gate passed.
    #[must_use]
    pub fn blocked_by(&self) -> Vec<String> {
        let mut why = Vec::new();
        let mut add = |name: &str, passed: bool, reason: &Option<String>| {
            if !passed {
                why.push(format!(
                    "{name}: {}",
                    reason.as_deref().unwrap_or("it did not pass")
                ));
            }
        };
        add(
            "improvement",
            self.improvement.passed,
            &self.improvement.reason,
        );
        add("retention", self.retention.passed, &self.retention.reason);
        add("anchor", self.anchor.passed, &self.anchor.reason);
        add("serve", self.serve.passed, &self.serve.reason);
        why
    }

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
            prompts: Vec::new(),
            interval_seed: None,
        }
    }
}

/// The improvement check over `outcomes` on `suite`.
#[must_use]
pub fn improvement(
    suite: SuiteSummary,
    variants: Option<SuiteSummary>,
    outcomes: &[PairedOutcome],
    alpha: f64,
    significance: &dyn Significance,
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
    let pairs = by_cluster(outcomes);
    let test = significance.sign_test(&pairs);
    let failure = (test.p_value > alpha).then(|| {
        format!(
            "no significant improvement: the candidate won {} of {} discordant task(s) (tasks \
             about one family of sources count once), p = {:.4} > alpha {alpha}",
            test.candidate_wins, test.discordant, test.p_value
        )
    });
    Check::decided(
        Improvement {
            suite,
            variants,
            comparison,
            sign_test: test,
            alpha,
            generalisation: None,
            recall: None,
            task_level: None,
            agentic: None,
            power: None,
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
        let lost = outcomes
            .iter()
            .filter(|o| o.paired() == Some((false, true)))
            .map(|o| o.item.clone())
            .collect();
        retained.push(RetainedSuite {
            release,
            suite,
            comparison,
            drop,
            passed,
            lost,
            agentic: None,
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
            drop_interval: None,
            items_to_see_two_points: None,
            workspace_tasks: None,
            agentic: None,
        },
        failure,
    )
}

/// The share of the re-answered tasks, in percent, that must come back
/// alike for the serve check to pass (for a task an agent works on, graded
/// the same: [`Agreement::Verdicts`]). Two processes decoding one model sum
/// in a different order, so now and then one long greedy continuation parts
/// ways in its last tenth; an adapter bound wrongly or not applied changes
/// the answers to most tasks.
pub const SERVE_AGREEMENT_PERCENT: usize = 75;

/// The serve check over what the served candidate reported and answered.
#[must_use]
pub fn serve(measured: Serve) -> Check<Serve> {
    let differently = match measured.agreement {
        Agreement::Answers => "answered differently from",
        Agreement::Verdicts => "was graded differently from",
    };
    let failure = if measured.served_digest != measured.expected_digest {
        Some(format!(
            "brain serve reported adapter {} but the candidate is {}",
            measured.served_digest, measured.expected_digest
        ))
    } else if measured.sampled == 0 {
        return Check::unmeasured("no held-out task to re-answer through the served endpoint");
    } else if measured.agreed * 100 < measured.sampled * SERVE_AGREEMENT_PERCENT {
        Some(format!(
            "the served candidate {differently} in-process on {} of {} task(s); \
             at least {SERVE_AGREEMENT_PERCENT}% must agree",
            measured.sampled - measured.agreed,
            measured.sampled
        ))
    } else {
        None
    };
    Check::decided(measured, failure)
}

#[cfg(test)]
mod serve_tests {
    use super::*;

    fn measured(sampled: usize, agreed: usize) -> Serve {
        Serve {
            binary: "brain".into(),
            startup_line: String::new(),
            served_digest: "sha256:a".into(),
            expected_digest: "sha256:a".into(),
            sampled,
            agreed,
            disagreed: Vec::new(),
            agreement: Agreement::default(),
        }
    }

    #[test]
    fn a_rare_disagreement_passes_and_a_majority_of_them_does_not() {
        assert!(serve(measured(8, 7)).passed);
        assert!(serve(measured(8, 6)).passed, "exactly three quarters");
        assert!(!serve(measured(8, 5)).passed);
        assert!(!serve(measured(6, 0)).passed);
    }

    #[test]
    fn a_served_agent_is_held_to_its_verdicts_and_the_report_says_so() {
        let by_verdict = Serve {
            agreement: Agreement::Verdicts,
            ..measured(8, 5)
        };
        let check = serve(by_verdict);
        assert!(!check.passed);
        let why = check.reason.unwrap();
        assert!(
            why.contains("graded differently") && why.contains("3 of 8"),
            "{why}"
        );
        assert!(
            serve(Serve {
                agreement: Agreement::Verdicts,
                ..measured(8, 6)
            })
            .passed
        );
    }
}
