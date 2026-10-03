// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements closed-book evaluation suites graded by
// verifiers the model under test cannot reach, for its clients. If your
// team needs expertise in model evaluation, you can procure our services
// by sending an email to info@swedishembedded.com.

//! `eval`: one model graded closed-book on the gate's suites, and the
//! anchor suite frozen or shown.
//!
//! The model is a candidate (by id or prefix: its base with its adapter)
//! or a model reference, decoding greedily as the gate's arms do. Which tasks a suite holds depends on it:
//!
//! | Suite | A candidate | `policy:<alias>` |
//! |---|---|---|
//! | `held-out` | its new datasets' held-out tasks | the alias's release's |
//! | `retention` | every release in its champion's lineage, per release | every release before the alias's |
//! | `anchor` | the anchor suite in force | the same |
//! | a file | the anchor-format tasks in it | the same |
//!
//! A `local:` or `remote:` model has no release, so only `anchor` and a
//! file apply to it. `--freeze FILE` (with `--suite anchor`) makes the
//! tasks in FILE the anchor suite's next version first; with no model,
//! the anchor suite in force is shown.

use std::path::PathBuf;
use std::str::FromStr;

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_data::DatasetId;
use splinter_eval::paired::accuracy;
use sven_sdk::CancelToken;

use crate::context::Context;
use crate::error::CampaignError;
use crate::model_ref::ModelRef;
use crate::release::probe::{self, Suite, SuiteSummary};
use crate::release::{anchor, arm, ReleaseId, StoredRelease};
use crate::runs::{record, Recorded};
use crate::train::load_candidate;

/// Which suite `eval` grades on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SuiteChoice {
    /// The new data's held-out tasks.
    HeldOut,
    /// Every earlier release's held-out tasks, per release.
    Retention,
    /// The anchor suite in force.
    Anchor,
    /// Anchor-format tasks in a file.
    File(PathBuf),
}

impl FromStr for SuiteChoice {
    type Err = CampaignError;

    fn from_str(text: &str) -> Result<Self, CampaignError> {
        Ok(match text {
            "held-out" => Self::HeldOut,
            "retention" => Self::Retention,
            "anchor" => Self::Anchor,
            "" => {
                return Err(CampaignError::Refused(
                    "a suite is held-out, retention, anchor or a file".into(),
                ))
            }
            file => Self::File(PathBuf::from(file)),
        })
    }
}

/// One `eval`.
#[derive(Clone, Debug, Serialize)]
pub struct EvalRequest {
    /// A candidate id (or prefix) or a model reference; `None` only to
    /// show or freeze the anchor suite.
    pub model: Option<String>,
    /// The suite.
    pub suite: SuiteChoice,
    /// Freeze this file as the anchor suite's next version first.
    pub freeze: Option<PathBuf>,
}

/// The anchor suite in force, as `eval` shows it.
#[derive(Clone, Debug, Serialize)]
pub struct AnchorShown {
    /// Its version.
    pub version: u32,
    /// Its digest.
    pub digest: Digest,
    /// Tasks in it.
    pub tasks: usize,
}

/// One suite's score.
#[derive(Clone, Debug, Serialize)]
pub struct SuiteScore {
    /// The suite.
    pub suite: SuiteSummary,
    /// The release whose held-out tasks it is, for held-out and retention.
    pub release: Option<ReleaseId>,
    /// Tasks a verifier decided.
    pub graded: usize,
    /// Of those, the fraction right; `None` when none was decided.
    pub accuracy: Option<f64>,
}

/// What `eval` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Evaluated {
    /// The model graded, as it was named; `None` when none was.
    pub model: Option<String>,
    /// The reference it was graded through.
    pub reference: Option<String>,
    /// The anchor suite in force, when the anchor suite was asked for.
    pub anchor: Option<AnchorShown>,
    /// Each suite's score.
    pub scores: Vec<SuiteScore>,
}

/// What the model under evaluation is, with the releases its suites come
/// from.
struct Subject {
    reference: ModelRef,
    held_out: Option<(Option<ReleaseId>, Vec<DatasetId>)>,
    earlier: Vec<StoredRelease>,
}

fn subject(ctx: &Context, named: &str) -> Result<Subject, CampaignError> {
    let store = ctx.releases();
    let lineage = |parent: Option<&ReleaseId>| match parent {
        Some(id) => store.lineage(id),
        None => Ok(Vec::new()),
    };
    if let Ok(reference) = named.parse::<ModelRef>() {
        let ModelRef::Policy(alias) = &reference else {
            return Ok(Subject {
                reference,
                held_out: None,
                earlier: Vec::new(),
            });
        };
        let pin = ctx.policy_pin(alias)?;
        let release = pin.map(|p| store.get(&p.release)).transpose()?;
        let earlier = lineage(release.as_ref().and_then(|r| r.manifest.parent.as_ref()))?;
        return Ok(Subject {
            held_out: release.map(|r| (Some(r.id), r.manifest.datasets)),
            reference,
            earlier,
        });
    }
    let candidate = load_candidate(ctx, named)?;
    Ok(Subject {
        reference: arm(ctx.config(), Some(&candidate.adapter)),
        held_out: Some((None, candidate.datasets)),
        earlier: lineage(candidate.parent.as_ref())?,
    })
}

/// Runs `request`; see the module documentation.
pub fn eval(
    ctx: &Context,
    request: &EvalRequest,
    cancel: &CancelToken,
) -> Result<Evaluated, CampaignError> {
    if request.freeze.is_some() && request.suite != SuiteChoice::Anchor {
        return Err(CampaignError::Refused(
            "--freeze makes an anchor suite; give it with --suite anchor".into(),
        ));
    }
    let frozen = match &request.freeze {
        Some(file) => Some(anchor::freeze(ctx, file)?),
        None if request.suite == SuiteChoice::Anchor => anchor::current(ctx)?,
        None => None,
    };
    let anchor_shown = frozen.as_ref().map(|f| AnchorShown {
        version: f.suite.version,
        digest: f.digest.clone(),
        tasks: f.suite.tasks.len(),
    });
    let Some(named) = &request.model else {
        if request.suite == SuiteChoice::Anchor {
            return Ok(Evaluated {
                model: None,
                reference: None,
                anchor: anchor_shown,
                scores: Vec::new(),
            });
        }
        return Err(CampaignError::Refused(
            "name a candidate or a model reference to evaluate".into(),
        ));
    };
    let subject = subject(ctx, named)?;
    let suites: Vec<(Option<ReleaseId>, Suite)> = match &request.suite {
        SuiteChoice::HeldOut => {
            let Some((release, datasets)) = &subject.held_out else {
                return Err(no_release(named, "held-out"));
            };
            vec![(release.clone(), probe::held_out(ctx, "held-out", datasets)?)]
        }
        SuiteChoice::Retention => {
            if subject.held_out.is_none() {
                return Err(no_release(named, "retention"));
            }
            subject
                .earlier
                .iter()
                .map(|release| {
                    let name = format!("retention {}", release.id);
                    probe::held_out(ctx, name, &release.manifest.datasets)
                        .map(|suite| (Some(release.id.clone()), suite))
                })
                .collect::<Result<_, _>>()?
        }
        SuiteChoice::Anchor => {
            let Some(frozen) = &frozen else {
                return Err(CampaignError::Refused(
                    "no anchor suite is frozen: `splinter eval --suite anchor --freeze FILE`"
                        .into(),
                ));
            };
            vec![(None, frozen.probe_suite())]
        }
        SuiteChoice::File(file) => {
            vec![(
                None,
                Suite::of_tasks(file.display().to_string(), anchor::read_tasks(file)?),
            )]
        }
    };
    let model = probe::greedy(ctx, &subject.reference)?;
    let mut scores = Vec::with_capacity(suites.len());
    for (release, suite) in &suites {
        let outcomes: Vec<Option<bool>> = probe::grade(ctx, &model, suite, cancel)?
            .into_iter()
            .map(|probe| probe.verdict)
            .collect();
        let (accuracy, graded) = accuracy(&outcomes);
        scores.push(SuiteScore {
            suite: suite.summary(),
            release: release.clone(),
            graded,
            accuracy,
        });
    }
    Ok(Evaluated {
        model: Some(named.clone()),
        reference: Some(subject.reference.to_string()),
        anchor: anchor_shown,
        scores,
    })
}

/// What the `eval` command reports: the anchor suite shown, or a recorded
/// run that froze a suite or graded a model.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum EvalReport {
    /// Only shown: nothing was written or run.
    Shown(Evaluated),
    /// A recorded run and its report.
    Ran(Box<Recorded<Evaluated>>),
}

/// The `eval` command: showing the anchor suite is a read; freezing one
/// or grading a model is a recorded run.
pub fn evaluate(ctx: &Context, request: &EvalRequest) -> Result<EvalReport, CampaignError> {
    if request.model.is_none() && request.freeze.is_none() {
        return Ok(EvalReport::Shown(eval(ctx, request, &CancelToken::new())?));
    }
    let recorded = record(ctx, "eval", request, |run| {
        eval(ctx, request, &run.cancel_token())
    })?;
    Ok(EvalReport::Ran(Box::new(recorded)))
}

fn no_release(named: &str, suite: &str) -> CampaignError {
    CampaignError::Refused(format!(
        "{named} has no release or candidate data, so there is no {suite} suite for it; \
         name a candidate or a policy alias that points at a release"
    ))
}
