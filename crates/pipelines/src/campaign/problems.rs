// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for choosing the next training
// experiment from the evidence already paid for, for its clients. If your team
// needs expertise in planning model-improvement campaigns you can procure our
// services by sending an email to info@swedishembedded.com.

//! A campaign's problem state on disk, and what it says about the campaign's
//! files: the rules themselves are [`splinter_eval::campaign::problems`].
//!
//! **Where it lives.** `<state>/campaigns/<name>/problems/`, a stage ledger
//! of its own beside the campaign's, so it can be read and revised while
//! `campaign run` holds the campaign's ledger. Each event is one unit,
//! `event/<key>`, its key the digest of the event: an event recorded again is
//! found and nothing is recorded, and one the state refuses (a second verdict
//! from one experiment) is refused before anything is written.
//!
//! **A campaign file as a configuration.** The single-variable and firewall
//! checks compare two campaign files' specifications as JSON
//! ([`configuration`]), without what is not a variable of an experiment:
//! its name, its generations and budget (how far it may go), its seed (the
//! draw) and its starting champion (the current best, by construction).
//!
//! **What the firewall protects.** An engineering change may touch only the
//! state's engineering paths, and no configured list may reach a path under
//! [`PROTECTED`]: the base, the rollout, curation and training settings, the
//! practice material and profile, and the examiner.
//!
//! **A hypothesis as an arm.** [`campaign_next`] proposes the top candidate
//! and, when it declares a value, the arm: the campaign file continuing from
//! the current champion under the same seed, with that one change, checked
//! before it is returned.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_eval::campaign::problems::{
    self as rules, classify_promotion, firewall, single_variable, under, Document, Effect, Event,
    Evidence, Phase, ProblemState, Proposal, Recorded, Verdict,
};
use splinter_eval::promotion::Promotion;
use splinter_orchestrator::context::Context;
use splinter_store::stage_ledger::{
    KeyParts, OnCorrupt, StageLedger, StageWork, Standing, LEDGER_FILE,
};

use super::account::{begun, campaign_status};
use super::controller::{campaign_dir, record, unit};
use super::spec::CampaignSpec;
use crate::arm_compare::ArmComparison;
use crate::served::ServedError;

/// The problem state's directory under a campaign's.
pub const PROBLEMS_DIR: &str = "problems";

/// The configuration paths no engineering list may reach: the model, the
/// algorithm, the data and the measurement.
pub const PROTECTED: [&str; 10] = [
    "base",
    "rollout",
    "curate",
    "train",
    "practice.dir",
    "practice.profile",
    "practice.sven_build",
    "practice.rung",
    "practice.families",
    "examine",
];

/// The fields of a campaign file that are not a variable of an experiment.
const NOT_VARIABLES: [&str; 5] = ["name", "generations", "budget", "seed", "champion"];

/// The sections of a campaign file too wide to be one variable.
const SECTIONS: [&str; 6] = [
    "rollout", "curate", "train", "practice", "examine", "budget",
];

/// A campaign's problem state, with its identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProblemsReport {
    /// The campaign.
    pub campaign: String,
    /// The state's digest.
    pub digest: String,
    /// The events it folds.
    pub events: usize,
    /// Whether this call recorded an event (false: it was recorded before,
    /// or nothing was asked to be recorded).
    pub recorded: bool,
    /// The state.
    pub state: ProblemState,
    /// The hypotheses in the order `campaign next` proposes them.
    pub ranked: Vec<Proposal>,
}

/// The next hypothesis, and its arm when the campaign file can express it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NextReport {
    /// The campaign.
    pub campaign: String,
    /// The proposal; `None` when no candidate is left.
    pub proposal: Option<Proposal>,
    /// The arm's campaign file; `None` without a proposal or a value.
    pub arm: Option<CampaignSpec>,
    /// The paths the arm changes.
    pub changed: Vec<String>,
}

/// What a checked arm changes.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ArmCheck {
    /// The variable it was held to; `None` for an engineering check alone.
    pub variable: Option<String>,
    /// Whether the firewall was applied.
    pub engineering: bool,
    /// The paths it changes.
    pub changed: Vec<String>,
}

/// Where a verdict comes from.
#[derive(Clone, Debug)]
pub enum VerdictSource {
    /// Stated by a coordinator, with its reason.
    Stated(Verdict),
    /// A contrast of an `experiences compare-arms` report.
    Compare {
        /// The report.
        report: ArmComparison,
        /// The arm whose contrast against the report's reference is read.
        arm: String,
    },
    /// The promotion gate of an arm's campaign generation.
    Gate {
        /// The arm's campaign file.
        arm: Box<CampaignSpec>,
        /// Its generation.
        generation: u32,
    },
}

/// A verdict to record.
#[derive(Clone, Debug)]
pub struct VerdictRequest {
    /// The candidate.
    pub candidate: String,
    /// The experiment.
    pub experiment: String,
    /// Where the verdict comes from.
    pub source: VerdictSource,
    /// Where it is recorded.
    pub evidence: Vec<Evidence>,
    /// Why; required for a stated verdict, written from the measurement
    /// otherwise.
    pub reason: Option<String>,
}

fn config(why: impl Into<String>) -> ServedError {
    ServedError::Config(why.into())
}

fn unseeded(campaign: &str) -> ServedError {
    config(format!(
        "campaign {campaign} has no problem state: add a document first (campaign problems add)"
    ))
}

/// The campaign file `spec` as a configuration: its specification as JSON
/// without the fields that are not a variable of an experiment.
///
/// # Errors
/// A specification that does not serialise.
pub fn configuration(spec: &CampaignSpec) -> Result<Value, ServedError> {
    let mut value =
        serde_json::to_value(spec).map_err(|e| config(format!("campaign specification: {e}")))?;
    if let Value::Object(map) = &mut value {
        for field in NOT_VARIABLES {
            map.remove(field);
        }
    }
    Ok(value)
}

/// The problem-state document in `text`: JSON when `path` ends in `.json`,
/// TOML otherwise.
///
/// # Errors
/// Not a document of this shape.
pub fn parse_document(text: &str, path: &Path) -> Result<Document, ServedError> {
    let parsed = if path.extension().is_some_and(|e| e == "json") {
        serde_json::from_str(text).map_err(|e| e.to_string())
    } else {
        toml::from_str(text).map_err(|e| e.to_string())
    };
    parsed.map_err(|e| config(format!("{}: {e}", path.display())))
}

/// Refuses what the rules cannot know is wrong with a campaign: a variable
/// that names a whole section, and an engineering list that reaches what
/// the firewall protects.
fn check_document(document: &Document) -> Result<(), ServedError> {
    if let Some(parameters) = &document.parameters {
        for path in &parameters.engineering {
            if let Some(protected) = PROTECTED.iter().find(|p| under(path, p) || under(p, path)) {
                return Err(config(format!(
                    "parameters.engineering names {path:?}, which reaches {protected:?}: \
                     an engineering change never touches the model, the algorithm, the data \
                     or the measurement"
                )));
            }
        }
    }
    for candidate in document.problems.iter().flat_map(|p| &p.candidates) {
        if SECTIONS.contains(&candidate.variable.as_str()) {
            return Err(config(format!(
                "candidate {:?} declares the whole section {:?} as its variable: name the one setting it changes",
                candidate.id, candidate.variable
            )));
        }
    }
    Ok(())
}

/// The problem state's ledger, held while this value lives.
struct Problems {
    campaign: String,
    ledger: StageLedger,
}

impl Problems {
    /// The ledger of `spec`'s problem state; with `create` false, a state
    /// never begun is refused without creating anything.
    fn open(ctx: &Context, spec: &CampaignSpec, create: bool) -> Result<Self, ServedError> {
        spec.validate()?;
        let dir = campaign_dir(ctx, &spec.name).join(PROBLEMS_DIR);
        if !create && !dir.join(LEDGER_FILE).is_file() {
            return Err(unseeded(&spec.name));
        }
        Ok(Self {
            campaign: spec.name.clone(),
            ledger: StageLedger::open(&dir)?,
        })
    }

    fn events(&self) -> Result<Vec<Event>, ServedError> {
        self.ledger
            .commits()?
            .iter()
            .map(|c| {
                record::<Recorded>(c)?
                    .into_event()
                    .map_err(|e| config(e.to_string()))
            })
            .collect()
    }

    fn report(&self, recorded: bool) -> Result<ProblemsReport, ServedError> {
        let events = self.events()?;
        let state = ProblemState::replay(&events)
            .map_err(|e| config(e.to_string()))?
            .ok_or_else(|| unseeded(&self.campaign))?;
        Ok(ProblemsReport {
            campaign: self.campaign.clone(),
            digest: state.digest().to_string(),
            events: events.len(),
            recorded,
            ranked: state.ranked(Phase::Hypothesis),
            state,
        })
    }

    /// Records `event` unless it was recorded before; refuses it, writing
    /// nothing, when the state would refuse it.
    fn record(&self, event: Event) -> Result<ProblemsReport, ServedError> {
        let summary = serde_json::to_value(Recorded::new(event.clone()))
            .map_err(|e| config(format!("a problem-state event: {e}")))?;
        let key = KeyParts::new("problem-event").config(&summary).key();
        let name = format!("event/{key}");
        if let Standing::Committed(_) = self.ledger.standing(&name, &key)? {
            return self.report(false);
        }
        let state = ProblemState::replay(&self.events()?).map_err(|e| config(e.to_string()))?;
        ProblemState::fold(state, &event).map_err(|e| config(e.to_string()))?;
        self.ledger.run(&name, &key, OnCorrupt::Refuse, |_| {
            Ok::<_, String>(StageWork::new(summary))
        })?;
        self.report(true)
    }
}

/// The problem state of the campaign `spec` names.
///
/// # Errors
/// No state yet, another process holding it, or its ledger.
pub fn problems_show(ctx: &Context, spec: &CampaignSpec) -> Result<ProblemsReport, ServedError> {
    Problems::open(ctx, spec, false)?.report(false)
}

/// The engineering paths of the campaign's problem state; the default ones
/// when it has none yet.
///
/// # Errors
/// A state that cannot be read, or another process holding it.
pub fn engineering_paths(ctx: &Context, spec: &CampaignSpec) -> Result<Vec<String>, ServedError> {
    let ledger = campaign_dir(ctx, &spec.name)
        .join(PROBLEMS_DIR)
        .join(LEDGER_FILE);
    if ledger.is_file() {
        Ok(problems_show(ctx, spec)?.state.parameters.engineering)
    } else {
        Ok(rules::Parameters::default().engineering)
    }
}

/// Seeds the campaign's problem state with `document`, or extends it.
///
/// # Errors
/// A document the rules or the campaign checks refuse, or the ledger.
pub fn problems_add(
    ctx: &Context,
    spec: &CampaignSpec,
    document: Document,
) -> Result<ProblemsReport, ServedError> {
    check_document(&document)?;
    Problems::open(ctx, spec, true)?.record(Event::Add { document })
}

/// Records a coordinator's revision: a [`Event::Belief`] or an
/// [`Event::Status`].
///
/// # Errors
/// Another kind of event, one the state refuses, or the ledger.
pub fn problems_revise(
    ctx: &Context,
    spec: &CampaignSpec,
    revision: Event,
) -> Result<ProblemsReport, ServedError> {
    if !matches!(revision, Event::Belief { .. } | Event::Status { .. }) {
        return Err(config("a revision sets a belief or a status"));
    }
    Problems::open(ctx, spec, false)?.record(revision)
}

/// Records one experiment's verdict on one candidate.
///
/// # Errors
/// A stated verdict without a reason; a report at another level or without
/// the arm; an arm that is not a single-variable change from `spec` in the
/// candidate's variable, or whose generation has no gate record; one the
/// state refuses; or the ledger.
pub fn problems_update(
    ctx: &Context,
    spec: &CampaignSpec,
    request: VerdictRequest,
) -> Result<ProblemsReport, ServedError> {
    let problems = Problems::open(ctx, spec, false)?;
    let state = problems.report(false)?.state;
    let (problem, candidate) = state
        .candidate(&request.candidate)
        .ok_or_else(|| config(format!("no candidate {:?}", request.candidate)))?;
    let target = state.target(problem).map_err(|e| config(e.to_string()))?;
    let (verdict, effect, measured) = match &request.source {
        VerdictSource::Stated(verdict) => (*verdict, None, None),
        VerdictSource::Compare { report, arm } => {
            let effect = compared(report, arm, state.parameters.alpha)?;
            let verdict = rules::classify_effect(effect.as_ref(), target, state.parameters.alpha);
            let text = format!(
                "experiences compare-arms: {arm} against {}",
                report.reference
            );
            (verdict, effect, Some(text))
        }
        VerdictSource::Gate { arm, generation } => {
            let changed = check_arm(
                spec,
                arm,
                Some(&candidate.variable),
                candidate.phase == Phase::Engineering,
                &state.parameters.engineering,
            )?;
            let promotion = gate_record(ctx, arm, *generation)?;
            let (verdict, effect) = classify_promotion(&promotion, target);
            let text = format!(
                "the promotion gate of {} generation {generation} ({} changed): {}",
                arm.name,
                changed.changed.join(", "),
                if promotion.promoted {
                    "promoted"
                } else {
                    "kept"
                }
            );
            (verdict, effect, Some(text))
        }
    };
    let reason = match (request.reason, measured) {
        (Some(reason), _) if !reason.trim().is_empty() => reason,
        (_, Some(measured)) => measured,
        _ => {
            return Err(config(
                "a stated verdict is recorded with its reason: give one",
            ))
        }
    };
    problems.record(Event::Verdict {
        candidate: request.candidate,
        experiment: request.experiment,
        verdict,
        effect,
        evidence: request.evidence,
        reason,
    })
}

/// The effect of `arm`'s contrast in `report`, at the state's level: the
/// family interval when the report has one, the task interval otherwise, and
/// the Holm-adjusted two-sided sign test.
fn compared(report: &ArmComparison, arm: &str, alpha: f64) -> Result<Option<Effect>, ServedError> {
    if (report.level - (1.0 - alpha)).abs() > 1e-9 {
        return Err(config(format!(
            "the report's intervals hold {} of the resamples; the problem state decides at {}",
            report.level,
            1.0 - alpha
        )));
    }
    let contrast = report
        .contrasts
        .iter()
        .find(|c| c.arm == arm)
        .ok_or_else(|| {
            config(format!(
                "the report has no contrast of arm {arm:?} against {:?}",
                report.reference
            ))
        })?;
    Ok(contrast.difference.and_then(|difference| {
        let bounds = contrast.family_interval.or(contrast.task_interval)?;
        Some(Effect {
            difference,
            low: bounds.low,
            high: bounds.high,
            p: Some(contrast.p_two_sided_holm),
        })
    }))
}

fn gate_record(
    ctx: &Context,
    arm: &CampaignSpec,
    generation: u32,
) -> Result<Promotion, ServedError> {
    let (ledger, _) =
        begun(ctx, arm)?.ok_or_else(|| config(format!("campaign {} has not begun", arm.name)))?;
    let name = unit(generation, "gate");
    let commits = ledger.commits()?;
    let commit = commits
        .iter()
        .rev()
        .find(|c| c.unit == name)
        .ok_or_else(|| {
            config(format!(
                "campaign {} records no gate of generation {generation}",
                arm.name
            ))
        })?;
    record(commit)
}

/// Whether `arm` is a single-variable change from `champion` in `variable`
/// (when one is given) and, with `engineering`, stays inside the
/// `allowed` engineering paths.
///
/// # Errors
/// An arm that changes nothing, more than the variable, or (with
/// `engineering`) anything outside the engineering paths; each path named.
pub fn check_arm(
    champion: &CampaignSpec,
    arm: &CampaignSpec,
    variable: Option<&str>,
    engineering: bool,
    allowed: &[String],
) -> Result<ArmCheck, ServedError> {
    if variable.is_none() && !engineering {
        return Err(config(
            "check an arm against a variable, the engineering paths, or both",
        ));
    }
    let (ours, theirs) = (configuration(champion)?, configuration(arm)?);
    let refused = |e: rules::VariableError| config(e.to_string());
    let mut changed = Vec::new();
    if let Some(variable) = variable {
        changed = single_variable(&ours, &theirs, variable).map_err(refused)?;
    }
    if engineering {
        changed = firewall(&ours, &theirs, allowed).map_err(refused)?;
    }
    Ok(ArmCheck {
        variable: variable.map(ToString::to_string),
        engineering,
        changed,
    })
}

/// The next hypothesis of `phase` for the campaign `spec` names, and its
/// arm when the candidate declares a value.
///
/// # Errors
/// No problem state, a value the campaign file cannot hold, an arm that
/// fails its own checks, or the ledgers.
pub fn campaign_next(
    ctx: &Context,
    spec: &CampaignSpec,
    phase: Phase,
) -> Result<NextReport, ServedError> {
    let state = problems_show(ctx, spec)?.state;
    let proposal = state.next(phase);
    let mut report = NextReport {
        campaign: spec.name.clone(),
        proposal: proposal.clone(),
        arm: None,
        changed: Vec::new(),
    };
    let Some((proposal, value)) = proposal.and_then(|p| p.value.clone().map(|v| (p, v))) else {
        return Ok(report);
    };
    let mut arm = spec.clone();
    // The campaign's ledger is released at the end of this statement, before
    // its status opens it again.
    let seed = begun(ctx, spec)?.map(|(_, seed)| seed);
    if let Some(seed) = seed {
        arm.seed = Some(seed);
        let champion = campaign_status(ctx, spec)?.champion;
        if champion.generation > 0 {
            arm.champion = Some(champion.model);
        }
    }
    arm.name = arm_name(&spec.name, &proposal.candidate);
    let mut json =
        serde_json::to_value(&arm).map_err(|e| config(format!("campaign specification: {e}")))?;
    set_path(&mut json, &proposal.variable, value)?;
    let arm: CampaignSpec = serde_json::from_value(json).map_err(|e| {
        config(format!(
            "candidate {} sets {} to a value the campaign file cannot hold: {e}",
            proposal.candidate, proposal.variable
        ))
    })?;
    arm.validate()?;
    let checked = check_arm(
        spec,
        &arm,
        Some(&proposal.variable),
        phase == Phase::Engineering,
        &state.parameters.engineering,
    )?;
    report.changed = checked.changed;
    report.arm = Some(arm);
    Ok(report)
}

fn arm_name(campaign: &str, candidate: &str) -> String {
    let tail: String = candidate
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_".contains(c) {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let name = format!("{campaign}-{tail}");
    name.chars().take(64).collect()
}

fn set_path(value: &mut Value, path: &str, leaf: Value) -> Result<(), ServedError> {
    let mut at = value;
    let segments: Vec<&str> = path.split('.').collect();
    for (i, segment) in segments.iter().enumerate() {
        let map = match at {
            Value::Object(map) => map,
            other @ Value::Null => {
                *other = Value::Object(serde_json::Map::new());
                match other {
                    Value::Object(map) => map,
                    _ => unreachable!("just made an object"),
                }
            }
            _ => {
                return Err(config(format!(
                    "{path}: {} is a setting, not a section",
                    segments[..i].join(".")
                )))
            }
        };
        if i + 1 == segments.len() {
            map.insert((*segment).to_string(), leaf);
            return Ok(());
        }
        at = map.entry((*segment).to_string()).or_insert(Value::Null);
    }
    Ok(())
}

/// `arm` as campaign-file text, read back and compared before it is
/// returned. The seed is written as text, which holds any seed a TOML
/// integer cannot.
///
/// # Errors
/// A specification TOML cannot hold, or one that does not read back as
/// itself.
pub fn arm_text(arm: &CampaignSpec) -> Result<String, ServedError> {
    let mut unseeded = arm.clone();
    let seed = unseeded.seed.take();
    let mut text =
        toml::to_string(&unseeded).map_err(|e| config(format!("the arm as TOML: {e}")))?;
    if let Some(seed) = seed {
        text = format!("seed = \"{seed}\"\n{text}");
    }
    let read = CampaignSpec::parse(&text)?;
    if &read != arm {
        return Err(config(
            "the arm's campaign file does not read back as the arm",
        ));
    }
    Ok(text)
}
