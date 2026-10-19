// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements generations of a code-editing agent that
// keep what earlier generations learned, for its clients. If your team needs
// expertise in measuring retention across successive fine-tunes of an
// agent, you can procure our services by sending an email to
// info@swedishembedded.com.

//! One generation: measure the champion on every protected task, train a
//! candidate on the new admitted experience, measure it the same way, and
//! release it only if it keeps what the champion had.
//!
//! The champion and the candidate are measured on the same protected
//! tasks, under the same bounds and seeds ([`super::panel`]); the frozen
//! incumbent is the champion as it was when the generation began, whatever
//! happens to the alias meanwhile. A candidate is rejected, whatever the
//! release gate would say, when
//!
//! * any protected family's pass rate falls by more than
//!   [`Limits::max_family_drop`] (a retention limit),
//! * the mean over the protected families falls by more than
//!   [`Limits::max_mean_drop`] (a regression limit), or
//! * it does not beat the champion on the new generation's exam by at least
//!   [`Limits::min_new_gain`].
//!
//! One that passes goes through the release gate ([`crate::release`]) like
//! any candidate: the gate's sign test on the new exam, the retention check
//! on every earlier exam, the canaries and the served check. Only the gate
//! moves the alias; a candidate it blocks is rejected with its reasons.
//! Every generation, accepted or not, is kept in a ledger
//! ([`read_ledger`]) with the measurements it was decided on.

use serde::{Deserialize, Serialize};
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_data::ReplayPolicy;
use splinter_eval::gate::GateConfig;
use splinter_store::tasks::TaskSetId;

use super::panel::{measure, Cell, Panel, PanelColumn, PanelRow};
use crate::practice::{exam_canaries, practice, PracticeReport, PracticeRequest};
use crate::release::exams;
use crate::release::probe::Arm;
use crate::release::{arm, release_recorded, ReleaseRequest};
use crate::train::Trainer;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::runs::{record, Recorded};

/// How the earlier releases' experience is carried into the next
/// generation's training.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayChoice {
    /// Nothing is replayed: the candidate continues the champion on the new
    /// records alone.
    None,
    /// A seeded share of every earlier record, whatever it is of.
    #[default]
    Uniform,
    /// The same number of records of each capability.
    CapabilityBalanced,
    /// Records of the capabilities the champion fails most, in proportion.
    FailureTargeted,
    /// A fresh adapter on the frozen base, trained on all the verified
    /// experience accumulated so far and the new, nothing continued.
    RebuildFromBase,
}

/// What a candidate may cost the capabilities already held.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Limits {
    /// The most a protected family's pass rate may fall.
    pub max_family_drop: f64,
    /// The most the mean over protected families may fall.
    pub max_mean_drop: f64,
    /// What the candidate must gain on the new exam: more than this.
    pub min_new_gain: f64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_family_drop: 0.2,
            max_mean_drop: 0.05,
            min_new_gain: 0.0,
        }
    }
}

impl Limits {
    fn validate(&self) -> Result<(), OrchestratorError> {
        let fraction = |name: &str, value: f64| {
            if value.is_finite() && (0.0..=1.0).contains(&value) {
                Ok(())
            } else {
                Err(OrchestratorError::Refused(format!(
                    "the limit {name} is {value}, not a share in [0, 1]"
                )))
            }
        };
        fraction("max_family_drop", self.max_family_drop)?;
        fraction("max_mean_drop", self.max_mean_drop)?;
        fraction("min_new_gain", self.min_new_gain)
    }
}

/// One generation's request.
#[derive(Clone, Debug, Serialize)]
pub struct GenerationRequest {
    /// The experiment this generation belongs to: names its ledger.
    pub experiment: String,
    /// The repository practised and how; the run stops at the candidate and
    /// the release is this stage's, so `no_release` is set by it.
    pub practice: PracticeRequest,
    /// How earlier experience is replayed.
    pub replay: ReplayChoice,
    /// What the candidate may cost.
    pub limits: Limits,
}

/// Why a family is named in a rejection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FamilyChange {
    /// The family.
    pub family: String,
    /// The champion's pass rate on it.
    pub champion: f64,
    /// The candidate's.
    pub candidate: f64,
}

/// What the measurements say, before the gate is asked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Assessment {
    /// The candidate's gain over the champion on the new exam; `None` when
    /// it could not be measured.
    pub new_gain: Option<f64>,
    /// The mean change over the protected families both were measured on;
    /// `None` when there is none (the first generation).
    pub mean_change: Option<f64>,
    /// Protected families the candidate lost more of than the limit.
    pub regressions: Vec<FamilyChange>,
    /// Protected families one of the two had no verdict on: counted nowhere.
    pub unmeasured: Vec<String>,
    /// Why the candidate is rejected; empty when it is not.
    pub reasons: Vec<String>,
}

impl Assessment {
    /// Whether the measurements let the candidate go on to the gate.
    #[must_use]
    pub fn accepted(&self) -> bool {
        self.reasons.is_empty()
    }
}

/// Compares `candidate` with `champion`, rows over the same panel whose
/// column `new_column` is the new generation's exam; see the module
/// documentation.
#[must_use]
pub fn assess(
    champion: &PanelRow,
    candidate: &PanelRow,
    new_column: usize,
    limits: &Limits,
) -> Assessment {
    let protected = |row: &PanelRow| {
        let kept = PanelRow {
            cells: row
                .cells
                .iter()
                .filter(|c| c.column != new_column)
                .cloned()
                .collect(),
            ..row.clone()
        };
        kept.by_family()
    };
    let (before, after) = (protected(champion), protected(candidate));
    let mut regressions = Vec::new();
    let mut changes = Vec::new();
    let mut unmeasured = Vec::new();
    let mut families: Vec<&String> = before.keys().chain(after.keys()).collect();
    families.sort();
    families.dedup();
    for family in families {
        match (before.get(family), after.get(family)) {
            (Some((was, _)), Some((is, _))) => {
                changes.push(is - was);
                if was - is > limits.max_family_drop {
                    regressions.push(FamilyChange {
                        family: family.clone(),
                        champion: *was,
                        candidate: *is,
                    });
                }
            }
            _ => unmeasured.push(family.clone()),
        }
    }
    let mean_change =
        (!changes.is_empty()).then(|| changes.iter().sum::<f64>() / changes.len() as f64);
    let new_gain = match (champion.rate(new_column).0, candidate.rate(new_column).0) {
        (Some(was), Some(is)) => Some(is - was),
        _ => None,
    };
    let mut reasons = Vec::new();
    for r in &regressions {
        reasons.push(format!(
            "retention: family {} fell from {:.3} to {:.3}, more than the limit {}",
            r.family, r.champion, r.candidate, limits.max_family_drop
        ));
    }
    if let Some(change) = mean_change.filter(|c| -c > limits.max_mean_drop) {
        reasons.push(format!(
            "regression: the mean over protected families fell by {:.3}, more than the limit {}",
            -change, limits.max_mean_drop
        ));
    }
    match new_gain {
        None => reasons.push("the new exam could not be measured for both models".into()),
        Some(gain) if gain <= limits.min_new_gain => reasons.push(format!(
            "gain: the candidate gained {gain:.3} on the new exam; it must gain more than {}",
            limits.min_new_gain
        )),
        Some(_) => {}
    }
    Assessment {
        new_gain,
        mean_change,
        regressions,
        unmeasured,
        reasons,
    }
}

/// How a generation ended.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    /// Released: the alias points at it now.
    Accepted {
        /// The release.
        release: ReleaseId,
    },
    /// Not released, and why.
    Rejected {
        /// The reasons, from the limits and the gate.
        reasons: Vec<String>,
    },
    /// Nothing was trained, and why.
    Stopped {
        /// Why.
        reason: String,
    },
}

/// What the ledger keeps of one generation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationRecord {
    /// The experiment.
    pub experiment: String,
    /// The generation, counted from 1.
    pub generation: usize,
    /// The record before this one.
    pub previous: Option<Digest>,
    /// The champion the generation began from; `None` for the base.
    pub champion: Option<ReleaseId>,
    /// The candidate trained, when one was.
    pub candidate: Option<String>,
    /// How replay was chosen, and what it was.
    pub replay: ReplayChoice,
    /// The replay policy the trainer was given.
    pub replay_policy: ReplayPolicy,
    /// What the candidate was allowed to cost.
    pub limits: Limits,
    /// The champion on the protected panel and the new exam.
    pub champion_row: Option<PanelRow>,
    /// The candidate on the same.
    pub candidate_row: Option<PanelRow>,
    /// What the measurements said.
    pub assessment: Option<Assessment>,
    /// How it ended.
    pub outcome: Outcome,
    /// When, from the injected clock.
    pub decided_at: String,
}

/// What a generation reports.
#[derive(Clone, Debug, Serialize)]
pub struct GenerationReport {
    /// The ledger entry, as kept.
    pub record: GenerationRecord,
    /// Where it is kept.
    pub ledger: Digest,
    /// What `practice` did.
    pub practice: PracticeReport,
}

const LEDGER: &str = "generation";
const POINTER_PREFIX: &str = "generations-";

fn pointer_of(experiment: &str) -> Result<String, OrchestratorError> {
    let ok = !experiment.is_empty()
        && experiment.len() <= 64
        && experiment
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_');
    if ok {
        Ok(format!("{POINTER_PREFIX}{experiment}"))
    } else {
        Err(OrchestratorError::Refused(format!(
            "{experiment:?} is not an experiment name: lowercase letters, digits, - and _, at most \
             64"
        )))
    }
}

/// Every generation of `experiment`, oldest first.
pub fn read_ledger(
    ctx: &Context,
    experiment: &str,
) -> Result<Vec<GenerationRecord>, OrchestratorError> {
    ctx.workspace().refresh()?;
    let Some((_, head)) = ctx.workspace().pointer(&pointer_of(experiment)?)? else {
        return Ok(Vec::new());
    };
    let mut next = Some(Digest::parse(&head).map_err(|e| {
        OrchestratorError::Refused(format!("the ledger of {experiment} is corrupt: {e}"))
    })?);
    let mut records = Vec::new();
    while let Some(digest) = next {
        let record: GenerationRecord =
            ctx.workspace()
                .get_document(LEDGER, &digest)?
                .ok_or_else(|| {
                    OrchestratorError::Refused(format!(
                        "the ledger of {experiment} names a record {digest} that is not kept"
                    ))
                })?;
        next = record.previous.clone();
        records.push(record);
    }
    records.reverse();
    Ok(records)
}

fn append(ctx: &Context, record: &GenerationRecord) -> Result<Digest, OrchestratorError> {
    let pointer = pointer_of(&record.experiment)?;
    let stored = ctx.workspace().put_document(LEDGER, record)?;
    let current = ctx.workspace().pointer(&pointer)?.map(|(_, v)| v);
    ctx.workspace().move_pointer(
        &pointer,
        current.as_deref(),
        stored.as_str(),
        &ctx.clock().utc_now(),
    )?;
    Digest::parse(stored.as_str()).map_err(|e| {
        OrchestratorError::Refused(format!("the ledger kept record {stored} unreadably: {e}"))
    })
}

/// The replay weights the champion's failures give: every group the earlier
/// release `k` trained on weighs the champion's failure rate on that
/// release's exam.
fn failure_weights(
    ctx: &Context,
    champion: &PanelRow,
    lineage: &[splinter_orchestrator::releases::StoredRelease],
    panel: &Panel,
) -> Result<std::collections::BTreeMap<String, f64>, OrchestratorError> {
    let mut weights = std::collections::BTreeMap::new();
    for (column, entry) in panel.columns.iter().enumerate() {
        let Some(release) = lineage.get(entry.learned_at - 1) else {
            continue;
        };
        let failure = champion.rate(column).0.map_or(0.0, |rate| 1.0 - rate);
        for family in exams::trained_families(ctx, &release.manifest.datasets, &[])? {
            weights.insert(family, failure);
        }
    }
    Ok(weights)
}

/// Runs one generation of `request.experiment`; see the module
/// documentation.
pub fn run_generation(
    ctx: &Context,
    request: &GenerationRequest,
    trainer: &dyn Trainer,
    cancel: &CancelToken,
) -> Result<Recorded<GenerationReport>, OrchestratorError> {
    request.limits.validate()?;
    pointer_of(&request.experiment)?;
    if request.practice.exam_families == Some(0) {
        return Err(OrchestratorError::Refused(
            "a generation is judged on the exam it reserves: --exam-families 0 reserves none"
                .into(),
        ));
    }
    record(ctx, "generation", request, |run| {
        let store = ctx.releases();
        let champion_id = store.alias(splinter_core::model_ref::POLICY_DEFAULT)?;
        let mut lineage = match &champion_id {
            Some(id) => store.lineage(id)?,
            None => Vec::new(),
        };
        lineage.reverse();
        let generation = lineage.len() + 1;
        let work = request.practice.agentic();
        let config = ctx.config();

        // The frozen incumbent, measured before anything is trained.
        let panel = Panel::of_lineage(ctx, splinter_core::model_ref::POLICY_DEFAULT)?;
        let champion_adapter = lineage
            .last()
            .map(splinter_orchestrator::releases::StoredRelease::adapter)
            .transpose()?;
        let champion_ref = arm(config, champion_adapter);
        let champion_model = ctx.model(&champion_ref)?;
        let champion_arm = Arm::named(&champion_ref, &champion_model);
        let who = (lineage.len(), champion_id.clone());
        let before = measure(ctx, &champion_arm, who.clone(), &panel, work, cancel)?;

        let (policy, rebuild) = match request.replay {
            ReplayChoice::None => (ReplayPolicy::None, false),
            ReplayChoice::Uniform => (ReplayPolicy::Uniform, false),
            ReplayChoice::CapabilityBalanced => (ReplayPolicy::CapabilityBalanced, false),
            ReplayChoice::FailureTargeted => (
                ReplayPolicy::FailureTargeted {
                    weights: failure_weights(ctx, &before, &lineage, &panel)?,
                },
                false,
            ),
            ReplayChoice::RebuildFromBase => (ReplayPolicy::None, true),
        };
        let practised = practice(
            ctx,
            &PracticeRequest {
                no_release: true,
                replay_policy: policy.clone(),
                rebuild,
                ..request.practice.clone()
            },
            trainer,
        )?;
        let practice_report = practised.report;
        let mut entry = GenerationRecord {
            experiment: request.experiment.clone(),
            generation,
            previous: None,
            champion: champion_id.clone(),
            candidate: None,
            replay: request.replay,
            replay_policy: policy,
            limits: request.limits,
            champion_row: None,
            candidate_row: None,
            assessment: None,
            outcome: Outcome::Stopped {
                reason: String::new(),
            },
            decided_at: ctx.clock().utc_now(),
        };
        let (Some(candidate), Some(reserve)) = (
            practice_report.candidate.clone(),
            practice_report.reserve.clone(),
        ) else {
            entry.outcome = Outcome::Stopped {
                reason: practice_report
                    .stopped
                    .clone()
                    .unwrap_or_else(|| "practice trained no candidate".into()),
            };
            return keep(ctx, entry, practice_report);
        };
        entry.candidate = Some(candidate.candidate.clone());

        // The new exam is a column of the panel for both models.
        let exam = new_exam(ctx, reserve.exam_task_set.as_ref(), generation)?;
        let Some(exam) = exam else {
            entry.outcome = Outcome::Stopped {
                reason: "the run reserved no exam task to judge the candidate on".into(),
            };
            return keep(ctx, entry, practice_report);
        };
        // The champion was measured on the protected panel before anything
        // was trained; only the new exam is left to measure it on.
        let new_column = panel.columns.len();
        let added = measure(
            ctx,
            &champion_arm,
            who,
            &Panel {
                columns: vec![exam.clone()],
            },
            work,
            cancel,
        )?;
        let mut champion_row = before;
        champion_row
            .cells
            .extend(added.cells.into_iter().map(|cell| Cell {
                column: cell.column + new_column,
                ..cell
            }));
        let full = panel.clone().with(exam);
        let candidate_ref = arm(config, Some(&candidate.adapter));
        let candidate_model = ctx.model(&candidate_ref)?;
        let candidate_row = measure(
            ctx,
            &Arm::named(&candidate_ref, &candidate_model),
            (generation, None),
            &full,
            work,
            cancel,
        )?;
        let assessment = assess(&champion_row, &candidate_row, new_column, &request.limits);
        entry.champion_row = Some(champion_row);
        entry.candidate_row = Some(candidate_row);
        entry.assessment = Some(assessment.clone());
        if !assessment.accepted() {
            entry.outcome = Outcome::Rejected {
                reasons: assessment.reasons,
            };
            return keep(ctx, entry, practice_report);
        }

        // Only the gate moves the alias.
        let released = release_recorded(
            ctx,
            run,
            &ReleaseRequest {
                gate: request.practice.gate.applied_to(GateConfig {
                    agentic: work,
                    ..GateConfig::default()
                }),
                canaries: exam_canaries(ctx, Some(&reserve), request.practice.canaries)?,
                ..ReleaseRequest::new(candidate.candidate.clone())
            },
        );
        entry.outcome = match released {
            Ok(released) => match released.release {
                Some(release) => Outcome::Accepted { release },
                None => Outcome::Rejected {
                    reasons: released
                        .gate
                        .blocked_by()
                        .into_iter()
                        .map(|why| format!("gate: {why}"))
                        .collect(),
                },
            },
            Err(e) if e.is_refusal() => Outcome::Rejected {
                reasons: vec![format!("gate: {e}")],
            },
            Err(e) => return Err(e),
        };
        keep(ctx, entry, practice_report)
    })
}

/// The new generation's exam as a column, when the run kept tasks for it.
fn new_exam(
    ctx: &Context,
    set: Option<&TaskSetId>,
    generation: usize,
) -> Result<Option<PanelColumn>, OrchestratorError> {
    let Some(set) = set else { return Ok(None) };
    let mut tasks: Vec<splinter_core::experience::Task> = Vec::new();
    for entry in ctx.tasks().get_set(set)?.members {
        let task = ctx.tasks().get(&entry.task)?;
        if !tasks.iter().any(|t| t.task.id == task.task.id) {
            tasks.push(task);
        }
    }
    Ok((!tasks.is_empty()).then(|| PanelColumn {
        name: format!("exam of generation {generation}"),
        learned_at: generation,
        tasks,
    }))
}

/// Appends `entry` to its experiment's ledger and reports it.
fn keep(
    ctx: &Context,
    mut entry: GenerationRecord,
    practice: PracticeReport,
) -> Result<GenerationReport, OrchestratorError> {
    entry.previous = head_of(ctx, &entry.experiment)?;
    let ledger = append(ctx, &entry)?;
    Ok(GenerationReport {
        record: entry,
        ledger,
        practice,
    })
}

/// The digest of the newest record of `experiment`.
fn head_of(ctx: &Context, experiment: &str) -> Result<Option<Digest>, OrchestratorError> {
    let Some((_, head)) = ctx.workspace().pointer(&pointer_of(experiment)?)? else {
        return Ok(None);
    };
    Digest::parse(&head)
        .map(Some)
        .map_err(|e| OrchestratorError::Refused(format!("the ledger of {experiment}: {e}")))
}
