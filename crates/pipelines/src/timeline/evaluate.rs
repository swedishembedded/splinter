// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements head-to-head evaluation of a candidate risk
// model against the champion in place, on participants neither was fitted
// on, under requirements written before scoring. If your team needs expertise
// in honest release decisions for prediction models, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The evaluation stage: a candidate and a champion scored on the SAME
//! held-out units, under requirements registered before anything is scored.
//!
//! The units are the test part of the split the candidate was cut from. Before
//! scoring, the stage refuses a test part of another split, a subject outside
//! the part the split assigns, and a champion that was trained or
//! early-stopped on any test unit: numbers over units a model has seen are not
//! a held-out measurement. Both arms are loaded the way a consumer loads them
//! (the packed file, unpacked, loaded by plain brain), so what is measured is
//! the shipped model. The metrics are brain's (`TimelineModel::evaluate`) and
//! survival arithmetic through the model adapter
//! ([`splinter_model::timeline::scoring`]); this stage names the arms, hands
//! the numbers on and keeps the evaluation as an immutable document. It also
//! measures serving correctness on the test units ([`super::serving`]).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_data::split::Part;
use splinter_eval::metric_gate::Evidence;
use splinter_eval::predictive_gate::PredictiveSpec;
use splinter_eval::timeline_metrics::TimelinePlan;
use splinter_model::timeline::scoring::{compare, predict_arm, Comparison, ScoreSpec};
use splinter_model::timeline::{read_jsonl, Subject};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::ids;

use super::records::TimelineCandidate;
use super::release::resolve_release;
use super::serving::{check_serving, ServingCheck};
use super::train::{lift, load_bundle, load_timeline_candidate, Scratch};

/// The `format` of an evaluation record.
pub const EVALUATION_FORMAT: &str = "splinter-timeline-evaluation-v1";
/// The document class of evaluations.
pub const EVALUATION_CLASS: &str = "timeline_evaluation";

/// Which model an arm of an evaluation is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelRef {
    /// A candidate that was trained, released or not (a baseline, say).
    Candidate {
        /// Its id.
        id: String,
    },
    /// A release.
    Release {
        /// Its id.
        id: ReleaseId,
    },
}

/// Who the candidate is measured against, as a command names it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Champion {
    /// A trained candidate (an additive or age-only baseline, say), by id.
    Candidate(String),
    /// A release, by id or unique prefix, or the alias that points at one.
    Release(String),
}

/// One `evaluate_timeline` command.
#[derive(Clone, Debug, Serialize)]
pub struct TimelineEvaluateRequest {
    /// The candidate, by id or unique prefix.
    pub candidate: String,
    /// The model it would replace.
    pub champion: Champion,
    /// The stored test part both are scored on, by id or unique prefix.
    pub test: String,
    /// How the arms are scored.
    pub scoring: ScoreSpec,
    /// The requirements, registered before scoring.
    pub plan: TimelinePlan,
}

/// A candidate scored against a champion on held-out units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimelineEvaluation {
    /// [`EVALUATION_FORMAT`].
    pub format: String,
    /// The candidate's id.
    pub candidate: String,
    /// The champion it was scored against.
    pub champion: ModelRef,
    /// The test dataset both were scored on.
    pub test: DatasetId,
    /// The address of the split the test part belongs to.
    pub split: Digest,
    /// The requirements, written before anything was scored.
    pub spec: PredictiveSpec,
    /// How the arms were scored.
    pub scoring: ScoreSpec,
    /// Every number of both arms and their differences.
    pub comparison: Comparison,
    /// The same numbers by the names the gate reads, with the serving check.
    pub evidence: Evidence,
    /// What the serving check measured on the test units.
    #[serde(default)]
    pub serving: Option<ServingCheck>,
}

impl TimelineEvaluation {
    /// The evaluation's address.
    pub fn digest(&self) -> Result<Digest, serde_json::Error> {
        splinter_core::digest::canonical_json(self).map(|b| Digest::of(&b))
    }
}

/// What `evaluate_timeline` reports.
#[derive(Clone, Debug, Serialize)]
pub struct TimelineEvaluated {
    /// The evaluation's address, which `release_timeline` is given.
    pub id: Digest,
    /// The evaluation.
    pub evaluation: TimelineEvaluation,
}

fn subjects_of(ctx: &Context, id: &DatasetId) -> Result<Vec<Subject>, OrchestratorError> {
    let stored = ctx.timeline_datasets().get(id)?;
    read_jsonl(&stored.path).map_err(|e| lift(e.into()))
}

/// The models of an arm: the record that trained it and the packed file.
struct Arm {
    model_ref: ModelRef,
    record: TimelineCandidate,
    file: std::path::PathBuf,
}

fn arm_of(ctx: &Context, champion: &Champion) -> Result<Arm, OrchestratorError> {
    match champion {
        Champion::Candidate(id) => {
            let (id, record) = load_timeline_candidate(ctx, id)?;
            let file = ctx.artifacts().path(&record.checkpoint)?;
            Ok(Arm {
                model_ref: ModelRef::Candidate { id },
                record,
                file,
            })
        }
        Champion::Release(given) => {
            let store = ctx.releases();
            let id = resolve_release(ctx, given)?;
            let release = store.get(&id)?;
            let (_, record) =
                load_timeline_candidate(ctx, &release.manifest.candidate).map_err(|e| {
                    OrchestratorError::Refused(format!(
                        "release {id} was not made from a timeline candidate this store holds: {e}"
                    ))
                })?;
            Ok(Arm {
                model_ref: ModelRef::Release { id },
                record,
                file: release.artifact,
            })
        }
    }
}

/// The ids of every unit a record's model was fitted or early-stopped on.
fn seen_by(
    ctx: &Context,
    record: &TimelineCandidate,
) -> Result<HashSet<String>, OrchestratorError> {
    let mut seen = HashSet::new();
    for dataset in [&record.train, &record.held_out] {
        seen.extend(subjects_of(ctx, dataset)?.into_iter().map(|s| s.subject_id));
    }
    Ok(seen)
}

/// Scores `request.candidate` against `request.champion` on the test part and
/// keeps the evaluation. Nothing is kept when a check fails.
pub fn evaluate_timeline(
    ctx: &Context,
    request: &TimelineEvaluateRequest,
) -> Result<TimelineEvaluated, OrchestratorError> {
    let (candidate_id, record) = load_timeline_candidate(ctx, &request.candidate)?;
    let candidate = Arm {
        model_ref: ModelRef::Candidate {
            id: candidate_id.clone(),
        },
        file: ctx.artifacts().path(&record.checkpoint)?,
        record,
    };
    let champion = arm_of(ctx, &request.champion)?;

    let store = ctx.timeline_datasets();
    let test_id = DatasetId(ids::resolve(
        "timeline dataset",
        &request.test,
        store.list()?.into_iter().map(|d| d.0),
    )?);
    let test = store.get(&test_id)?;
    if test.manifest.part != Some(Part::Test) {
        return Err(OrchestratorError::Refused(format!(
            "dataset {test_id} is the {:?} part of its split, not the Test part: held-out scores are taken on the test part",
            test.manifest.part
        )));
    }
    if test.manifest.split.as_ref() != Some(&candidate.record.split) {
        return Err(OrchestratorError::Refused(format!(
            "the test part is of split {:?} but the candidate was cut from split {}: the test units are not known to be unseen",
            test.manifest.split, candidate.record.split
        )));
    }
    let split = store.split(&candidate.record.split)?;
    let subjects = read_jsonl(&test.path).map_err(|e| lift(e.into()))?;
    let assigned: HashSet<&str> = split.ids(Part::Test).iter().map(String::as_str).collect();
    if let Some(stray) = subjects
        .iter()
        .find(|s| !assigned.contains(s.subject_id.as_str()))
    {
        return Err(OrchestratorError::Refused(format!(
            "test dataset {test_id} holds a unit its split does not assign to the test part (e.g. {:?})",
            stray.subject_id
        )));
    }
    for (name, arm) in [("candidate", &candidate), ("champion", &champion)] {
        let seen = seen_by(ctx, &arm.record)?;
        let leaked = subjects
            .iter()
            .filter(|s| seen.contains(&s.subject_id))
            .count();
        if leaked > 0 {
            return Err(OrchestratorError::Refused(format!(
                "the {name} was fitted or early-stopped on {leaked} of the {} test units: a score on them is not held out",
                subjects.len()
            )));
        }
    }

    for (name, arm) in [("candidate", &candidate), ("champion", &champion)] {
        let (mut trained, mut asked) = (
            arm.record.config.absorbing.clone(),
            request.scoring.absorbing.clone(),
        );
        trained.sort();
        asked.sort();
        if trained != asked {
            return Err(OrchestratorError::Refused(format!(
                "the {name} was trained with absorbing codes {trained:?} but the scoring names {asked:?}: \
                 brain scores a code against the codes its model ended follow-up for"
            )));
        }
    }

    let scratch = Scratch::new(ctx, "evaluate")?;
    let champion_model = load_bundle(&champion.file, &scratch.0, "champion")?;
    let candidate_model = load_bundle(&candidate.file, &scratch.0, "candidate")?;
    let serving = check_serving(
        &candidate.record,
        &candidate.file,
        &scratch.0.join("served"),
        &candidate_model,
        &subjects,
        &subjects_of(ctx, &candidate.record.held_out)?,
        &request.scoring.horizons,
    )?;
    let champion_arm = predict_arm(&champion_model, &subjects, &request.scoring).map_err(lift)?;
    let candidate_arm = predict_arm(&candidate_model, &subjects, &request.scoring).map_err(lift)?;
    let comparison =
        compare(&champion_arm, &candidate_arm, &subjects, &request.scoring).map_err(lift)?;
    let mut evidence = comparison.evidence();
    serving.add_to(&mut evidence);

    let evaluation = TimelineEvaluation {
        format: EVALUATION_FORMAT.into(),
        candidate: candidate_id,
        champion: champion.model_ref,
        test: test_id,
        split: candidate.record.split.clone(),
        spec: request.plan.spec(),
        scoring: request.scoring.clone(),
        comparison,
        evidence,
        serving: Some(serving),
    };
    let id = ctx
        .workspace()
        .put_document(EVALUATION_CLASS, &evaluation)?;
    Ok(TimelineEvaluated { id, evaluation })
}

/// The evaluation `id` (a digest or a unique prefix) names.
pub fn load_timeline_evaluation(
    ctx: &Context,
    id: &str,
) -> Result<(Digest, TimelineEvaluation), OrchestratorError> {
    let stored = ctx.workspace().document_ids(EVALUATION_CLASS)?;
    let digest = ids::resolve("timeline evaluation", id, stored)?;
    let evaluation = ctx
        .workspace()
        .get_document(EVALUATION_CLASS, &digest)?
        .ok_or_else(|| OrchestratorError::NotFound {
            what: "timeline evaluation",
            id: id.to_string(),
        })?;
    Ok((digest, evaluation))
}
