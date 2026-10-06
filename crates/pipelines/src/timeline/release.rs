// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated releases of risk models: a candidate
// replaces the model in place only when pre-registered requirements hold on
// held-out evidence, and the decision is recorded either way. If your team
// needs expertise in release governance for prediction models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The release stage for a timeline candidate: an evaluation becomes a
//! release only through the predictive gate.
//!
//! The evaluation says what was measured and under which requirements; this
//! stage checks that it was measured against the champion the alias really
//! points at (a candidate that beat a weaker model is not thereby better than
//! the release in place), hands the checkpoint, the numbers and the lineage
//! to [`release_predictive`](crate::release::predictive::release_predictive)
//! and reports its decision. A candidate that fails is recorded as rejected
//! and the alias stays where it was.

use splinter_core::terms::Distribution;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::releases::Provenance;

use super::evaluate::{load_timeline_evaluation, ModelRef};
use super::records::ARCHITECTURE;
use super::train::load_timeline_candidate;
use crate::release::predictive::{
    release_predictive, CheckpointFile, CheckpointTraining, PredictiveRelease, PredictiveReleased,
};
use splinter_eval::predictive_gate::Measurements;

/// One `release_timeline` command.
#[derive(Clone, Debug)]
pub struct TimelineReleaseRequest {
    /// The evaluation to release on, by address or unique prefix.
    pub evaluation: String,
    /// The alias the release replaces its champion on.
    pub alias: String,
    /// How widely the release may be handed on; the default keeps it where
    /// it was made.
    pub distribution: Distribution,
}

/// Releases the evaluated candidate if, and only if, the gate passes.
pub fn release_timeline(
    ctx: &Context,
    request: &TimelineReleaseRequest,
) -> Result<PredictiveReleased, OrchestratorError> {
    let (_, evaluation) = load_timeline_evaluation(ctx, &request.evaluation)?;
    let (candidate_id, record) = load_timeline_candidate(ctx, &evaluation.candidate)?;
    let in_place = ctx.releases().alias(&request.alias)?;
    match (&in_place, &evaluation.champion) {
        (Some(current), ModelRef::Release { id }) if current == id => {}
        (Some(current), measured) => {
            return Err(OrchestratorError::Refused(format!(
                "alias {} points at release {current} but the candidate was measured against \
                 {measured:?}: evaluate it against the release in place",
                request.alias
            )))
        }
        (None, _) => {}
    }
    let units = evaluation.comparison.units.clone();
    let training_records = ctx.timeline_datasets().get(&record.train)?.manifest.records;
    release_predictive(
        ctx,
        &PredictiveRelease {
            candidate: candidate_id,
            alias: request.alias.clone(),
            datasets: vec![record.train.to_string(), record.held_out.to_string()],
            checkpoint: CheckpointFile {
                path: ctx.artifacts().path(&record.checkpoint)?,
                architecture: ARCHITECTURE.into(),
                digest: record.checkpoint_sha256.clone(),
            },
            training: CheckpointTraining {
                from: "scratch".into(),
                objective: "timeline".into(),
                steps: record.outcome.steps,
                records: training_records,
                record: serde_json::to_value(&record).map_err(|source| {
                    OrchestratorError::Json {
                        what: "timeline candidate record".into(),
                        source,
                    }
                })?,
            },
            spec: evaluation.spec.clone(),
            measurements: Measurements {
                champion_units: units.clone(),
                candidate_units: units,
                evidence: evaluation.evidence.clone(),
            },
            distribution: request.distribution,
            provenance: Provenance {
                training_config: Some(record.config_digest.clone()),
                brain_commit: record.brain_commit.clone(),
                splinter_commit: record.splinter_commit.clone(),
                ..Provenance::default()
            },
        },
    )
}
