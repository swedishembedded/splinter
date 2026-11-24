// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning pipelines whose interrupted
// runs resume and whose finished candidates are served automatically, for
// its clients. If your team needs expertise in model training
// infrastructure, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A training stage: one variant trained on the dataset product, then
//! served once so it counts only when the trainer's server loads it.
//!
//! The stage identity is the dataset product, the base, the method, the
//! parameters, the seed and the trainer: a changed rank is a new candidate,
//! and the dataset and rollouts it was built from are reused. The trainer
//! is asked under a resume key derived from the identity, so a run cut short
//! continues where its last checkpoint left it.
//!
//! The product holds the trained files, `candidate.json` (the
//! [`ModelCandidate`]) and `report.json` (the trainer's report).

use serde_json::json;
use splinter_core::factory::execution::FileRef;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{CandidateKind, ModelCandidate, CANDIDATE_SCHEMA};
use splinter_store::factory::products::Staging;

use super::dataset::dataset_of;
use super::{io, stage_error, write_json, Factory, RunError};
use crate::ops::TrainParams;
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;
use crate::training::{sha256, TrainRequest, Trainer};

/// The candidate's description in a training product.
pub const CANDIDATE_FILE: &str = "candidate.json";

/// A `model.train` stage with its base resolved.
pub(crate) struct Variant {
    /// The champion whose adapter the candidate continues (`start`).
    pub(crate) start: Option<crate::champion::Champion>,
    pub(crate) name: String,
    pub(crate) engine: String,
    pub(crate) base: ModelRef,
    pub(crate) method: String,
    pub(crate) params: serde_json::Value,
    pub(crate) seed: u64,
}

pub(crate) fn variant(factory: &Factory<'_>, stage: &PlannedStage) -> Result<Variant, RunError> {
    let params: TrainParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
    let base = factory
        .resolved
        .model(&params.base)
        .cloned()
        .ok_or_else(|| {
            stage_error(
                &stage.key,
                format!("model role {} is not bound", params.base),
            )
        })?;
    let start = params
        .start
        .as_deref()
        .map(|role| {
            let reference = factory.resolved.model(role).ok_or_else(|| {
                stage_error(&stage.key, format!("model role {role} is not bound"))
            })?;
            let champion = factory
                .resolved
                .champions
                .get(&reference.to_string())
                .cloned()
                .ok_or_else(|| stage_error(&stage.key, format!("start {role} is not a champion")))?;
            if champion.record.base != base {
                return Err(stage_error(
                    &stage.key,
                    format!(
                        "start {role} is served from {}, not from the base {base} the candidate trains on",
                        champion.record.base
                    ),
                ));
            }
            if champion.adapter.is_none() {
                return Err(stage_error(&stage.key, format!("start {role} has no adapter to continue")));
            }
            Ok(champion)
        })
        .transpose()?;
    Ok(Variant {
        start,
        name: stage.key.clone(),
        engine: params.engine,
        base,
        method: params.method,
        params: params.params,
        seed: params.seed,
    })
}

fn trainer<'a>(factory: &Factory<'a>, stage: &PlannedStage) -> Result<&'a dyn Trainer, RunError> {
    factory
        .services
        .trainer
        .ok_or_else(|| stage_error(&stage.key, "no trainer is configured"))
}

fn dataset_key(stage: &PlannedStage) -> Result<String, RunError> {
    stage
        .source("dataset")
        .map(str::to_owned)
        .ok_or_else(|| stage_error(&stage.key, "trains on no dataset"))
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let variant = variant(factory, stage)?;
    let key = dataset_key(stage)?;
    let inputs = vec![(key.clone(), factory.product(&key)?.1.digest())];
    let mut parameters = json!({
        "engine": variant.engine, "base": variant.base, "method": variant.method,
        // As the trainer will train them: a changed default is a new candidate.
        "params": trainer(factory, stage)?.resolve(&variant.method, &variant.params).map_err(|e| stage_error(&stage.key, e))?,
    });
    if let Some(start) = &variant.start {
        // By content: the same adapter continued is the same input, wherever its run is. A candidate
        // trained from nothing keeps the identity it had before a start could be named.
        parameters["start"] = json!(start.digest.hex());
    }
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &variant.name,
        parameters,
        inputs,
        &trainer(factory, stage)?.identity(),
        Some(variant.seed),
    ))
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    identity: &StageIdentity,
    staging: &Staging,
) -> Result<(), RunError> {
    let variant = variant(factory, stage)?;
    let trainer = trainer(factory, stage)?;
    let key = dataset_key(stage)?;
    let view = factory.view(&key);
    let dataset = dataset_of(&view)?;
    let stage_id = identity.digest();
    // The tag names the candidate at the trainer; it is the identity, so a
    // changed variant is a different candidate there too.
    let tag = format!("{}-{}", variant.name, &stage_id.hex()[..12]);
    let out = factory.work(&stage.key).join(&tag);
    std::fs::create_dir_all(&out).map_err(io(&out))?;
    let trained = trainer
        .train(&TrainRequest {
            base: &variant.base,
            method: &variant.method,
            params: &variant.params,
            objective: &dataset.objective,
            train: &view.join(&dataset.train.path),
            validation: &view.join(&dataset.validation.path),
            seed: variant.seed,
            // The mode the dataset was built for.
            thinking: dataset.thinking,
            resume_key: &tag,
            out: &out,
            start: variant.start.as_ref().and_then(|c| c.adapter.as_deref()),
        })
        .map_err(|e| stage_error(&stage.key, e))?;
    let mut artifacts = Vec::new();
    for file in &trained.files {
        let name = file.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
            stage_error(&stage.key, format!("{} has no file name", file.display()))
        })?;
        std::fs::copy(file, staging.path().join(name)).map_err(io(file))?;
        artifacts.push(FileRef {
            path: name.to_owned(),
            digest: sha256(file).map_err(io(file))?,
        });
    }
    let adapter = trained
        .adapter
        .as_ref()
        .and_then(|a| a.file_name())
        .map(|n| staging.path().join(n));
    // The load check: the candidate counts only once the trainer's server lists it.
    let served_as = trainer
        .serve(&variant.base, adapter.as_deref())
        .map_err(|e| stage_error(&stage.key, e))?;
    let candidate = ModelCandidate {
        schema: CANDIDATE_SCHEMA.into(),
        name: variant.name.clone(),
        kind: if trained.adapter.is_some() {
            CandidateKind::Adapter
        } else {
            CandidateKind::Full
        },
        base: variant.base.to_string(),
        artifacts,
        dataset: factory.product(&key)?.1.digest(),
        method: variant.method.clone(),
        params: trained.params,
        trainer: trainer.identity(),
        seed: variant.seed,
        report: trained.report.clone(),
        served_as: Some(served_as),
        load_checked: true,
    };
    write_json(&staging.path().join("report.json"), &trained.report)?;
    write_json(&staging.path().join(CANDIDATE_FILE), &candidate)
}

/// The candidate of a committed training view.
///
/// # Errors
/// Its description cannot be read.
pub fn candidate_of(view: &std::path::Path) -> Result<ModelCandidate, RunError> {
    super::read_json(&view.join(CANDIDATE_FILE))
}
