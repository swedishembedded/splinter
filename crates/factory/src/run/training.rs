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
use splinter_core::factory::identity::{StageIdentity, StageKind};
use splinter_core::factory::product::{CandidateKind, ModelCandidate, CANDIDATE_SCHEMA};
use splinter_store::factory::products::Staging;

use super::dataset::dataset_of;
use super::{io, stage_error, write_json, Factory, RunError};
use crate::plan::{stage_key, PlannedStage};
use crate::recipe::{TrainingSection, TrainingVariant};
use crate::training::{sha256, TrainRequest, Trainer};

/// The candidate's description in a training product.
pub const CANDIDATE_FILE: &str = "candidate.json";

fn variant<'r>(
    factory: &'r Factory<'_>,
    stage: &PlannedStage,
) -> Result<(&'r TrainingSection, &'r TrainingVariant), RunError> {
    let training = factory
        .resolved
        .recipe
        .training
        .as_ref()
        .ok_or_else(|| stage_error(&stage.key, "the recipe trains nothing"))?;
    let variant = training
        .variants
        .iter()
        .find(|v| v.name == stage.name)
        .ok_or_else(|| stage_error(&stage.key, "is not a training variant"))?;
    Ok((training, variant))
}

fn trainer<'a>(factory: &Factory<'a>, stage: &PlannedStage) -> Result<&'a dyn Trainer, RunError> {
    factory
        .services
        .trainer
        .ok_or_else(|| stage_error(&stage.key, "no trainer is configured"))
}

fn dataset_key(factory: &Factory<'_>, stage: &PlannedStage) -> Result<String, RunError> {
    let dataset = factory
        .resolved
        .recipe
        .dataset
        .as_ref()
        .ok_or_else(|| stage_error(&stage.key, "no dataset"))?;
    Ok(stage_key(StageKind::Dataset, &dataset.name))
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let (training, variant) = variant(factory, stage)?;
    let key = dataset_key(factory, stage)?;
    let inputs = vec![(key.clone(), factory.product(&key)?.1.digest())];
    let parameters = json!({
        "engine": training.engine, "base": training.base, "method": variant.method, "params": variant.params,
    });
    Ok(super::identity_of(
        StageKind::Training,
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
    let (training, variant) = variant(factory, stage)?;
    let trainer = trainer(factory, stage)?;
    let key = dataset_key(factory, stage)?;
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
            base: &training.base,
            method: &variant.method,
            params: &variant.params,
            train: &view.join(&dataset.train.path),
            validation: &view.join(&dataset.validation.path),
            seed: variant.seed,
            thinking: factory
                .resolved
                .recipe
                .dataset
                .as_ref()
                .is_some_and(|d| d.thinking),
            resume_key: &tag,
            out: &out,
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
        .serve(&training.base, adapter.as_deref())
        .map_err(|e| stage_error(&stage.key, e))?;
    let candidate = ModelCandidate {
        schema: CANDIDATE_SCHEMA.into(),
        name: variant.name.clone(),
        kind: if trained.adapter.is_some() {
            CandidateKind::Adapter
        } else {
            CandidateKind::Full
        },
        base: training.base.to_string(),
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
