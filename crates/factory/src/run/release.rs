// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model releases promoted only on an
// independent, pre-registered verdict, for its clients. If your team needs
// expertise in safe model release processes, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The release stage: what the run decided about its candidates.
//!
//! Every candidate's evaluation is recorded. With `promote: true`, the
//! candidate an evaluation found `improved` with the largest pass-rate
//! difference is promoted (a protected plan's only on a signed verdict);
//! an inconclusive or worse evaluation promotes nothing and keeps the
//! candidate. With `promote: false` the run is an experiment: nothing is
//! promoted whatever the verdicts.
//!
//! The release also names the champion the run leaves in use, which a
//! later generation's recipe names `champion:RUN` ([`crate::champion`]):
//! the promoted candidate; else the evaluation's baseline, kept (a champion
//! baseline stays the champion). A trained champion's adapter is copied
//! into the product, so the record is whole without the run's other
//! products.
//!
//! The product holds `release.json` and the champion's adapter.

use std::path::PathBuf;

use serde_json::json;
use splinter_core::digest::Digest;
use splinter_core::factory::identity::{StageIdentity, StageKind};
use splinter_core::factory::product::Decision;
use splinter_store::factory::products::Staging;

use super::eval::result_of;
use super::training::candidate_of;
use super::{io, stage_error, write_json, Factory, RunError};
use crate::champion::{Champion, ChampionRecord, ADAPTER_FILE};
use crate::plan::{stage_key, PlannedStage};
use crate::recipe::ModelRef;

/// The release decision in a release product.
pub const RELEASE_FILE: &str = "release.json";

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let mut inputs = Vec::new();
    for key in &stage.depends_on {
        inputs.push((key.clone(), factory.product(key)?.1.digest()));
    }
    let promote = factory
        .resolved
        .recipe
        .release
        .as_ref()
        .is_some_and(|r| r.promote);
    let kept = kept_champion(factory, &stage.key)?;
    Ok(super::identity_of(
        StageKind::Release,
        &stage.name,
        json!({ "promote": promote, "kept": kept.0, "kept_digest": kept.1.map(|c| c.digest) }),
        inputs,
        "",
        None,
    ))
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
) -> Result<(), RunError> {
    let promote = factory
        .resolved
        .recipe
        .release
        .as_ref()
        .is_some_and(|r| r.promote);
    let mut candidates = Vec::new();
    let mut best: Option<(f64, String)> = None;
    for key in &stage.depends_on {
        let result = result_of(&factory.view(key))?;
        let name = result.name.clone();
        let candidate = candidate_of(&factory.view(&stage_key(StageKind::Training, &name)))?;
        let difference = result
            .statistics
            .get("difference")
            .and_then(serde_json::Value::as_f64);
        let eligible = result.decision == Decision::Improved
            && (!result.protected || result.signature.is_some());
        if eligible {
            let gain = difference.unwrap_or(0.0);
            if best.as_ref().is_none_or(|(b, _)| gain > *b) {
                best = Some((gain, name.clone()));
            }
        }
        candidates.push(json!({
            "candidate": name, "decision": result.decision, "protected": result.protected,
            "signed": result.signature.is_some(), "difference": difference,
            "served_as": candidate.served_as, "artifacts": candidate.artifacts,
        }));
    }
    let promoted = if promote {
        best.map(|(_, name)| name)
    } else {
        None
    };
    let champion = match &promoted {
        Some(name) => {
            let view = factory.view(&stage_key(StageKind::Training, name));
            let candidate = candidate_of(&view)?;
            let adapter = candidate
                .artifacts
                .iter()
                .find(|a| a.path.ends_with(".safetensors"))
                .map(|a| (view.join(&a.path), a.digest.clone()));
            let base = super::eval::training_base(factory, &stage.key)?.clone();
            champion_record(staging, base, adapter, Some(name.clone()))?
        }
        None => match kept_champion(factory, &stage.key)? {
            (_, Some(kept)) => champion_record(
                staging,
                kept.record.base.clone(),
                kept.adapter.clone().zip(kept.record.adapter_digest.clone()),
                kept.record.candidate.clone(),
            )?,
            (base, None) => champion_record(staging, base, None, None)?,
        },
    };
    write_json(
        &staging.path().join(RELEASE_FILE),
        &json!({
            "schema": "splinter.release/1", "promote": promote, "promoted": promoted,
            "candidates": candidates, "champion": champion,
        }),
    )
}

/// The model the run keeps when it promotes nothing: the evaluation's
/// baseline (a champion, resolved), else the base its variants train from.
fn kept_champion(
    factory: &Factory<'_>,
    key: &str,
) -> Result<(ModelRef, Option<Champion>), RunError> {
    let recipe = &factory.resolved.recipe;
    let model = match (&recipe.eval, &recipe.training) {
        (Some(eval), _) => eval.baseline.clone(),
        (None, Some(training)) => training.base.clone(),
        (None, None) => {
            return Err(stage_error(
                key,
                "a release decides about trained candidates",
            ))
        }
    };
    let champion = factory.resolved.champions.get(&model.to_string()).cloned();
    Ok((model, champion))
}

/// A champion record, its adapter copied into the release product.
fn champion_record(
    staging: &Staging,
    base: ModelRef,
    adapter: Option<(PathBuf, Digest)>,
    candidate: Option<String>,
) -> Result<ChampionRecord, RunError> {
    let (adapter, adapter_digest) = match adapter {
        Some((source, digest)) => {
            let target = staging.path().join(ADAPTER_FILE);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(io(parent))?;
            }
            std::fs::copy(&source, &target).map_err(io(&source))?;
            (Some(ADAPTER_FILE.to_owned()), Some(digest))
        }
        None => (None, None),
    };
    Ok(ChampionRecord {
        base,
        adapter,
        adapter_digest,
        candidate,
    })
}
