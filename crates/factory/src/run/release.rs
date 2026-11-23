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
//! The product holds `release.json`.

use serde_json::json;
use splinter_core::factory::identity::{StageIdentity, StageKind};
use splinter_core::factory::product::Decision;
use splinter_store::factory::products::Staging;

use super::eval::result_of;
use super::training::candidate_of;
use super::{write_json, Factory, RunError};
use crate::plan::{stage_key, PlannedStage};

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
    Ok(super::identity_of(
        StageKind::Release,
        &stage.name,
        json!({ "promote": promote }),
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
        let name = result.candidate.arm.clone();
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
    write_json(
        &staging.path().join(RELEASE_FILE),
        &json!({ "schema": "splinter.release/1", "promote": promote, "promoted": promoted, "candidates": candidates }),
    )
}
