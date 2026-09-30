// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements calibrated model judges whose verdicts
// carry the precision they were measured at, for its clients. If your team
// needs expertise in LLM-as-judge evaluation or reward modelling, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `judge calibrate`: a judge measured against stored experiences whose
//! correct outcome a person labelled, and the measurement kept as that
//! judge's calibration, which `verify --judge` gates its verdicts by.
//!
//! The labelled file is JSON Lines, one experience per line:
//! `{"experience": "<id or unique prefix>", "label": "pass" | "fail"}`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_lab::verifiers::calibration::{calibrate, Calibration};
use splinter_store::annotation::Outcome;
use splinter_store::experience::ExperienceId;

use crate::context::Context;
use crate::error::{io, CampaignError};
use crate::experiences::resolve_experience;
use crate::model_ref::ModelRef;
use crate::verify::{judge_verifier, store_calibration};

/// One line of a labelled file.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Labelled {
    experience: String,
    label: Outcome,
}

/// What `judge calibrate` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Calibrated {
    /// The judge's identity.
    pub judge: String,
    /// The measurement: how often its passes and its fails were right.
    pub calibration: Calibration,
    /// Where it is kept, as the judge's latest calibration.
    pub path: PathBuf,
}

/// Measures the judge `judge` names on the experiences `labelled` lists.
pub fn calibrate_judge(
    ctx: &Context,
    labelled: &Path,
    judge: &ModelRef,
) -> Result<Calibrated, CampaignError> {
    let text = std::fs::read_to_string(labelled).map_err(io(labelled))?;
    let mut examples = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let entry: Labelled = serde_json::from_str(line).map_err(|source| CampaignError::Json {
            what: format!("{} line {}", labelled.display(), index + 1),
            source,
        })?;
        if entry.label == Outcome::Abstain {
            return Err(CampaignError::Refused(format!(
                "{} line {}: a label is pass or fail",
                labelled.display(),
                index + 1
            )));
        }
        let id: ExperienceId = resolve_experience(ctx, &entry.experience)?;
        let experience = ctx.experiences().get(&id)?;
        examples.push((experience.to_task(), experience, entry.label));
    }
    if examples.is_empty() {
        return Err(CampaignError::Refused(format!(
            "{} labels no experience",
            labelled.display()
        )));
    }
    let model = ctx.model(judge)?;
    let verifier = judge_verifier(ctx, &model);
    let calibration = calibrate(&verifier, &examples)?;
    let path = store_calibration(ctx, &calibration)?;
    Ok(Calibrated {
        judge: model.identity,
        calibration,
        path,
    })
}
