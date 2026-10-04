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

use std::path::Path;

use serde::{Deserialize, Serialize};
use splinter_core::annotation::Outcome;
use splinter_core::digest::Digest;
use splinter_core::experience::ExperienceId;
use splinter_eval::verifiers::calibration::{calibrate, Calibration};

use crate::exam::{misjudged, Misjudged};
use crate::experiences::resolve_experience;
use crate::judging::{controls, reference, spaced};
use crate::verify::{
    judge_verifier, judge_verifier_for, store_calibration, Judging, MAX_CALIBRATION_TASKS,
};
use splinter_core::model_ref::ModelRef;
use splinter_eval::verifiers::calibration::{measure, DEFAULT_MIN_PRECISION};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};
use splinter_store::tasks::TaskSetId;

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
    /// Its address: kept, and from now on the judge's latest calibration.
    pub stored: Digest,
}

/// Measures the judge `judge` names on the experiences `labelled` lists.
pub fn calibrate_judge(
    ctx: &Context,
    labelled: &Path,
    judge: &ModelRef,
) -> Result<Calibrated, OrchestratorError> {
    let text = std::fs::read_to_string(labelled).map_err(io(labelled))?;
    let mut examples = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let entry: Labelled =
            serde_json::from_str(line).map_err(|source| OrchestratorError::Json {
                what: format!("{} line {}", labelled.display(), index + 1),
                source,
            })?;
        if entry.label == Outcome::Abstain {
            return Err(OrchestratorError::Refused(format!(
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
        return Err(OrchestratorError::Refused(format!(
            "{} labels no experience",
            labelled.display()
        )));
    }
    let model = ctx.model(judge)?;
    let verifier = judge_verifier(ctx, &model);
    let calibration = calibrate(&verifier, &examples)?;
    let stored = store_calibration(ctx, &calibration)?;
    Ok(Calibrated {
        judge: model.identity,
        calibration,
        stored,
    })
}

/// What `judge measure` reports.
#[derive(Clone, Debug, Serialize)]
pub struct JudgeMeasured {
    /// The judge's identity.
    pub judge: String,
    /// What it was asked: `reference` or `fit`.
    pub judging: &'static str,
    /// Controls it was measured on.
    pub controls: usize,
    /// Whether its precision, on passes and on fails, reaches the threshold
    /// and the controls are as many as the configuration asks for: whether
    /// its verdicts would count.
    pub trusted: bool,
    /// The measurement.
    pub calibration: Calibration,
    /// Where it was kept, when it rests on enough controls to be reused.
    pub stored: Option<Digest>,
    /// The controls it did not judge as labelled.
    pub misjudged: Vec<Misjudged>,
}

/// Measures the judge `judge` names on the controls the references of the
/// tasks of `set` give ([`controls`]): each task's reference as the right
/// answer to it, and another family's as the wrong one. What
/// [`crate::verify::Judge::calibrated_for`] would measure and refuse on, shown
/// whatever it comes to, with the controls the judge got wrong; kept as the
/// judge's calibration when it rests on enough of them, so a run reuses it.
pub fn measure_judge(
    ctx: &Context,
    set: &TaskSetId,
    judge: &ModelRef,
    judging: Judging,
) -> Result<JudgeMeasured, OrchestratorError> {
    let store = ctx.tasks();
    let tasks: Vec<splinter_core::experience::Task> = store
        .get_set(set)?
        .members
        .iter()
        .map(|entry| store.get(&entry.task))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|task| reference(task).is_some())
        .collect();
    let labelled = controls(ctx, &spaced(&tasks, MAX_CALIBRATION_TASKS))?;
    if labelled.is_empty() {
        return Err(OrchestratorError::Refused(
            "the tasks give no control: a judge is measured on tasks with a reference from at \
             least two families of sources"
                .into(),
        ));
    }
    let model = ctx.model(judge)?;
    let verifier = judge_verifier_for(ctx, &model, judging);
    let (calibration, measurements) = measure(&verifier, &labelled)?;
    let minimum = ctx.config().min_calibration_controls;
    let precise = |p: Option<f64>| p.is_some_and(|p| p >= DEFAULT_MIN_PRECISION);
    let enough = labelled.len() >= minimum;
    let stored = enough
        .then(|| store_calibration(ctx, &calibration))
        .transpose()?;
    Ok(JudgeMeasured {
        judge: model.identity,
        judging: match judging {
            Judging::Reference => "reference",
            Judging::Fit => "fit",
        },
        controls: labelled.len(),
        trusted: enough
            && precise(calibration.precision_pass)
            && precise(calibration.precision_fail),
        calibration,
        stored,
        misjudged: misjudged(&labelled, &measurements),
    })
}
