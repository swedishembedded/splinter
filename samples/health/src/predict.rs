// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the deployment side of a released risk
// model: the shipped file, loaded by the model's own runtime with nothing of
// the training system around it. If your team needs expertise in deploying
// audited prediction models, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `predict`: the released artifact, loaded by plain brain.
//!
//! A release is one file, a plain ustar archive of the saved model's files.
//! This step unpacks it with the system `tar`, loads the directory with
//! brain's own `TimelineModel::load` and predicts: no Splinter store, gate or
//! adapter is involved after the release is found. It predicts the risk curves
//! of one history, then appends a new checkup (a measurement taken after the
//! first prediction time, which makes it the new prediction time, since a
//! record has one) and predicts again. The two sets of curves must differ.

use std::path::Path;

use anyhow::{Context as _, Result};
use splinter_sdk::model::timeline::{
    read_jsonl, AtRisk, Observation, Prediction, Subject, TimelineModel, Value,
};
use splinter_sdk::timeline::release::resolve_release;
use splinter_sdk::vocabulary::dataset::DatasetId;
use splinter_sdk::vocabulary::digest::Digest;

use crate::state::Run;

/// The horizons the curves are printed at, in the dataset's time unit.
pub const HORIZONS: [f64; 4] = [1.0, 2.0, 5.0, 10.0];

/// How a checkup is appended.
#[derive(Clone, Debug)]
pub struct Checkup {
    /// The variable measured.
    pub var: String,
    /// Its value.
    pub value: f64,
    /// How long after the first prediction time.
    pub after: f64,
}

/// Unpacks the release file into `into` with the system `tar`: it is a plain
/// ustar archive, so this needs nothing of Splinter's.
pub fn unpack(file: &Path, into: &Path) -> Result<()> {
    std::fs::create_dir_all(into)?;
    let status = std::process::Command::new("tar")
        .arg("-xf")
        .arg(file)
        .arg("-C")
        .arg(into)
        .status()
        .context("running tar to unpack the release")?;
    anyhow::ensure!(status.success(), "tar could not unpack {}", file.display());
    Ok(())
}

fn curves(model: &TimelineModel, subject: &Subject) -> Result<serde_json::Value> {
    let p: Prediction = model.predict(std::slice::from_ref(subject))?.remove(0);
    let mut out = serde_json::Map::new();
    for code in model.codes() {
        let by: Vec<f64> = HORIZONS
            .iter()
            .map(|h| p.cif(code, *h).unwrap_or(f64::NAN))
            .collect();
        out.insert(code.clone(), serde_json::json!(by));
    }
    out.insert(
        "survival".into(),
        serde_json::json!(HORIZONS.iter().map(|h| p.survival(*h)).collect::<Vec<_>>()),
    );
    Ok(serde_json::Value::Object(out))
}

/// The subject as of a checkup `after` later at which `var` was measured as
/// `value`: everything known up to then, and none of the outcomes that follow.
pub fn after_checkup(subject: &Subject, checkup: &Checkup) -> Subject {
    let at = subject.entry + checkup.after;
    let mut later = subject.clone();
    later.events.retain(|e| e.t < subject.entry);
    later.observations.push(Observation {
        t: at,
        var: checkup.var.clone(),
        value: Value::Number(checkup.value),
    });
    later.entry = at;
    let end = subject
        .at_risk
        .iter()
        .map(|w| w.to)
        .fold(at + 10.0, f64::max);
    later.at_risk = vec![AtRisk {
        code: "*".into(),
        from: at,
        to: end,
    }];
    later
}

/// Predicts subject `index` of the test part from the release `release`
/// before and after a new checkup.
pub fn predict(
    run: &Run,
    release: &str,
    index: usize,
    checkup: &Checkup,
) -> Result<serde_json::Value> {
    let ctx = run.context()?;
    let id = resolve_release(&ctx, release)?;
    let stored = ctx.releases().get(&id)?;
    let dir = run.predict_dir().join(id.0.hex());
    if !dir.exists() {
        unpack(&stored.artifact, &dir)?;
    }
    // From here on, brain alone.
    let model = TimelineModel::load(&dir)?;
    let split = run.split()?;
    let test = ctx.timeline_datasets().get(&DatasetId(
        Digest::parse(&split.report.test).map_err(anyhow::Error::msg)?,
    ))?;
    let subjects = read_jsonl(&test.path)?;
    let subject = subjects.get(index).with_context(|| {
        format!(
            "the test part has {} subjects; there is no subject {index}",
            subjects.len()
        )
    })?;
    let before = curves(&model, subject)?;
    let later = after_checkup(subject, checkup);
    let after = curves(&model, &later)?;
    let changed = before != after;
    anyhow::ensure!(
        changed,
        "appending a checkup of {} = {} did not change the risk curves",
        checkup.var,
        checkup.value
    );
    Ok(serde_json::json!({
        "release": id.to_string(),
        "loaded_by": "brain TimelineModel::load on the unpacked file",
        "horizons": HORIZONS,
        "entry": subject.entry,
        "before": before,
        "appended": {"var": checkup.var, "value": checkup.value, "at": later.entry},
        "after": after,
        "curves_changed": changed,
    }))
}
