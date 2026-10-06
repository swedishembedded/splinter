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
//! brain's own `TimelineModel::load` and asks it for a structured forecast of
//! a patient history: no Splinter store, gate or adapter is involved after
//! the release is found. The history is brain's patient-history format (`as_of`,
//! optional `birth`, and `events` that are measurements when they carry a
//! `value` and a `unit`, events otherwise). A new checkup is appended by
//! sending the whole history again with one more record; nothing is kept
//! between calls. The report prints both forecasts as brain returns them: the
//! as-of time, the model's identity, what the history covered, the curves, the
//! risk at each horizon (calibrated, with its interval, only where the model
//! was calibrated for exactly that horizon), the data-quality warnings and
//! whether the model abstained.

use std::path::Path;

use anyhow::{Context as _, Result};
use serde_json::{json, Value as Json};
use splinter_sdk::model::timeline::serving::{history_json, parse_history};
use splinter_sdk::model::timeline::{read_jsonl, ForecastRequest, Subject, TimelineModel};
use splinter_sdk::timeline::release::resolve_release;
use splinter_sdk::vocabulary::dataset::DatasetId;
use splinter_sdk::vocabulary::digest::Digest;

use crate::state::Run;

/// The horizons the forecast is given at, in the dataset's time unit: the
/// ones the evaluation judges.
pub const HORIZONS: [f64; 2] = [5.0, 10.0];

/// How a checkup is appended.
#[derive(Clone, Debug)]
pub struct Checkup {
    /// The variable measured.
    pub var: String,
    /// Its value.
    pub value: f64,
    /// The unit it is stated in; the model refuses one other than the unit it
    /// was trained on, and never converts.
    pub unit: Option<String>,
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

/// `history` sent again with the checkup appended: one more record, and the
/// as-of time moved to it (a measurement at the as-of time is known).
pub fn with_checkup(history: &Json, checkup: &Checkup) -> Result<Json> {
    let as_of = history["as_of"]
        .as_f64()
        .context("the history's as_of is not a number on the model's clock")?;
    let at = as_of + checkup.after;
    let mut record = json!({"time": at, "code": checkup.var, "value": checkup.value});
    if let Some(unit) = &checkup.unit {
        record["unit"] = json!(unit);
    }
    let mut later = history.clone();
    later["as_of"] = json!(at);
    // The calendar time moves with the as-of time; it is stated, not derived,
    // for a history on the model's own clock.
    let calendar = history["calendar"]
        .as_f64()
        .context("the history states no calendar time")?;
    later["calendar"] = json!(calendar + checkup.after);
    later["events"]
        .as_array_mut()
        .context("the history has no events list")?
        .push(record);
    Ok(later)
}

/// The forecast of `history` as brain serves it, as JSON.
pub fn forecast(model: &TimelineModel, history: &Json, horizons: &[f64]) -> Result<Json> {
    let history = parse_history(history)?;
    let forecast = model
        .forecast_history(&history, &ForecastRequest::new(horizons.iter().copied()))
        .context("forecasting the patient history")?;
    Ok(serde_json::to_value(&forecast)?)
}

/// Both forecasts of `subject`: before, and after `checkup` is appended to its
/// history. A checkup that is already in the history changes nothing: the
/// forecast is the same bit for bit and the duplicate is reported among the
/// input warnings.
pub fn compare(
    model: &TimelineModel,
    subject: &Subject,
    checkup: &Checkup,
    horizons: &[f64],
) -> Result<Json> {
    let history = history_json(subject);
    let appended = with_checkup(&history, checkup)?;
    let before = forecast(model, &history, horizons)?;
    let after = forecast(model, &appended, horizons)?;
    let numbers = |forecast: &Json| {
        let mut f = forecast.clone();
        f.as_object_mut().map(|o| o.remove("input_warnings"));
        f
    };
    Ok(json!({
        "history": history,
        "appended": appended,
        "before": before,
        "after": after,
        "forecast_changed": numbers(&before) != numbers(&after),
    }))
}

/// Predicts subject `index` of the test part from the release `release`
/// before and after a new checkup.
pub fn predict(
    run: &Run,
    release: &str,
    index: usize,
    checkup: &Checkup,
    horizons: &[f64],
) -> Result<Json> {
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
    let mut report = compare(&model, subject, checkup, horizons)?;
    report["release"] = json!(id.to_string());
    report["loaded_by"] = json!("brain TimelineModel::load on the unpacked file");
    report["horizons"] = json!(horizons);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_sdk::model::timeline::{synthetic, train_timeline, TimelineTraining};

    /// A tiny model trained on a cohort that states `sd` as the unit of its
    /// first risk factor, and a subject of that cohort.
    fn fixture() -> (TimelineModel, Subject) {
        let (mut all, _) = synthetic::population(300, 5);
        for s in &mut all {
            for o in s.observations.iter_mut().filter(|o| o.var == "x1") {
                o.unit = Some("sd".into());
            }
        }
        let (fit, rest) = all.split_at(220);
        let mut config =
            TimelineTraining::new(["death:a", "death:b", "onset"], ["death:a", "death:b"]);
        config.steps = 20;
        config.batch = 32;
        config.max_tokens = Some(16);
        config.eval_interval = 10;
        let model = train_timeline(fit, rest, &config).unwrap().model;
        (model, rest[0].clone())
    }

    fn checkup(value: f64, unit: Option<&str>, after: f64) -> Checkup {
        Checkup {
            var: "x1".into(),
            value,
            unit: unit.map(str::to_owned),
            after,
        }
    }

    fn value_of(subject: &Subject, var: &str) -> f64 {
        match subject
            .observations
            .iter()
            .find(|o| o.var == var && o.t == subject.entry)
            .unwrap()
            .value
        {
            splinter_sdk::model::timeline::Value::Number(v) => v,
            _ => panic!("{var} is not a number"),
        }
    }

    #[test]
    fn a_new_checkup_changes_the_forecast_and_the_report_carries_brains_fields() {
        let (model, subject) = fixture();
        let new = value_of(&subject, "x1") + 2.0;
        let report = compare(
            &model,
            &subject,
            &checkup(new, Some("sd"), 2.0),
            &[2.0, 4.0],
        )
        .unwrap();
        assert_eq!(report["forecast_changed"], true);
        let (before, after) = (&report["before"], &report["after"]);
        for forecast in [before, after] {
            for field in [
                "as_of", "model", "coverage", "curves", "horizons", "support",
            ] {
                assert!(!forecast[field].is_null(), "{field} in {forecast}");
            }
            assert_eq!(forecast["horizons"].as_array().unwrap().len(), 2);
            assert!(
                forecast["model"]["weights_sha256"].is_string(),
                "{}",
                forecast["model"]
            );
        }
        assert_ne!(
            before["as_of"], after["as_of"],
            "the as-of time moved to the checkup"
        );
        assert_eq!(
            before["model"], after["model"],
            "the same model answered both"
        );
    }

    #[test]
    fn a_duplicate_checkup_leaves_the_forecast_bit_identical_and_is_reported() {
        let (model, subject) = fixture();
        let same = value_of(&subject, "x1");
        let report = compare(
            &model,
            &subject,
            &checkup(same, Some("sd"), 0.0),
            &[2.0, 4.0],
        )
        .unwrap();
        assert_eq!(report["forecast_changed"], false);
        let (before, after) = (&report["before"], &report["after"]);
        for field in [
            "as_of", "model", "coverage", "curves", "horizons", "support", "risk",
        ] {
            assert_eq!(before[field], after[field], "{field}");
        }
        assert!(before["input_warnings"].as_array().unwrap().is_empty());
        let warnings = after["input_warnings"].to_string();
        assert!(warnings.contains("duplicate_ignored"), "{warnings}");
    }

    #[test]
    fn a_malformed_or_mismatched_unit_is_refused_by_name() {
        let (model, subject) = fixture();
        let new = value_of(&subject, "x1") + 1.0;
        for malformed in [" sd", "", "a-unit-name-that-is-far-longer-than-thirty-two"] {
            let why = compare(
                &model,
                &subject,
                &checkup(new, Some(malformed), 1.0),
                &[2.0],
            )
            .unwrap_err()
            .to_string();
            assert!(why.contains("unit"), "{malformed:?}: {why}");
        }
        // A well-formed unit the model was not trained on is refused, never converted.
        let why = format!(
            "{:#}",
            compare(&model, &subject, &checkup(new, Some("mg/dL"), 1.0), &[2.0]).unwrap_err()
        );
        assert!(
            why.contains("x1") && why.contains("sd") && why.contains("mg/dL"),
            "{why}"
        );
    }
}
