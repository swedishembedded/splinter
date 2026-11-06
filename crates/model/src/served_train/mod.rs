// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose training runs as a
// job on a shared model service, scheduled with its inference, for its
// clients. If your team needs expertise in moving fine-tuning off the
// application and onto a serving cluster, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Training as a request to a brain service.
//!
//! With brain endpoints configured, a fine-tune ([`fine_tune_served`]) or a
//! policy-gradient run ([`train_policy_gradient_served`]) is a job of the
//! service (`brain serve --training --artifact-store DIR`), not a model loaded
//! in this process: the service owns the cards, takes them from idle serving
//! replicas for the run and gives them back.
//!
//! Nothing bulk goes through the API. The datasets (and an adapter to
//! continue) are published to the artifact store the endpoints share with
//! this machine ([`ArtifactStore`], `--brain-artifact-store`), named by their
//! BLAKE3 digest; the job names them by digest; the trained adapter and the
//! training record come back the same way. The base is named by the id
//! brain's model store gives it (`<org>/<model>`, as for inference). The
//! settings are the ones the in-process trainers take, so a run trains the
//! same either way; what changes is where the work happens.
//!
//! The datasets are checked here exactly as the in-process path checks them,
//! before anything is sent: a record the trainer would refuse is named by its
//! line without spending a queue slot.

mod artifacts;
mod job;
mod request;

pub use artifacts::ArtifactStore;
pub use job::REFUSED_WAIT;

use std::path::Path;

use serde_json::Value;
use splinter_core::digest::Digest;
use splinter_core::training::{
    CurvePoint, HeldOutScore, PolicyGradientStep, PolicyGradientSummary, Selection, TrainingCurve,
};

use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::policy_gradient::{PolicyGradientTune, TrainedPolicyGradient};
use crate::progress::StepClock;
use crate::train::{FineTune, StepReport, Trained};
use crate::train_checks::check_request;

/// The adapter file as the in-process trainer names it in the attempt's directory.
const ADAPTER_FILE: &str = "adapter.safetensors";
/// brain's training record as the in-process trainer names it in the attempt's directory.
const RECORD_FILE: &str = "training.json";

/// Fine-tunes a LoRA on `request.train` as a job of the brain service, and
/// scores base and tuned on the held-out records: [`crate::train::fine_tune`]
/// with the work done by the service.
pub fn fine_tune_served(
    brain: &BrainEndpoints,
    request: &FineTune<'_>,
) -> Result<Trained, PolicyError> {
    let records = check_request(request)?;
    let params = request::fine_tune_params(brain, request)?;
    let clock = StepClock::start();
    let report = run(
        brain,
        request.attempt_dir,
        request.seed,
        &params,
        request.cancel,
        &mut |frame| {
            if let Some(step) = step_of(frame) {
                clock.tick(step.step);
                eprintln!("{}", step_line(frame));
                if let Some(hook) = request.on_step {
                    (hook.0)(&step);
                }
            }
        },
    )?;
    let failed = |reason: String| PolicyError::Train {
        dir: request.attempt_dir.to_path_buf(),
        reason,
    };
    let delivered = deliver(brain, request.attempt_dir, &report)?;
    let score = |name: &str| -> Result<HeldOutScore, PolicyError> {
        serde_json::from_value(report[name].clone())
            .map_err(|_| failed(format!("the service reported no {name}")))
    };
    let mut curve: TrainingCurve = TrainingCurve {
        steps: field(&report, "steps").map_err(failed)?,
        steps_completed: field(&report, "steps_completed").map_err(failed)?,
        eval_every: field(&report, "eval_every").map_err(failed)?,
        patience: field(&report, "patience").map_err(failed)?,
        monitor_records: field(&report, "monitor_records").map_err(failed)?,
        points: serde_json::from_value::<Vec<CurvePoint>>(report["curve"].clone())
            .map_err(|e| failed(format!("the service's monitoring curve: {e}")))?,
        selected_step: field(&report, "selected_step").map_err(failed)?,
        selection: serde_json::from_value::<Selection>(report["selection"].clone())
            .map_err(|e| failed(format!("the service's selection: {e}")))?,
        stopped_early: field(&report, "stopped_early").map_err(failed)?,
        monitoring_skipped: None,
        speed: None,
    };
    curve.speed = clock.finish(request.grad_accum, field(&report, "block").map_err(failed)?);
    let evaluations = match request.keep_evaluations {
        Some(dir) => fetch_evaluations(brain, dir, &report)?,
        None => Vec::new(),
    };
    Ok(Trained {
        adapter: delivered.adapter,
        adapter_digest: delivered.adapter_digest,
        base_digest: delivered.base_digest,
        training_record: delivered.record,
        records,
        block: field(&report, "block").map_err(failed)?,
        base: score("base_score")?,
        tuned: score("tuned_score")?,
        curve,
        evaluations,
    })
}

/// Fine-tunes a LoRA on the rollout records of `request.train` by the clipped
/// policy gradient as a job of the brain service:
/// [`crate::policy_gradient::train_policy_gradient`] with the work done by the
/// service.
pub fn train_policy_gradient_served(
    brain: &BrainEndpoints,
    request: &PolicyGradientTune<'_>,
) -> Result<TrainedPolicyGradient, PolicyError> {
    let (params, records) = request::policy_gradient_params(brain, request)?;
    let report = run(
        brain,
        request.attempt_dir,
        request.seed,
        &params,
        request.cancel,
        &mut |frame| {
            if step_of(frame).is_some() {
                eprintln!("{}", step_line(frame));
            }
        },
    )?;
    let failed = |reason: String| PolicyError::Train {
        dir: request.attempt_dir.to_path_buf(),
        reason,
    };
    let delivered = deliver(brain, request.attempt_dir, &report)?;
    // The series of every step is in the training record (a frame is droppable
    // on the bus and the report is one message), so it is read from there.
    let record: Value =
        serde_json::from_slice(&std::fs::read(&delivered.record).map_err(|source| {
            PolicyError::Io {
                path: delivered.record.clone(),
                source,
            }
        })?)
        .map_err(|e| failed(format!("the training record: {e}")))?;
    let steps: Vec<PolicyGradientStep> =
        serde_json::from_value(record["policy_gradient"]["steps"].clone())
            .map_err(|e| failed(format!("the record's policy-gradient steps: {e}")))?;
    let skipped = report["policy_gradient"]["steps_skipped_nonfinite"]
        .as_u64()
        .map_or(0, |n| usize::try_from(n).unwrap_or(usize::MAX));
    if report["policy_gradient"].is_null() {
        return Err(failed(
            "the service trained the records by imitation and reported no policy-gradient run"
                .into(),
        ));
    }
    Ok(TrainedPolicyGradient {
        adapter: delivered.adapter,
        adapter_digest: delivered.adapter_digest,
        base_digest: delivered.base_digest,
        training_record: delivered.record,
        records,
        block: field(&report, "block").map_err(failed)?,
        statistics: PolicyGradientSummary::from_steps(steps, skipped),
    })
}

/// Runs the job on the endpoints, mapping how it ended to a [`PolicyError`].
fn run(
    brain: &BrainEndpoints,
    attempt_dir: &Path,
    seed: u64,
    params: &Value,
    cancel: Option<&sven_sdk::CancelToken>,
    on_frame: &mut dyn FnMut(&Value),
) -> Result<Value, PolicyError> {
    let first = brain.pin(&seed.to_le_bytes());
    job::run(
        brain.urls(),
        brain.key(),
        brain.refused_wait(),
        first,
        params,
        cancel.map(|c| c as &dyn job::Cancel),
        on_frame,
    )
    .map_err(|e| match e {
        job::JobError::Cancelled => PolicyError::Cancelled {
            dir: attempt_dir.to_path_buf(),
        },
        job::JobError::Failed(reason) => PolicyError::Train {
            dir: attempt_dir.to_path_buf(),
            reason,
        },
        job::JobError::Transport(url, reason) => PolicyError::Endpoint { url, reason },
    })
}

/// The adapter and the training record a job published, fetched into the
/// attempt's directory where the in-process trainer would have written them.
struct Delivered {
    adapter: std::path::PathBuf,
    adapter_digest: String,
    base_digest: String,
    record: std::path::PathBuf,
}

fn deliver(
    brain: &BrainEndpoints,
    attempt_dir: &Path,
    report: &Value,
) -> Result<Delivered, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: attempt_dir.to_path_buf(),
        reason,
    };
    let store = brain.artifact_store()?;
    std::fs::create_dir_all(attempt_dir).map_err(|source| PolicyError::Io {
        path: attempt_dir.to_path_buf(),
        source,
    })?;
    let digest_of = |name: &str| -> Result<Digest, PolicyError> {
        let hex = report[name]
            .as_str()
            .ok_or_else(|| failed(format!("the service reported no {name}")))?;
        Digest::from_content_hex(hex).map_err(|e| failed(format!("{name}: {e}")))
    };
    let adapter = attempt_dir.join(ADAPTER_FILE);
    store.fetch(&digest_of("adapter_digest")?, &adapter)?;
    let record = attempt_dir.join(RECORD_FILE);
    store.fetch(&digest_of("record_digest")?, &record)?;
    // The digest brain serves the adapter under, checked against the file that
    // arrived: what was trained is what was fetched.
    let reported = report["adapter_sha256"]
        .as_str()
        .ok_or_else(|| failed("the service reported no sha256 of the adapter".into()))?;
    let actual = Digest::sha256_of_reader(std::fs::File::open(&adapter).map_err(|source| {
        PolicyError::Io {
            path: adapter.clone(),
            source,
        }
    })?)
    .map_err(|source| PolicyError::Io {
        path: adapter.clone(),
        source,
    })?;
    if actual.as_str() != reported {
        return Err(failed(format!(
            "the adapter fetched hashes to {actual}, not the {reported} the service reported"
        )));
    }
    Ok(Delivered {
        adapter,
        adapter_digest: reported.to_string(),
        base_digest: report["base_digest"]
            .as_str()
            .ok_or_else(|| failed("the service reported no base digest".into()))?
            .to_string(),
        record,
    })
}

/// The adapter of every evaluation, fetched into `dir` as `step-<N>.safetensors`.
fn fetch_evaluations(
    brain: &BrainEndpoints,
    dir: &Path,
    report: &Value,
) -> Result<Vec<(u32, std::path::PathBuf)>, PolicyError> {
    let store = brain.artifact_store()?;
    std::fs::create_dir_all(dir).map_err(|source| PolicyError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    let mut kept = Vec::new();
    for entry in report["evaluations"].as_array().into_iter().flatten() {
        let (Some(step), Some(hex)) = (entry["step"].as_u64(), entry["digest"].as_str()) else {
            continue;
        };
        let step = u32::try_from(step).unwrap_or(u32::MAX);
        let digest = Digest::from_content_hex(hex).map_err(|e| PolicyError::Endpoint {
            url: store.root().display().to_string(),
            reason: format!("an evaluation's digest: {e}"),
        })?;
        let path = dir.join(format!("step-{step}.safetensors"));
        store.fetch(&digest, &path)?;
        kept.push((step, path));
    }
    Ok(kept)
}

/// A report field as the type the caller needs.
fn field<T: serde::de::DeserializeOwned>(report: &Value, name: &str) -> Result<T, String> {
    serde_json::from_value(report[name].clone())
        .map_err(|e| format!("the service's report has no usable {name}: {e}"))
}

/// The step a `step` frame reports.
fn step_of(frame: &Value) -> Option<StepReport> {
    (frame["type"] == "step").then_some(())?;
    Some(StepReport {
        step: u32::try_from(frame["step"].as_u64()?).ok()?,
        steps: u32::try_from(frame["steps"].as_u64()?).ok()?,
        loss: frame["loss"].as_f64().map_or(f32::NAN, |v| v as f32),
    })
}

/// A step frame as the line the in-process trainer would have printed for it.
fn step_line(frame: &Value) -> String {
    let number = |v: &Value| v.as_f64().map(|n| format!("{n:.4}"));
    let mut line = format!(
        "brain train: step {}/{}  loss {}",
        frame["step"],
        frame["steps"],
        number(&frame["loss"]).unwrap_or_else(|| "n/a".into())
    );
    for (name, label) in [
        ("kl", "kl"),
        ("clip_fraction", "clip"),
        ("ratio_mean", "ratio"),
        ("scoring_seconds", "scoring_s"),
    ] {
        if let Some(value) = number(&frame["pg"][name]) {
            line.push_str(&format!("  {label} {value}"));
        }
    }
    if let Some(seconds) = frame["seconds"].as_f64() {
        line.push_str(&format!("  ({seconds:.0}s)"));
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_step_frame_is_a_step_report_and_a_line() {
        let frame = serde_json::json!({
            "type": "step", "step": 3, "steps": 100, "loss": 1.5, "seconds": 93.2,
            "pg": {"kl": 0.0123, "clip_fraction": 0.0, "ratio_mean": 1.001, "scoring_seconds": 0.5}
        });
        let step = step_of(&frame).unwrap();
        assert_eq!((step.step, step.steps, step.loss), (3, 100, 1.5));
        assert_eq!(
            step_line(&frame),
            "brain train: step 3/100  loss 1.5000  kl 0.0123  clip 0.0000  ratio 1.0010  scoring_s 0.5000  (93s)"
        );
        assert!(step_of(&serde_json::json!({"type": "stage", "stage": "loading"})).is_none());
        // A loss brain could not represent (NaN is null in JSON) stays unmeasured.
        let nan = serde_json::json!({"type": "step", "step": 1, "steps": 2, "loss": null});
        assert!(step_of(&nan).unwrap().loss.is_nan());
    }
}
