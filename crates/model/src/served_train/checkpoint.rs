// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that consolidate what a
// model learned into its weights through a shared training service, for its
// clients. If your team needs expertise in full-parameter tuning and adapter
// merging run as service jobs, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Jobs that produce a checkpoint instead of an adapter: `full_train` (every
//! weight of a base trained on chat records) and `merge` (an adapter folded
//! into the weights it targets).
//!
//! A checkpoint is a directory, so it leaves the service as a manifest of
//! per-file digests in the artifact store. The job installs it in the
//! server's model store as `local/<name>` ([`Install`]), where the server
//! serves it and a later job names it as its base; the directory a caller
//! reads it from is `<models>/local/<name>`, which is where a server sharing
//! the model store installed it. A server with a store of its own installed
//! it there, and its files are fetched from the artifact store into the same
//! place, each checked against its digest.

use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use serde_json::{Map, Value};
use splinter_core::digest::Digest;

use super::check::{check_dataset, CheckSpec};
use super::request::{set, source};
use crate::consolidate::{Checkpoint, FullTune};
use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::selection::local_model_name;
use crate::train_checks::MIN_BLOCK;

/// Where a job installs the checkpoint it makes.
#[derive(Clone, Debug)]
pub struct Install<'a> {
    /// The model store as this machine sees it: the one the server was
    /// started over, or where an installed checkpoint's files are fetched
    /// to.
    pub models_dir: &'a Path,
    /// The name it is installed under, `local/<name>`: letters, digits, `.`,
    /// `_` and `-`, not already taken.
    pub name: &'a str,
}

impl Install<'_> {
    /// The directory the checkpoint is read from.
    #[must_use]
    pub fn dir(&self) -> PathBuf {
        self.models_dir.join("local").join(self.name)
    }

    fn checked(&self, attempt_dir: &Path) -> Result<(), PolicyError> {
        let valid = !self.name.is_empty()
            && !self.name.starts_with('.')
            && self
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
        if valid {
            Ok(())
        } else {
            Err(PolicyError::Train {
                dir: attempt_dir.to_path_buf(),
                reason: format!(
                    "{:?} cannot name an installed checkpoint: letters, digits, '.', '_' and '-' only",
                    self.name
                ),
            })
        }
    }
}

/// Trains every weight of `request.model_dir` on `request.train` as a
/// `full_train` job, and installs the checkpoint as `install` says.
///
/// # Errors
/// A record is refused, no artifact store is configured, the job was
/// refused, failed or cancelled, or the checkpoint's files do not arrive.
pub fn full_fine_tune_served(
    brain: &BrainEndpoints,
    request: &FullTune<'_>,
    install: &Install<'_>,
) -> Result<Checkpoint, PolicyError> {
    install.checked(request.attempt_dir)?;
    let base = local_model_name(request.model_dir);
    let spec = CheckSpec::chat(&base, request.thinking, false);
    let mut block = MIN_BLOCK;
    let datasets = std::iter::once(request.train)
        .chain(request.replay.iter().map(PathBuf::as_path))
        .chain(request.weighted_replay.iter().map(|(p, _)| p.as_path()));
    for dataset in datasets {
        let report = check_dataset(
            brain,
            dataset,
            &CheckSpec {
                per_record: false,
                ..spec.clone()
            },
        )?;
        report.refuse_invalid(dataset)?;
        block = block.max(report.longest());
    }
    let mut data = Map::new();
    set(&mut data, "train", source(brain, request.train)?);
    if !request.replay.is_empty() {
        let replay: Result<Vec<Value>, PolicyError> =
            request.replay.iter().map(|p| source(brain, p)).collect();
        set(&mut data, "replay", replay?);
    }
    if !request.weighted_replay.is_empty() {
        let mut weighted = Vec::new();
        for (path, share) in request.weighted_replay {
            let mut entry = source(brain, path)?;
            entry["share"] = serde_json::json!(share);
            weighted.push(entry);
        }
        set(&mut data, "weighted_replay", weighted);
    }
    let mut params = Map::new();
    set(&mut params, "base", base.as_str());
    set(&mut params, "data", Value::Object(data).to_string());
    set(&mut params, "steps", request.steps);
    set(&mut params, "weight_decay", request.weight_decay);
    set(&mut params, "grad_accum", request.grad_accum);
    set(&mut params, "seed", request.seed);
    set(&mut params, "max_block", block);
    set(&mut params, "thinking", request.thinking);
    set(&mut params, "install_as", install.name);
    if let Some(lr) = request.learning_rate {
        set(&mut params, "lr", lr);
    }
    if let Some(share) = request.replay_share {
        set(&mut params, "replay_share", share);
    }
    if let Some(precision) = request.train_precision {
        set(&mut params, "train_precision", precision);
    }
    let report = super::run_action(
        brain,
        request.attempt_dir,
        request.seed,
        "full_train",
        &Value::Object(params),
        request.cancel,
        &mut |frame| {
            if let Some(step) = super::step_of(frame) {
                eprintln!("{}", super::step_line(frame));
                if let Some(hook) = request.on_step {
                    (hook.0)(&step);
                }
            }
        },
    )?;
    received(brain, request.attempt_dir, &report, install)
}

/// Folds `adapter` into the weights of `base` it targets as a `merge` job,
/// and installs the result as `install` says. Cancelled, the job stops at
/// its next tensor and installs nothing.
///
/// # Errors
/// No artifact store is configured, the job was refused, failed or
/// cancelled, or the checkpoint's files do not arrive.
pub fn merge_served(
    brain: &BrainEndpoints,
    base: &Path,
    adapter: &Path,
    install: &Install<'_>,
    cancel: Option<&sven_sdk::CancelToken>,
) -> Result<Checkpoint, PolicyError> {
    let dir = install.dir();
    install.checked(&dir)?;
    let mut params = Map::new();
    set(&mut params, "base", local_model_name(base));
    set(
        &mut params,
        "adapter_digest",
        brain.artifact_store()?.publish(adapter)?.hex(),
    );
    set(&mut params, "install_as", install.name);
    let report = super::run_action(
        brain,
        &dir,
        0,
        "merge",
        &Value::Object(params),
        cancel,
        &mut |_| {},
    )?;
    received(brain, &dir, &report, install)
}

/// One file of a checkpoint's manifest.
#[derive(Deserialize)]
struct ManifestFile {
    name: String,
    digest: String,
}

/// The checkpoint a job reported, read from where it was installed; its
/// files fetched there when this machine does not share the server's store.
fn received(
    brain: &BrainEndpoints,
    attempt_dir: &Path,
    report: &Value,
    install: &Install<'_>,
) -> Result<Checkpoint, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: attempt_dir.to_path_buf(),
        reason,
    };
    let files: Vec<ManifestFile> = serde_json::from_value(report["files"].clone())
        .map_err(|e| failed(format!("the service listed no checkpoint files: {e}")))?;
    let dir = install.dir();
    if !dir.is_dir() {
        let store = brain.artifact_store()?;
        let staging = install
            .models_dir
            .join("local")
            .join(format!(".{}.fetching", install.name));
        let _ = std::fs::remove_dir_all(&staging);
        for file in &files {
            let relative = Path::new(&file.name);
            if relative
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            {
                return Err(failed(format!(
                    "the checkpoint names a file outside itself: {:?}",
                    file.name
                )));
            }
            let dest = staging.join(relative);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).map_err(|source| PolicyError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
            let digest = Digest::from_content_hex(&file.digest)
                .map_err(|e| failed(format!("{}: {e}", file.name)))?;
            store.fetch(&digest, &dest)?;
        }
        std::fs::rename(&staging, &dir).map_err(|source| PolicyError::Io {
            path: dir.clone(),
            source,
        })?;
    }
    let mut weights: Vec<&str> = files
        .iter()
        .map(|f| f.name.as_str())
        .filter(|n| n.starts_with("model") && n.ends_with(".safetensors"))
        .collect();
    weights.sort_unstable();
    let weights = weights
        .first()
        .ok_or_else(|| failed("the checkpoint holds no weights file".into()))?;
    Ok(Checkpoint {
        weights: dir.join(weights),
        dir,
        weights_digest: report["weights_sha256"]
            .as_str()
            .ok_or_else(|| failed("the service reported no weights digest".into()))?
            .to_string(),
        steps_completed: report["steps_completed"]
            .as_u64()
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX)),
        initial_loss: report["initial_loss"].as_f64().map(|v| v as f32),
        final_loss: report["final_loss"].as_f64().map(|v| v as f32),
    })
}
