// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-training services reached over a
// typed API, so a learning pipeline never links the trainer, for its
// clients. If your team needs expertise in training small models from
// scratch behind a service boundary, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Training a model from random weights through brain: what a pipeline asks
//! for, and the implementation over brain's HTTP API.
//!
//! [`ModelService`] is the seam: a tokenizer and a random-init checkpoint of
//! a named architecture, a training run (raw-text pretraining or a full chat
//! fine-tune), a served model and a raw-text completion. Nothing here names
//! an architecture: [`ArchSpec`] carries the architecture's name and its own
//! configuration to brain unread, so one training sequence runs on any
//! architecture brain declares the capabilities for, and brain refuses one
//! that lacks a capability by name. [`ServedBrain`] is the implementation
//! over brain's API; a capability brain does not offer yet fails with
//! `needs <capability>` instead of a stand-in result. A test supplies its
//! own implementation.
//!
//! Every checkpoint is copied into the caller's output directory, so a
//! caller that keeps digests of its outputs covers the weights themselves.

use std::path::{Path, PathBuf};

use std::sync::Mutex;

use anyhow::{anyhow, bail, Context as _};
use serde_json::{json, Value};
use splinter_core::digest::Digest;

use crate::endpoint::BrainEndpoints;
use crate::served_train::ArtifactStore;
use crate::service::{ManagedBrain, ServiceSettings};
use serde::{Deserialize, Serialize};

/// A model to make: an architecture and that architecture's configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchSpec {
    /// The architecture, as brain names it (`ouro`, `qwen3`).
    pub arch: String,
    /// Tokenizer vocabulary, special tokens included.
    pub vocab: u32,
    /// Longest sequence, in tokens.
    pub context: u32,
    /// The architecture's own configuration (widths, layers, loop passes),
    /// handed to brain as it is.
    pub config: serde_json::Value,
}

/// The chat format a from-scratch checkpoint carries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatFormat {
    /// Its chat template, over OpenAI-shaped `messages` and `tools`.
    pub template: String,
    /// The special tokens the tokenizer keeps whole, in vocabulary order.
    pub specials: Vec<String>,
    /// The token that ends a turn (one of `specials`).
    pub eos: String,
    /// The padding token, when there is one (one of `specials`).
    pub pad: Option<String>,
}

/// One training run's budget.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    /// Optimiser steps.
    pub steps: u32,
    /// Sequences per step.
    pub batch: u32,
    /// Peak learning rate.
    pub lr: f64,
    /// Tokens per training row: the pretraining window, the longest
    /// fine-tuning record.
    pub block: u32,
}

/// Where a served model is reached.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Served {
    /// The server's base URL (`http://host:port/v1`).
    pub base_url: String,
    /// The model id it serves the checkpoint as.
    pub model: String,
    /// The model's context window, in tokens.
    pub context_tokens: u32,
    /// The key requests to it are made with.
    pub key: String,
}

/// A checkpoint the service holds, by the name it is installed under.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The model id it is served as (`local/<name>`).
    pub model: String,
    /// The directory in the run directory holding a copy of its files.
    pub dir: PathBuf,
}

/// The tokenizer and random-init checkpoint to make.
#[derive(Clone, Debug)]
pub struct CheckpointRequest<'a> {
    /// The architecture and its configuration.
    pub model: &'a ArchSpec,
    /// The chat format it carries.
    pub format: &'a ChatFormat,
    /// The text the tokenizer is trained on (JSON lines of `{"text"}`).
    pub tokenizer_corpus: &'a Path,
    /// The seed of the weights' initialisation and the tokenizer's training.
    pub seed: u64,
    /// The name it is installed under.
    pub install_as: &'a str,
    /// The run's durable identity (brain's `resume_key`).
    pub identity: &'a str,
    /// Where its files are copied to.
    pub out: &'a Path,
}

/// What a training run optimises.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Objective {
    /// Next-token prediction on raw text, every token supervised.
    Pretrain,
    /// Every weight fine-tuned on chat records, only the supervised messages
    /// trained.
    ChatFull,
}

/// One training run.
#[derive(Clone, Debug)]
pub struct TrainRequest<'a> {
    /// The objective.
    pub objective: Objective,
    /// The checkpoint training starts from.
    pub base: &'a Checkpoint,
    /// The training data.
    pub data: &'a Path,
    /// Data scored during the run and never trained on, when there is any.
    pub held_out: Option<&'a Path>,
    /// Steps, batch and learning rate.
    pub budget: &'a Budget,
    /// The seed of the batch order.
    pub seed: u64,
    /// The name the result is installed under.
    pub install_as: &'a str,
    /// The run's durable identity: a run resubmitted under the same identity
    /// continues from its last saved state instead of starting over.
    pub identity: &'a str,
    /// Where the result's files are copied to.
    pub out: &'a Path,
}

/// What a training run produced.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trained {
    /// The checkpoint.
    pub checkpoint: Checkpoint,
    /// `(step, loss)` as the trainer reported them.
    pub losses: Vec<(u32, f64)>,
    /// True when the run continued from a saved state or was answered from
    /// its record.
    pub resumed: bool,
    /// The job's whole report.
    pub report: Value,
}

/// What the stages ask of the model service.
pub trait ModelService {
    /// Trains a tokenizer, makes a random-init checkpoint with it and
    /// installs it.
    ///
    /// # Errors
    /// The service cannot, saying which capability is missing.
    fn make_checkpoint(&self, request: &CheckpointRequest<'_>) -> anyhow::Result<Checkpoint>;

    /// Runs (or continues) a training run and installs its result.
    ///
    /// # Errors
    /// The service cannot, or the run failed.
    fn train(
        &self,
        request: &TrainRequest<'_>,
        on_step: &mut dyn FnMut(u32, f64),
    ) -> anyhow::Result<Trained>;

    /// Where an agent reaches `checkpoint` served with its template and
    /// tool calls.
    ///
    /// # Errors
    /// The service cannot serve it.
    fn serve(&self, checkpoint: &Checkpoint) -> anyhow::Result<Served>;

    /// `checkpoint`'s greedy continuation of raw text `prompt`.
    ///
    /// # Errors
    /// The service cannot complete with it.
    fn complete(
        &self,
        checkpoint: &Checkpoint,
        prompt: &str,
        max_tokens: u32,
    ) -> anyhow::Result<String>;
}

/// The error of a capability brain's API does not offer yet.
#[must_use]
pub fn needs(capability: &str) -> anyhow::Error {
    anyhow::anyhow!("needs {capability}")
}

/// How long a raw-text completion may take.
const COMPLETION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// How the brain service is reached.
enum Connection {
    /// A running server the caller gave; it cannot be restarted from here.
    Given(BrainEndpoints),
    /// A server this process starts when it is first needed, and restarts
    /// when a checkpoint it installed must be served (brain scans its model
    /// store at start). Every start is checked to schedule on `device`.
    Managed {
        settings: Box<ServiceSettings>,
        brain: Option<ManagedBrain>,
        cancel: sven_sdk::CancelToken,
        device: Option<String>,
    },
}

/// Brain, reached over its HTTP API: training jobs on `brain/train`
/// (`init_model`, `pretrain`, `full_train`), artifacts by digest, the
/// installed checkpoints served over `/v1/chat/completions`.
pub struct ServedBrain {
    connection: Mutex<Connection>,
    context_tokens: u32,
}

impl std::fmt::Debug for ServedBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServedBrain")
            .field("context_tokens", &self.context_tokens)
            .finish_non_exhaustive()
    }
}

/// The checkpoint a job's report names, its files fetched into `out`.
fn installed(report: &Value, store: &ArtifactStore, out: &Path) -> anyhow::Result<Checkpoint> {
    let name = report["installed_as"]
        .as_str()
        .ok_or_else(|| anyhow!("the job installed no checkpoint: {report}"))?;
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    for file in report["files"].as_array().into_iter().flatten() {
        let (Some(rel), Some(hex)) = (file["name"].as_str(), file["digest"].as_str()) else {
            bail!("the job's report lists a file without a name and digest: {file}");
        };
        if Path::new(rel)
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            bail!("the job's report names a file outside its checkpoint: {rel}");
        }
        let dest = out.join(rel);
        if let Some(dir) = dest.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let digest = Digest::from_content_hex(hex).map_err(|e| anyhow!("file {rel}: {e}"))?;
        store
            .fetch(&digest, &dest)
            .map_err(|e| anyhow!("fetching {rel}: {e}"))?;
    }
    let model = if name.starts_with("local/") {
        name.to_string()
    } else {
        format!("local/{name}")
    };
    Ok(Checkpoint {
        model,
        dir: out.to_path_buf(),
    })
}

/// Checks that the server at `endpoints` schedules on device `id` (`gpu3`),
/// as its `/v1/devices` lists them.
fn require_device(endpoints: &BrainEndpoints, id: &str) -> anyhow::Result<()> {
    let url = endpoints
        .urls()
        .first()
        .ok_or_else(|| anyhow!("brain has no endpoint"))?;
    let devices = crate::endpoint::get_json(
        &format!("{}/devices", url.trim_end_matches('/')),
        endpoints.key(),
    )
    .map_err(|e| anyhow!("asking brain for its devices: {e}"))?;
    let listed: Vec<&str> = devices["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|d| d["id"].as_str())
        .collect();
    if listed.contains(&id) {
        Ok(())
    } else {
        bail!("brain does not schedule on {id} (it lists {listed:?}): no usable card, so nothing is trained")
    }
}

/// The seed as brain takes it: an integer JSON represents exactly, below
/// 2^53; the run's seed is folded into that range.
fn job_seed(seed: u64) -> u64 {
    (seed ^ (seed >> 53)) & ((1 << 53) - 1)
}

/// A step frame's step and loss.
fn step_of(frame: &Value) -> Option<(u32, f64)> {
    Some((
        u32::try_from(frame["step"].as_u64()?).ok()?,
        frame["loss"].as_f64()?,
    ))
}

impl ServedBrain {
    /// A server the caller runs (`brain serve --training --artifact-store`),
    /// with its artifact store reachable as `endpoints` says.
    #[must_use]
    pub fn given(endpoints: BrainEndpoints, context_tokens: u32) -> Self {
        Self {
            connection: Mutex::new(Connection::Given(endpoints)),
            context_tokens,
        }
    }

    /// A server started from `settings` when a call first needs it (so a
    /// caller whose work is all done starts none), owned, and stopped when
    /// this is dropped. With `device` (`gpu3`), every start is checked to
    /// schedule on that device, as its `/v1/devices` lists it: work that must
    /// run on a card fails before it begins instead of running elsewhere.
    #[must_use]
    pub fn managed(settings: ServiceSettings, context_tokens: u32, device: Option<&str>) -> Self {
        Self {
            connection: Mutex::new(Connection::Managed {
                settings: Box::new(settings),
                brain: None,
                cancel: sven_sdk::CancelToken::new(),
                device: device.map(String::from),
            }),
            context_tokens,
        }
    }

    /// Starts the managed server and checks its device.
    fn start(
        settings: &ServiceSettings,
        cancel: &sven_sdk::CancelToken,
        device: Option<&str>,
    ) -> anyhow::Result<ManagedBrain> {
        let brain =
            ManagedBrain::start(settings, cancel).map_err(|e| anyhow!("starting brain: {e}"))?;
        if let Some(id) = device {
            require_device(brain.endpoints(), id)?;
        }
        Ok(brain)
    }
    fn endpoints(&self) -> anyhow::Result<BrainEndpoints> {
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| anyhow!("the brain connection is poisoned"))?;
        match &mut *connection {
            Connection::Given(endpoints) => Ok(endpoints.clone()),
            Connection::Managed {
                settings,
                brain,
                cancel,
                device,
            } => {
                if brain.is_none() {
                    *brain = Some(Self::start(settings, cancel, device.as_deref())?);
                }
                brain
                    .as_ref()
                    .map(|b| b.endpoints().clone())
                    .ok_or_else(|| anyhow!("the managed brain did not start"))
            }
        }
    }

    /// Restarts a managed server, so it scans its model store again; a given
    /// server cannot be.
    fn rescan(&self, model: &str) -> anyhow::Result<()> {
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| anyhow!("the brain connection is poisoned"))?;
        match &mut *connection {
            Connection::Given(endpoints) => bail!(
                "{} does not serve {model}: restart that server so it scans its model store",
                endpoints.urls().join(", ")
            ),
            Connection::Managed {
                settings,
                brain,
                cancel,
                device,
            } => {
                drop(brain.take());
                *brain = Some(
                    Self::start(settings, cancel, device.as_deref())
                        .context("restarting brain so it scans its model store")?,
                );
                Ok(())
            }
        }
    }

    fn job(
        &self,
        action: &str,
        params: &Value,
        out: &Path,
        on_step: &mut dyn FnMut(u32, f64),
    ) -> anyhow::Result<(Value, Checkpoint)> {
        let endpoints = self.endpoints()?;
        let report = crate::served_train::run_action(
            &endpoints,
            out,
            0,
            action,
            params,
            None,
            &mut |frame| {
                if let Some((step, loss)) = step_of(frame) {
                    on_step(step, loss);
                }
            },
        )
        .map_err(|e| anyhow!("brain/train {action}: {e}{}", self.server_output()))?;
        let store = endpoints.artifact_store().map_err(|e| anyhow!("{e}"))?;
        let checkpoint = installed(&report, store, out)?;
        Ok((report, checkpoint))
    }

    /// The managed server's last lines, for a failure the server explains in
    /// its own output; empty for a given server.
    fn server_output(&self) -> String {
        let Ok(connection) = self.connection.lock() else {
            return String::new();
        };
        match &*connection {
            Connection::Managed {
                brain: Some(brain), ..
            } => {
                let recent = brain.recent_output();
                if recent.trim().is_empty() {
                    String::new()
                } else {
                    format!("\nthe brain server's last lines:\n{recent}")
                }
            }
            _ => String::new(),
        }
    }

    fn publish(&self, file: &Path) -> anyhow::Result<String> {
        let endpoints = self.endpoints()?;
        let store = endpoints.artifact_store().map_err(|e| anyhow!("{e}"))?;
        let digest = store
            .publish(file)
            .map_err(|e| anyhow!("publishing {}: {e}", file.display()))?;
        Ok(digest.hex().to_string())
    }
}

impl ModelService for ServedBrain {
    fn make_checkpoint(&self, request: &CheckpointRequest<'_>) -> anyhow::Result<Checkpoint> {
        let corpus = self.publish(request.tokenizer_corpus)?;
        let mut tokenizer = json!({
            "vocab_size": request.model.vocab,
            "special_tokens": request.format.specials,
            "eos_token": request.format.eos,
        });
        if let Some(pad) = &request.format.pad {
            tokenizer["pad_token"] = json!(pad);
        }
        let params = json!({
            "arch": request.model.arch,
            "config": request.model.config.to_string(),
            "tokenizer": tokenizer.to_string(),
            "chat_template": request.format.template,
            "corpus_digest": corpus,
            "seed": job_seed(request.seed),
            "install_as": request.install_as,
            "resume_key": request.identity,
        });
        Ok(self
            .job("init_model", &params, request.out, &mut |_, _| {})?
            .1)
    }

    fn train(
        &self,
        request: &TrainRequest<'_>,
        on_step: &mut dyn FnMut(u32, f64),
    ) -> anyhow::Result<Trained> {
        let mut data = json!({ "train": { "digest": self.publish(request.data)? } });
        let base = request.base.model.as_str();
        let budget = request.budget;
        let (action, params) = match request.objective {
            Objective::Pretrain => {
                let mut params = json!({
                    "base": base, "steps": budget.steps, "batch": budget.batch, "block": budget.block,
                    "lr": budget.lr, "seed": job_seed(request.seed),
                });
                if let Some(held_out) = request.held_out {
                    data["held_out"] = json!({ "digest": self.publish(held_out)? });
                    params["eval_every"] = json!((budget.steps / 4).max(1));
                }
                ("pretrain", params)
            }
            Objective::ChatFull => (
                "full_train",
                json!({
                    "base": base, "steps": budget.steps, "grad_accum": budget.batch, "lr": budget.lr,
                    "max_block": budget.block, "thinking": false, "seed": job_seed(request.seed),
                }),
            ),
        };
        let mut params = params;
        params["data"] = json!(data.to_string());
        params["install_as"] = json!(request.install_as);
        params["resume_key"] = json!(request.identity);
        let mut losses = Vec::new();
        let (report, checkpoint) = self.job(action, &params, request.out, &mut |step, loss| {
            losses.push((step, loss));
            on_step(step, loss);
        })?;
        if losses.is_empty() {
            // A completed run answered from its record streams no steps; its
            // evaluations, when it kept any, are the curve.
            losses = report["evaluations"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|e| {
                    Some((
                        u32::try_from(e["step"].as_u64()?).ok()?,
                        e["train_loss"].as_f64()?,
                    ))
                })
                .collect();
        }
        let resumed = report["resume"]["state"] != "fresh";
        Ok(Trained {
            checkpoint,
            losses,
            resumed,
            report,
        })
    }

    fn serve(&self, checkpoint: &Checkpoint) -> anyhow::Result<Served> {
        if !checkpoint.model.starts_with("local/") {
            bail!("{} is not an installed local checkpoint", checkpoint.model);
        }
        let listed = |endpoints: &BrainEndpoints| -> anyhow::Result<bool> {
            let cards = endpoints.list(0).map_err(|e| anyhow!("{e}"))?;
            Ok(cards.iter().any(|c| c.id == checkpoint.model))
        };
        if !listed(&self.endpoints()?)? {
            self.rescan(&checkpoint.model)?;
            if !listed(&self.endpoints()?)? {
                bail!(
                    "brain does not serve {} after scanning its model store",
                    checkpoint.model
                );
            }
        }
        let endpoints = self.endpoints()?;
        let base_url = endpoints
            .urls()
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("brain has no endpoint"))?;
        Ok(Served {
            base_url,
            model: checkpoint.model.clone(),
            context_tokens: self.context_tokens,
            key: endpoints.key().to_string(),
        })
    }

    fn complete(
        &self,
        checkpoint: &Checkpoint,
        prompt: &str,
        max_tokens: u32,
    ) -> anyhow::Result<String> {
        let served = self.serve(checkpoint)?;
        let endpoints = self.endpoints()?;
        let body = json!({ "model": checkpoint.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0 });
        let url = format!("{}/completions", served.base_url.trim_end_matches('/'));
        let reply = crate::endpoint::post_json(&url, endpoints.key(), &body, COMPLETION_TIMEOUT)
            .map_err(|e| anyhow!("raw-text completion with {}: {e}", checkpoint.model))?;
        reply["choices"][0]["text"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| {
                anyhow!(
                    "raw-text completion with {}: no text in {reply}",
                    checkpoint.model
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_installs_only_files_inside_its_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("store")).unwrap();
        let store = ArtifactStore::open(&dir.path().join("store")).unwrap();
        let src = dir.path().join("w");
        std::fs::write(&src, b"weights").unwrap();
        let digest = store.publish(&src).unwrap();
        let report = json!({ "installed_as": "tiny", "files": [{ "name": "model.safetensors", "digest": digest.hex() }] });
        let made = installed(&report, &store, &dir.path().join("out")).unwrap();
        assert_eq!(made.model, "local/tiny");
        assert_eq!(
            std::fs::read(dir.path().join("out/model.safetensors")).unwrap(),
            b"weights"
        );
        let escaping = json!({ "installed_as": "tiny", "files": [{ "name": "../x", "digest": digest.hex() }] });
        assert!(installed(&escaping, &store, &dir.path().join("out2")).is_err());
        assert!(
            installed(&json!({}), &store, &dir.path().join("out3")).is_err(),
            "a report that installed nothing"
        );
    }
}
