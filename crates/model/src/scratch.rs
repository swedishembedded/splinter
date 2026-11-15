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

use anyhow::bail;
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
    /// The special tokens the tokenizer keeps whole.
    pub specials: Vec<String>,
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
    /// True when the run continued from a saved state.
    pub resumed: bool,
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

/// Brain, reached over its HTTP API at `base_url`.
#[derive(Clone, Debug)]
pub struct ServedBrain {
    /// The server's base URL (`http://host:port/v1`).
    pub base_url: String,
    /// The model's context window, in tokens.
    pub context_tokens: u32,
    /// The key requests are made with.
    pub key: String,
}

/// How long a raw-text completion may take.
const COMPLETION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

impl ModelService for ServedBrain {
    fn make_checkpoint(&self, _request: &CheckpointRequest<'_>) -> anyhow::Result<Checkpoint> {
        Err(needs(
            "brain's from-scratch action over brain/train: train a tokenizer on a text corpus with the \
             template's special tokens, make a random-init checkpoint of a named architecture and its \
             configuration with it, and install it as local/<name>",
        ))
    }

    fn train(
        &self,
        request: &TrainRequest<'_>,
        _on_step: &mut dyn FnMut(u32, f64),
    ) -> anyhow::Result<Trained> {
        match request.objective {
            Objective::Pretrain => Err(needs(
                "brain's raw-text pretraining over brain/train for any architecture that declares it, resumable by job identity, installing local/<name>",
            )),
            Objective::ChatFull => Err(needs(
                "brain's full_train of a from-scratch checkpoint on generic-messages-v2 records (masked chat SFT), \
                 resumable by job identity, installing local/<name>",
            )),
        }
    }

    fn serve(&self, checkpoint: &Checkpoint) -> anyhow::Result<Served> {
        if !checkpoint.model.starts_with("local/") {
            bail!("{} is not an installed local checkpoint", checkpoint.model);
        }
        Ok(Served {
            base_url: self.base_url.clone(),
            model: checkpoint.model.clone(),
            context_tokens: self.context_tokens,
        })
    }

    fn complete(
        &self,
        checkpoint: &Checkpoint,
        prompt: &str,
        max_tokens: u32,
    ) -> anyhow::Result<String> {
        let body = serde_json::json!({ "model": checkpoint.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0 });
        let url = format!("{}/completions", self.base_url.trim_end_matches('/'));
        let reply = crate::endpoint::post_json(&url, &self.key, &body, COMPLETION_TIMEOUT)
            .map_err(|e| anyhow::anyhow!("raw-text completion with {}: {e}", checkpoint.model))?;
        reply["choices"][0]["text"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| {
                anyhow::anyhow!(
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
    fn a_missing_capability_is_named_never_stood_in_for() {
        let brain = ServedBrain {
            base_url: "http://127.0.0.1:1/v1".into(),
            context_tokens: 1024,
            key: "k".into(),
        };
        let dir = std::env::temp_dir();
        let spec = ArchSpec {
            arch: "ouro".into(),
            vocab: 512,
            context: 256,
            config: serde_json::json!({}),
        };
        let format = ChatFormat {
            template: String::new(),
            specials: vec![],
        };
        let err = brain
            .make_checkpoint(&CheckpointRequest {
                model: &spec,
                format: &format,
                tokenizer_corpus: &dir,
                // seed: the refusal comes before any draw; any seed works.
                seed: 1,
                install_as: "x",
                out: &dir,
            })
            .unwrap_err();
        assert!(err.to_string().starts_with("needs "), "{err}");
        let checkpoint = Checkpoint {
            model: "local/x".into(),
            dir: PathBuf::from("c"),
        };
        assert_eq!(brain.serve(&checkpoint).unwrap().model, "local/x");
        assert!(brain
            .serve(&Checkpoint {
                model: "x".into(),
                dir: PathBuf::from("c")
            })
            .is_err());
        assert!(
            brain.complete(&checkpoint, "a", 4).is_err(),
            "an unreachable server is an error, not text"
        );
    }
}
