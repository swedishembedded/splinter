// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The world of generations: a trainer whose adapters, full fine-tunes and
//! merges are cards of the answers their weights hold, so that a base, an
//! adapter on it and a consolidated checkpoint each answer what they were
//! made to, and a scripted model answers from the card of whichever it is.
//!
//! * An adapter's card holds what its base's card held and what the
//!   records it was trained on teach.
//! * A full fine-tune's card holds the same, trained from the base it is
//!   handed.
//! * A merge's card holds the base's and the adapter's.
//!
//! A service in `drops` is lost by every adapter and every full fine-tune,
//! whatever the base held and the records say.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use regex::Regex;
use serde_json::json;
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::training::Regime;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::release::arm_on;
use splinter_pipelines::train::{ModelCheckpoint, TrainPlan, Trainer};

use crate::common::gate::{unscored, BASE_BYTES};
use crate::common::Scripted;

type Answers = BTreeMap<String, String>;

/// A scripted model that answers from what a card says.
fn taught(answers: &Answers) -> Model {
    let answers: Vec<(String, String)> = answers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let scripted = Scripted::new(move |prompt| {
        answers
            .iter()
            .find(|(key, _)| prompt.contains(key.as_str()))
            .map_or("I do not know.".to_string(), |(_, said)| said.clone())
    });
    Model::new(Arc::new(scripted), "scripted/taught")
}

/// The answers a weights file holds, when it is a card; none for any other.
pub fn card_of(weights: &Path) -> Answers {
    std::fs::read(weights)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|card| {
            card["answers"].as_object().map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                    .collect()
            })
        })
        .unwrap_or_default()
}

/// What the trained side of a plan's datasets teaches: for each service named
/// in a record trained on, the assistant text of its first record.
fn taught_by(plan: &TrainPlan) -> Answers {
    let svc = Regex::new(r"Svc(\d+) uses port (\d+)").unwrap();
    let mut answers = Answers::new();
    for dataset in &plan.datasets {
        let text = std::fs::read_to_string(&dataset.path).unwrap_or_default();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let record: serde_json::Value = serde_json::from_str(line).unwrap_or_default();
            if record["metadata"]["split"] == "held_out" {
                continue;
            }
            let said = record["messages"]
                .as_array()
                .and_then(|m| m.iter().rev().find(|m| m["train"] == true))
                .and_then(|m| m["content"].as_str())
                .unwrap_or_default()
                .to_string();
            if let Some(found) = svc.captures_iter(&said).last() {
                answers.entry(format!("Svc{}", &found[1])).or_insert(said);
            }
        }
    }
    answers
}

/// Trains cards. Keeps every plan it was given.
#[derive(Default)]
pub struct Smith {
    /// The adapters trained.
    pub adapters: Mutex<Vec<TrainPlan>>,
    /// The full fine-tunes trained.
    pub fulls: Mutex<Vec<TrainPlan>>,
    /// The merges made: the base and the adapter folded into it.
    pub merges: Mutex<Vec<(PathBuf, PathBuf)>>,
    /// Services lost by every adapter and every full fine-tune.
    pub drops: Mutex<Vec<usize>>,
}

impl Smith {
    fn kept(&self, base: &Path, extra: Answers) -> Answers {
        let mut answers = card_of(&base.join("model.safetensors"));
        answers.extend(extra);
        let drops = self.drops.lock().unwrap().clone();
        answers.retain(|key, _| !drops.iter().any(|n| *key == format!("Svc{n}")));
        answers
    }

    /// The last plan of a kind.
    pub fn last(plans: &Mutex<Vec<TrainPlan>>) -> TrainPlan {
        plans.lock().unwrap().last().cloned().unwrap()
    }
}

fn write_card(dir: &Path, name: &str, answers: &Answers) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(name);
    let card = json!({ "knows": [], "answers": answers });
    std::fs::write(&path, card.to_string()).unwrap();
    path
}

impl Trainer for Smith {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        self.adapters.lock().unwrap().push(plan.clone());
        let answers = self.kept(&plan.base, taught_by(plan));
        let card =
            json!({ "candidate": plan.candidate, "knows": [], "answers": answers }).to_string();
        let adapter = plan.dir.join("adapter.safetensors");
        std::fs::write(&adapter, &card).unwrap();
        let record = plan.dir.join("training.json");
        std::fs::write(&record, r#"{"trainer":"fake"}"#).unwrap();
        let kept = ctx
            .artifacts()
            .put_file(
                &adapter,
                &splinter_store::artifacts::ArtifactSpec::new("adapter", "test-trainer")
                    .with_extension(".safetensors")
                    .with_sha256(),
            )
            .unwrap();
        let path = ctx.artifacts().path(&kept.digest).unwrap();
        ctx.add_model(
            arm_on(ctx.config(), &plan.base, Some(&path)),
            taught(&answers),
        );
        Ok(Trained {
            adapter: path,
            adapter_digest: Digest::sha256_of(card.as_bytes()).to_string(),
            base_digest: Digest::sha256_of(BASE_BYTES).to_string(),
            training_record: record,
            records: plan
                .datasets
                .iter()
                .map(|d| d.manifest.counts.records)
                .sum(),
            block: 0,
            base: unscored(),
            tuned: unscored(),
            curve: splinter_core::training::TrainingCurve::unmonitored(plan.steps, plan.steps),
            evaluations: Vec::new(),
        })
    }

    fn train_preference(
        &self,
        _ctx: &Context,
        _plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        let _ = Regime::Dpo;
        Err(OrchestratorError::Refused("no preferences here".into()))
    }

    fn train_full(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<ModelCheckpoint, OrchestratorError> {
        self.fulls.lock().unwrap().push(plan.clone());
        let answers = self.kept(&plan.base, taught_by(plan));
        Ok(checkpoint(ctx, &plan.dir.join("checkpoint"), &answers, 7))
    }

    fn merge(
        &self,
        ctx: &Context,
        base: &Path,
        adapter: &Path,
        out_dir: &Path,
    ) -> Result<ModelCheckpoint, OrchestratorError> {
        self.merges
            .lock()
            .unwrap()
            .push((base.to_path_buf(), adapter.to_path_buf()));
        let mut answers = card_of(&base.join("model.safetensors"));
        answers.extend(card_of(adapter));
        Ok(checkpoint(ctx, out_dir, &answers, 0))
    }
}

/// A checkpoint directory holding `answers`, served as a base with no adapter.
fn checkpoint(ctx: &Context, dir: &Path, answers: &Answers, steps: u32) -> ModelCheckpoint {
    let weights = write_card(dir, "model.safetensors", answers);
    std::fs::write(
        dir.join("config.json"),
        br#"{"max_position_embeddings": 4096, "model_type": "qwen3"}"#,
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    ctx.add_model(arm_on(ctx.config(), dir, None), taught(answers));
    ModelCheckpoint {
        dir: dir.to_path_buf(),
        weights_digest: Digest::sha256_of(&std::fs::read(&weights).unwrap()).to_string(),
        weights,
        steps_completed: steps,
        initial_loss: (steps > 0).then_some(2.0),
        final_loss: (steps > 0).then_some(0.1),
    }
}

/// Hands `ctx` the model the release `id` serves: what its card answers,
/// on the base its generation sits on.
pub fn serve_release(ctx: &Context, id: &ReleaseId) {
    let stored = ctx.releases().get(id).unwrap();
    match stored.base_release() {
        Some(base) if stored.is_consolidation() => {
            let dir = ctx.generation_base(&base).unwrap();
            let answers = card_of(&dir.join("model.safetensors"));
            ctx.add_model(arm_on(ctx.config(), &dir, None), taught(&answers));
        }
        base => {
            let dir = base.map_or_else(
                || ctx.config().policy_base.clone(),
                |b| ctx.generation_base(&b).unwrap(),
            );
            let answers = card_of(&stored.artifact);
            ctx.add_model(
                arm_on(ctx.config(), &dir, Some(&stored.artifact)),
                taught(&answers),
            );
        }
    }
}

/// The answers the release `id` holds, by service.
pub fn answers_of(ctx: &Context, id: &ReleaseId) -> Answers {
    let stored = ctx.releases().get(id).unwrap();
    card_of(&stored.artifact)
}
